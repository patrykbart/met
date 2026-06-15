import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models




OUTPUT_DIM = {
	'resnet18'              :  512,
	'resnet50'              : 2048,
	'r18_sw-sup'             : 512,
	'dinov3_vitl16'         : 1024,
	'dinov3_vit7b16'        : 4096,
}

DINOV3_HF = {
	'dinov3_vitl16'  : 'facebook/dinov3-vitl16-pretrain-lvd1689m',
	'dinov3_vit7b16' : 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
}



class GeM(nn.Module):
	'''Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
	'''
	
	def __init__(self, p=3, eps=1e-6):
		super(GeM,self).__init__()
		self.p = Parameter(torch.ones(1)*p)
		self.eps = eps

	def forward(self, x):
		return gem(x, p=self.p, eps=self.eps)
		
	def __repr__(self):
		return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'



def gem(x, p=3, eps=1e-6):
	'''Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
	'''

	return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)



class DINOv3Trunk(nn.Module):
	'''Wraps a HuggingFace DINOv3 ViT so it plugs into Embedder exactly like a conv
	trunk: forward(img) -> (B, D, g, g), consumed by the existing pool + projector
	unchanged. readout='cls' (default) returns the global CLS token as (B,D,1,1) -- the
	representation the EXP-6 reproduction used (full Met GAP 48.16); readout='patch'
	returns the mean-poolable patch grid. The input is resized to img_size (a multiple
	of 16) so the patch grid g = img_size//16 is fixed. Optional LoRA adapters (peft)
	for parameter-efficient fine-tuning; otherwise the backbone is frozen (head-only).
	transformers/peft are imported lazily so backbone.py still imports in the R18 venv.
	'''

	def __init__(self, hf_name, img_size = 512, lora = False, lora_r = 16, lora_alpha = 32, readout = 'cls'):

		super(DINOv3Trunk, self).__init__()

		from transformers import AutoModel

		self.patch = 16
		self.img_size = (int(img_size) // self.patch) * self.patch   # must be a multiple of patch
		self.readout = readout   # 'cls' = global CLS token (EXP-6 reproduction, full GAP 48.16); 'patch' = mean-pooled patch grid

		model = AutoModel.from_pretrained(hf_name)

		if lora:
			from peft import LoraConfig, get_peft_model
			model = get_peft_model(model, LoraConfig(
				r = lora_r, lora_alpha = lora_alpha, lora_dropout = 0.0,
				bias = "none", target_modules = "all-linear"))
			print("DINOv3 + LoRA (r={}, alpha={}, all-linear)".format(lora_r, lora_alpha))
			model.print_trainable_parameters()
		else:
			for p in model.parameters():
				p.requires_grad = False
			print("DINOv3 frozen (head-only adaptation)")

		self.model = model

	def forward(self, img):

		if img.shape[-1] != self.img_size or img.shape[-2] != self.img_size:
			img = F.interpolate(img, size = (self.img_size, self.img_size),
								mode = "bilinear", align_corners = False)

		tok = self.model(pixel_values = img).last_hidden_state   # (B, prefix + P, D)

		if self.readout == 'cls':
			# global CLS token (index 0), the representation the EXP-6 DINOv3 reproduction used
			# (full Met GAP 48.16). Shape (B,D,1,1) so the Embedder's AdaptiveAvgPool2d is a no-op.
			cls = tok[:, 0, :]
			return cls[:, :, None, None]

		g = self.img_size // self.patch
		patches = tok[:, -(g * g):, :]                           # last g*g tokens = patch grid
		B, _, D = patches.shape

		return patches.transpose(1, 2).reshape(B, D, g, g)



class Embedder(nn.Module):
	'''Class that implements a descriptor extractor as a (fully convolutional backbone -> pooling -> l2 normalization).
	Optionally followed by a FC layer (fully convolutional backbone -> pooling -> l2 normalization -> FC -> l2 normalization)
	that can be initialized with the result of PCAw.
	'''

	def __init__(self,architecture,gem_p = 3,pretrained_flag = True,projector = False,init_projector = None,lora = False,dino_img_size = 512,dino_readout = 'cls'):
		'''The FC layer is called projector.
		'''

		super(Embedder, self).__init__()

		if architecture == "r18_sw-sup":
			#r18 facebook pretrained model (https://github.com/facebookresearch/semi-supervised-ImageNet1K-models)

			network = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models','resnet18_swsl')
			self.backbone = nn.Sequential(*list(network.children())[:-2])

		elif architecture.startswith("dinov3"):
			#frozen DINOv3 ViT (optionally +LoRA); patch tokens -> (B,D,g,g) feature map
			self.backbone = DINOv3Trunk(DINOV3_HF[architecture], img_size = dino_img_size, lora = lora, readout = dino_readout)

		else:

			#load the base model from PyTorch's pretrained models (imagenet pretrained)
			network = getattr(models,architecture)(pretrained=pretrained_flag)

			#keep only the convolutional layers, ends with relu to get non-negative descriptors
			if architecture.startswith('resnet'):
				self.backbone = nn.Sequential(*list(network.children())[:-2])

			elif architecture.startswith('alexnet'):
				self.backbone = nn.Sequential(*list(network.features.children())[:-1])

		#spatial pooling layer (mean for DINOv3 -- GeM's clamp(min=eps) is invalid for signed ViT features)
		if architecture.startswith("dinov3"):
			self.pool = nn.AdaptiveAvgPool2d(1)
		else:
			self.pool = GeM(p = gem_p)

		#normalize on the unit-hypershpere
		#self.norm = L2N()
		self.norm = F.normalize

		#information about the network
		self.meta = {
			'architecture' : architecture, 
			'pooling' : "gem",
			'mean' : [0.485, 0.456, 0.406], #imagenet statistics for imagenet pretrained models
			'std' : [0.229, 0.224, 0.225],
			'outputdim' : OUTPUT_DIM[architecture],
		}


		if projector:
			print("using FC layer in the backbone")
			self.projector = nn.Linear(self.meta['outputdim'],self.meta['outputdim'],bias = True)

			if init_projector is not None:

				print("initialising the backbone's project layer")

				self.projector.weight.data = torch.transpose(torch.Tensor(init_projector[1]),0,1)
				self.projector.bias.data = -torch.matmul(torch.Tensor(init_projector[0]),torch.Tensor(init_projector[1]))

		else:
			self.projector = None


	def forward(self, img):
		'''
		Output has shape: batch size x descriptor dimension
		'''

		x = self.norm(self.pool(self.backbone(img))).squeeze(-1).squeeze(-1)

		if self.projector is None:
			return x

		else:
			return self.norm(self.projector(x))



def extract_ss(net, input):
	'''Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
	'''

	return net(input).cpu().data.squeeze()



def extract_ms(net, input, ms, msp):
	'''Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
	'''
	
	v = torch.zeros(net.meta['outputdim'])
	
	for s in ms:

		if s == 1:
			input_t = input.clone()
		
		else:    
			input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
		
		v += net(input_t).pow(msp).cpu().data.squeeze()
		
	v /= len(ms)
	v = v.pow(1./msp)
	v /= v.norm()

	return v