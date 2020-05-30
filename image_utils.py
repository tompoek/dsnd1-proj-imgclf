from PIL import Image
import numpy as np
import torch

def process_image(image_path):
	''' Scales, crops, and normalizes a PIL image for a PyTorch model,
		returns a Tensor
	'''
	image = Image.open(image_path)
	if image.size[0] < image.size[1]:
		image.thumbnail((256, 256*image.size[1]/image.size[0]))
	else:
		image.thumbnail((256*image.size[0]/image.size[1], 256))
	row_crop = (image.size[0]-224)/2
	col_crop = (image.size[1]-224)/2
	image = image.crop((row_crop, col_crop, image.size[0]-row_crop, image.size[1]-col_crop))
	image = np.array(image).transpose((2,0,1))
	image = np.clip(image/255, 0, 1)
	mean = np.array([0.485, 0.456, 0.406])[:,None,None]
	std = np.array([0.229, 0.224, 0.225])[:,None,None]
	image = (image - mean) / std
	image = image[None,:,:,:]
	image = torch.from_numpy(image).type(torch.FloatTensor)
	return image
