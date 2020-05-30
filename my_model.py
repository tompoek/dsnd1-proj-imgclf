import torch, sys
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models

class MyClassifier(nn.Module):
	def __init__(self, n_input=1024, n_hidden=256, n_output=102):
		super().__init__()
		self.fc1 = nn.Linear(n_input, n_hidden)
		self.fc2 = nn.Linear(n_hidden, n_output)
		self.drop = nn.Dropout(p=0.4)
	def forward(self, x):
		x = x.view(x.shape[0], -1)
		x = self.drop(F.relu(self.fc1(x)))
		x = F.log_softmax(self.fc2(x), dim=1)
		return x

def save_checkpoint(model, dataset, optimizer, filepath, model_arch='densenet121', n_hidden=256):
	checkpoint = {'model_arch': model_arch,
				'n_hidden': n_hidden,
				'class_to_idx': dataset.class_to_idx,
				'model_state_dict': model.state_dict(),
				'optim_state_dict': optimizer.state_dict()}
	torch.save(checkpoint, filepath)
	return

def load_checkpoint(filepath):
	checkpoint = torch.load(filepath)
	model_arch = checkpoint['model_arch']
	n_hidden = checkpoint['n_hidden']
	if model_arch == 'densenet121':
		model = models.densenet121()
		n_input = 1024
	elif model_arch == 'vgg11':
		model = models.vgg11()
		n_input = 25088
	elif model_arch == 'alexnet':
		model = models.alexnet()
		n_input == 9216
	else:
		print('Error: Architecture not supported... Available options: densenet121 / vgg11 / alexnet')
		sys.exit()
	model.classifier = MyClassifier(n_input=n_input, n_hidden=n_hidden)
	model.class_to_idx = checkpoint['class_to_idx']
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer = optim.Adam(model.classifier.parameters(), lr=0.004)
	optimizer.load_state_dict(checkpoint['optim_state_dict'])
	for param in model.parameters():
		param.requires_grad = False
	return model, optimizer
