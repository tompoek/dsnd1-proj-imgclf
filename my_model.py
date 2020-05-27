import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models

class MyClassifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(1024, 256)
		self.fc2 = nn.Linear(256, 102)
		self.drop = nn.Dropout(p=0.4)
	def forward(self, x):
		x = x.view(x.shape[0], -1)
		x = self.drop(F.relu(self.fc1(x)))
		x = F.log_softmax(self.fc2(x), dim=1)
		return x

def save_checkpoint(model, dataset, optimizer, filepath):
	checkpoint = {'class_to_idx': dataset.class_to_idx,
				  'model_state_dict': model.state_dict(),
				  'optim_state_dict': optimizer.state_dict()}
	torch.save(checkpoint, filepath)
	return

def load_checkpoint(filepath):
	model = models.densenet121()
	model.classifier = MyClassifier()
	checkpoint = torch.load(filepath)
	model.class_to_idx = checkpoint['class_to_idx']
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer = optim.Adam(model.classifier.parameters(), lr=0.004)
	optimizer.load_state_dict(checkpoint['optim_state_dict'])
	for param in model.parameters():
		param.requires_grad = False
	return model, optimizer
