import torch, argparse, sys
from torch import nn, optim
from torchvision import datasets, transforms, models
from my_model import MyClassifier, save_checkpoint

parser = argparse.ArgumentParser(description='IMAGE CLASSIFICATION TRAINER, PLEASE PROVIDE DIRECTORY WITH IMAGE DATA')
parser.add_argument('data_dir', type=str, default='flowers', help='Directory of input image data')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save trained model')
parser.add_argument('--arch', type=str, default='densenet121', help='Model architecture options: densenet121 / vgg11 / alexnet')
parser.add_argument('--hidden_units', type=int, default=256, help='Number of perceptron units in hidden layer')
parser.add_argument('--epochs', type=int, default=2, help='Number of iterations')
parser.add_argument('--learning_rate', type=float, default=0.004, help='Learning rate per iteration')
parser.add_argument('--gpu', action='store_true', default=False, help='Set GPU required')

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
model_arch = args.arch
n_hidden = args.hidden_units
epochs = args.epochs
learn_rate = args.learning_rate
gpu = args.gpu

traindir = data_dir + '/train'
validdir = data_dir + '/valid'
testdir = data_dir + '/test'

traintransform = transforms.Compose([transforms.RandomRotation(30),
									transforms.RandomResizedCrop(224),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize([0.485, 0.456, 0.406],
														[0.229, 0.224, 0.225])])
validtransform = transforms.Compose([transforms.Resize(255),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									transforms.Normalize([0.485, 0.456, 0.406],
														[0.229, 0.224, 0.225])])

trainset = datasets.ImageFolder(traindir, transform=traintransform)
validset = datasets.ImageFolder(validdir, transform=validtransform)
testset = datasets.ImageFolder(testdir, transform=validtransform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=64)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)

if model_arch == 'densenet121':
	model = models.densenet121(pretrained=True)
	n_input = 1024
elif model_arch == 'vgg11':
	model = models.vgg11(pretrained=True)
	n_input = 25088
elif model_arch == 'alexnet':
	model = models.alexnet(pretrained=True)
	n_input == 9216
else:
	print('Error: Architecture not supported... Available options: densenet121 / vgg11 / alexnet')
	sys.exit()
for param in model.parameters():
	param.requires_grad = False
model.classifier = MyClassifier(n_input=n_input, n_hidden=n_hidden)

device = 'cuda' if gpu else 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

print_steps = 10
train_losses = []
valid_losses = []
train_loss_sum = 0
for e in range(epochs):
	for step, (images, labels) in enumerate(trainloader):
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		log_prob = model.forward(images)
		loss = criterion(log_prob, labels)
		loss.backward()
		optimizer.step()
		train_loss_sum += loss.item()

		if (step+1) % print_steps == 0:
			train_losses.append(train_loss_sum / (e*len(trainloader) + step+1))

			valid_loss_sum = 0
			valid_equality = torch.Tensor()
			with torch.no_grad():
				model.eval()
				for images, labels in validloader:
					images, labels = images.to(device), labels.to(device)
					log_prob = model.forward(images)
					loss = criterion(log_prob, labels)
					valid_loss_sum += loss.item()
					prob = torch.exp(log_prob)
					top_prob, top_class = prob.topk(1, dim=1)
					equality = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
					valid_equality = torch.cat([valid_equality, equality], dim=0)
				model.train()
			valid_losses.append(valid_loss_sum / len(validloader))
			accuracy = torch.mean(valid_equality).item()

			print(f'Epoch: {e+1}/{epochs}, Batch Step: {step+1}/{len(trainloader)}, Training Loss: {round(train_losses[-1], 2)}, Validation Loss: {round(valid_losses[-1], 2)}, Accuracy: {round(accuracy*100)}%')
			if len(valid_losses) > 1:
				if valid_losses[-1] > valid_losses[-2]:
					print('Training failed... Loss increasing!')
					break
				elif valid_losses[-1] > 0.995*valid_losses[-2]:
					print('Training stopped... Gradient vanishing!')
					break
			if accuracy > 0.8:
				print('Training succeed... Accuracy is sufficiently high!')
				break
	else:
		continue
	break

save_checkpoint(model, trainset, optimizer, save_dir, model_arch=model_arch, n_hidden=n_hidden)
print('Trained model saved to: {}'.format(save_dir))
