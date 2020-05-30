import torch, argparse, json
from my_model import load_checkpoint
from image_utils import process_image

parser = argparse.ArgumentParser(description='IMAGE CLASSIFIER, PLEASE PROVIDE PATH TO INPUT IMAGE AND MODEL CHECKPOINT')
parser.add_argument('input', type=str, default='flowers/train/7/image_07200.jpg', help='Path of input image')
parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='Path of model checkpoint')
parser.add_argument('--top_k', type=int, default=3, help='Number of top classes for me to guess')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path of category names mapping')
parser.add_argument('--gpu', action='store_true', default=False, help='Set GPU required')

args = parser.parse_args()
image_path = args.input
checkpoint_path = args.checkpoint
top_k = args.top_k
cat_dict = args.category_names
gpu = args.gpu

device = 'cuda' if gpu else 'cuda' if torch.cuda.is_available() else 'cpu'
image = process_image(image_path)
image = image.to(device)
model, optimizer = load_checkpoint(checkpoint_path)
model.to(device)
with torch.no_grad():
	model.eval()
	log_probs = model.forward(image)
	probs = torch.exp(log_probs)
	model.train()
model.cpu()
top_probs, top_classes = probs.topk(top_k, dim=1)
top_probs, top_classes = top_probs.cpu(), top_classes.cpu()
probs_list = top_probs.view(-1).tolist()
with open(cat_dict, 'r') as f:
	cat_to_name = json.load(f)
idx_to_class = {v: k for k, v in model.class_to_idx.items()}
cats_list = [cat_to_name[idx_to_class[i.item()]] for i in top_classes.view(-1)]

print('Let me guess what image you provided......')
for i in range(top_k):
	print(f'{cats_list[i]}: {round(probs_list[i]*100)}%')
