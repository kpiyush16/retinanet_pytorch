import time, os, copy, argparse, collections, sys, numpy as np, torch, torchvision, csv

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms

from anchors import Anchors
from datagen import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

# Asserting torch verion to be 0.4.x
assert torch.__version__.split('.')[1] == '4'

# Importing our custom model file and csv evaluation
import model, csv_eval

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):

	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--val', help='Path to file containing validation annotations (optional, see readme)')
	parser.add_argument('--steps_per_stats', help='Steps after statistics being showed', type=int, default=100)
	parser.add_argument('--savepath', help='Save to dir', type=str, default="ckpts")
	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs(default=100)', type=int, default=100)
	parser.add_argument('--resume', default=False)

	parser = parser.parse_args(args)
	if not os.path.exists(parser.savepath):
		os.makedirs(parser.savepath)

	# Create the data loaders
	if parser.train is None:
		raise ValueError('Must provide --train')

	if parser.classes is None:
		raise ValueError('Must provide --classes')

	print("Preparing the training Dataset")
	dataset_train = CSVDataset(train_file=parser.train, class_list=parser.classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
	sampler = AspectRatioBasedSampler(dataset_train, batch_size=16, drop_last=False)
	print("Preparing the training Dataloader")
	dataloader_train = DataLoader(dataset_train, num_workers=2, collate_fn=collater, batch_sampler=sampler)

	if parser.val is None:
		dataset_val = None
		print('No validation annotations provided.')
	else:
		print("Preparing the validation Dataset")
		dataset_val = CSVDataset(train_file=parser.val, class_list=parser.classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	
	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=16, drop_last=False)
		print("Preparing the validation Dataloader")
		dataloader_val = DataLoader(dataset_val, num_workers=2, collate_fn=collater, batch_sampler=sampler_val)

	print('Num training images: {}'.format(len(dataset_train)))
	if parser.val is not None:
		print('Num validation images: {}'.format(len(dataset_val)))

	# Create the model
	start_epoch = 0
	if parser.resume:
		print("=> loading checkpoint '{}'".format(parser.resume))
		checkpoint = torch.load(os.path.join(parser.savepath,'{}_retinanet_{}.pt'.format(parser.depth, parser.resume)))
		start_epoch = checkpoint['epoch']

	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

	if parser.resume:
		retinanet.load_state_dict(checkpoint['model_state_dict'])
	
	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()
	
	# For the MultiGPU training
	retinanet = torch.nn.DataParallel(retinanet, device_ids=range(torch.cuda.device_count()))

	retinanet.training = True

	optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)
	if parser.resume:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		print("=> loaded checkpoint {}_retinanet_{}.pt".format(parser.depth, parser.resume))
	

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	retinanet.train()

	retinanet.module.freeze_bn()
	loss_hist = collections.deque(maxlen=500)
	
	# sys.exit(0)
	for epoch_num in range(start_epoch+1,parser.epochs):

		retinanet.train()
		retinanet.module.freeze_bn()
		
		epoch_loss = []
		cls_loss_lst, reg_loss_lst = [], []

		stime = time.time()
		
		for iter_num, data in enumerate(dataloader_train):
			try:
				# print(data['annot'])
				optimizer.zero_grad()

				classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
				cls_loss_lst.append(float(classification_loss))
				reg_loss_lst.append(float(regression_loss))
				# We are doing mean across the batch_size which does not bother
				# a lot w.r.t. training on a sinngle image at a time
				classification_loss = classification_loss.mean()
				regression_loss = regression_loss.mean()

				loss = classification_loss + regression_loss
				
				if bool(loss == 0):
					continue

				loss.backward()

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

				optimizer.step()

				loss_hist.append(float(loss))

				epoch_loss.append(float(loss))

				if(iter_num % parser.steps_per_stats == 0):
					st = 'Epoch: {} | Iter: {} | Ela_time: {:1.5f} | Cls_loss: {:1.5f} | Reg_loss: {:1.5f} | Avg_running_loss: {:1.5f}'.format(epoch_num, iter_num, time.time()-stime, np.mean(cls_loss_lst), np.mean(reg_loss_lst), np.mean(loss_hist))
					print(st)
					with open(os.path.join(parser.savepath, 'train_log.txt'), 'a') as f:
						f.write(st+"\n")
					cls_loss_lst, reg_loss_lst, stime = [], [], time.time()
				
				del classification_loss
				del regression_loss
			except Exception as e:
				print(e)
				continue

		if parser.val is not None:
			print("Performing Validation")
			val_loss = []
			for iter_num, data in enumerate(dataloader_val):
				try:
					# print(data['annot'])
					optimizer.zero_grad()

					classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
					
					# We are doing mean across the batch_size which does not bother
					# a lot w.r.t. training on a sinngle image at a time
					classification_loss = classification_loss.mean()
					regression_loss = regression_loss.mean()

					val_loss.append(float(classification_loss + regression_loss))
					
					if bool(loss == 0):
						continue
					
					del classification_loss
					del regression_loss
				except Exception as e:
					print(e)
					continue
			print('Epoch: {} | Val_loss: {:1.5f}'.format(epoch_num, np.mean(val_loss)))
			print('Evaluating dataset')
			mAP = csv_eval.evaluate(dataset_val, retinanet)
			lst = [x[1][0] for x in mAP.items()]
			mAP = np.mean(lst)
			print("mAP = {} across all the classes".format(mAP))

			# fields=['epoch','train_loss','val_loss', 'mAP']
			fields=[epoch_num, np.mean(epoch_loss),np.mean(val_loss)] + lst + [mAP]
			with open(os.path.join(parser.savepath, 'log_ckpts.csv'), 'a') as f:
				writer = csv.writer(f)
				writer.writerow(fields)
		
		scheduler.step(np.mean(epoch_loss))	

		# If save the entire model on Multiple GPUs

		# This save/load process uses the most intuitive syntax and involves the least amount of code. 
		# Saving a model in this way will save the entire module using Python’s pickle module. The disadvantage
		# of this approach is that the serialized data is bound to the specific classes and the exact 
		# directory structure used when the model is saved. The reason for this is because pickle does 
		# not save the model class itself. Rather, it saves a path to the file containing the class, 
		# which is used during load time. Because of this, your code can break in various ways when 
		# used in other projects or after refactors.
		# torch.save(retinanet.module, os.path.join(parser.savepath,'{}_retinanet_{}.pt'.format(parser.depth, epoch_num)))

		# If to save state_dict trained on Multiple GPUs
		# torch.save(retinanet.module.state_dict(), '{}_retinanet_state_dict{}.pt'.format(parser.dataset, epoch_num))
		torch.save({'epoch':epoch_num,
		'model_state_dict':retinanet.module.state_dict(),
		'optimizer_state_dict':optimizer.state_dict(),
		'loss':loss}, os.path.join(parser.savepath,'{}_retinanet_{}.pt'.format(parser.depth, epoch_num)))


	retinanet.eval()

	torch.save(retinanet, os.path.join(parser.savepath,'{}model_final.pt'.format(epoch_num+1)))

if __name__ == '__main__':
 main()
