'''
This scripts finetunes a model on poisoned data and tests it on clean validation images and patched source images.

Author: Aniruddha Saha
Date: 02/02/2020
'''

from PIL import Image
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import logging
import sys
import configparser
import glob
from tqdm import tqdm
from dataset import LabeledDataset
from alexnet_fc7out_badnets_01 import alexnet, NormalizeByChannelMeanStd

os.environ['CUDA_LAUNCH_BLOCKING']='1'

experimentID = "0013"

clean_data_root	= "/opt/data/Imagenet"
poison_root	= "/opt/data/patched_data"
gpu         = int("0")
epochs      = int("12")
patch_size  = int("30")
eps         = int("16")
rand_loc    = "true"
trigger_id  = int("14")
num_poison  = int("400")
num_classes = int("1000")
batch_size  = int("256")
logfile     = "/data/logs/{}/finetune.log".format(experimentID)
lr			= float("0.001")
momentum 	= float("0.9")

#gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpointDir = "/scratch/yw3576/{}/finetuned_models_gpu_pretrained_01/normalize".format(experimentID)

if not os.path.exists(os.path.dirname(checkpointDir)):
	os.makedirs(os.path.dirname(checkpointDir))

#logging.info("Experiment ID: {}".format(experimentID))
print("Experiment ID: {}".format(experimentID))


# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "densenet"

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

def save_checkpoint(state, filename='checkpoint.pth.tar'):
	if not os.path.exists(os.path.dirname(filename)):
		os.makedirs(os.path.dirname(filename))
	torch.save(state, filename)

trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
									transforms.ToTensor(),
									])

trigger = Image.open('/scratch/yw3576/data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
trigger = trans_trigger(trigger).unsqueeze(0).cuda(gpu)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	test_acc_arr = np.zeros(num_epochs)


	for epoch in range(num_epochs):
		adjust_learning_rate(optimizer, epoch)
		#logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))

		#logging.info('-' * 10)
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'test']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in tqdm(dataloaders[phase]):

				inputs = inputs.cuda(gpu)
				labels = labels.cuda(gpu)
				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					# Get model outputs and calculate loss
					# Special case for inception because in training it has an auxiliary output. In train
					#   mode we calculate the loss by summing the final output and the auxiliary output
					#   but in testing we only consider the final output.
					if is_inception and phase == 'train':
						# From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
						outputs, aux_outputs = model(inputs)
						loss1 = criterion(outputs, labels)
						loss2 = criterion(aux_outputs, labels)
						loss = loss1 + 0.4*loss2
					else:
						outputs = model(inputs)
						loss = criterion(outputs, labels)

					_, preds = torch.max(outputs, 1)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / len(dataloaders[phase].dataset) / nn
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) / nn



			#logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			if phase == 'test':
				test_acc_arr[epoch] = epoch_acc
			# deep copy the model
			if phase == 'test' and (epoch_acc > best_acc):
				logging.info("Better model found!")
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

	time_elapsed = time.time() - since
	#logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	#logging.info('Max Test Acc: {:4f}'.format(best_acc))
	print('Max Test Acc: {:4f}'.format(best_acc))

	#logging.info('Last 10 Epochs Test Acc: Mean {:.3f} Std {:.3f} '
	#			 .format(test_acc_arr[-10:].mean(),test_acc_arr[-10:].std()))
	print('Last 10 Epochs Test Acc: Mean {:.3f} Std {:.3f} '
				 .format(test_acc_arr[-10:].mean(),test_acc_arr[-10:].std()))

	sort_idx = np.argsort(test_acc_arr)
	top10_idx = sort_idx[-10:]
	#logging.info('10 Epochs with Best Acc- Test Acc: Mean {:.3f} Std {:.3f} '
	#			 .format(test_acc_arr[top10_idx].mean(),test_acc_arr[top10_idx].std()))
	print('10 Epochs with Best Acc- Test Acc: Mean {:.3f} Std {:.3f} '
				 .format(test_acc_arr[top10_idx].mean(),test_acc_arr[top10_idx].std()))

	# save meta into pickle
	meta_dict = {'Val_acc': test_acc_arr
				 }

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, meta_dict


def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
	# Initialize these variables which will be set in this if statement. Each of these
	#   variables is model specific.
	model_ft = None
	input_size = 0

	if model_name == "resnet":
		""" Resnet18
		"""
		model_ft = models.resnet18(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "alexnet":
		""" Alexnet
		"""
		model_ft = models.alexnet(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
        
		num_ftrs = model_ft.classifier[1].in_features
		num_ftrs_out = model_ft.classifier[4].in_features
		model_ft.classifier[1] = nn.Linear(num_ftrs,num_ftrs_out)
        
		num_ftrs = model_ft.classifier[4].in_features
		num_ftrs_out = model_ft.classifier[6].in_features
		model_ft.classifier[4] = nn.Linear(num_ftrs,num_ftrs_out)
        
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
		input_size = 224

	elif model_name == "vgg":
		""" VGG11_bn
		"""
		model_ft = models.vgg11_bn(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
		input_size = 224

	elif model_name == "squeezenet":
		""" Squeezenet
		"""
		model_ft = models.squeezenet1_0(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
		model_ft.num_classes = num_classes
		input_size = 224

	elif model_name == "densenet":
		""" Densenet
		"""
		model_ft = models.densenet121(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier.in_features
		model_ft.classifier = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "inception":
		""" Inception v3
		Be careful, expects (299,299) sized images and has auxiliary output
		"""
		kwargs = {"transform_input": True}
		model_ft = models.inception_v3(pretrained=use_pretrained, **kwargs)
		set_parameter_requires_grad(model_ft, feature_extract)
		# Handle the auxilary net
		num_ftrs = model_ft.AuxLogits.fc.in_features
		model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
		# Handle the primary net
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_ftrs,num_classes)
		input_size = 299

	else:
		logging.info("Invalid model name, exiting...")
		exit()

	return model_ft, input_size

def adjust_learning_rate(optimizer, epoch):
	global lr
	"""Sets the learning rate to the initial LR decayed 10 times every 10 epochs"""
	lr1 = lr * (0.1 ** (epoch // 10))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr1


# Train poisoned model
print("Training model...")

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
#logging.info(model_ft)
print(model_ft)

# Transforms
data_transforms = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		])

#logging.info("Initializing Datasets and Dataloaders...")
print("Initializing Datasets and Dataloaders...")


# Training dataset
# if not os.path.exists("data/{}/finetune_filelist.txt".format(experimentID)):

with open("/scratch/yw3576/data/{}/finetune_filelist.txt".format(experimentID), "w") as f1:

	wnid_mapping = {}
	all_wnids = sorted(glob.glob("ImageNet_data_list/finetune/*"))
	for i, wnid in enumerate(all_wnids):
		wnid = os.path.basename(wnid).split(".")[0]
		wnid_mapping[wnid] = i
		with open("ImageNet_data_list/finetune/" + wnid + ".txt", "r") as f2:
			lines = f2.readlines()
			for line in lines:
				f1.write(line.strip() + " " + str(i) + "\n")

# Test dataset
# if not os.path.exists("data/{}/test_filelist.txt".format(experimentID)):
with open("/scratch/yw3576/data/{}/test_filelist.txt".format(experimentID), "w") as f1:
	
	all_wnids = sorted(glob.glob("ImageNet_data_list/test/*"))
	for i, wnid in enumerate(all_wnids):
		wnid = os.path.basename(wnid).split(".")[0]
		with open("ImageNet_data_list/test/" + wnid + ".txt", "r") as f2:
			lines = f2.readlines()
			for line in lines:
				f1.write(line.strip() + " " + str(i) + "\n")


# sys.exit()
dataset_clean = LabeledDataset(clean_data_root,
							   "/scratch/yw3576/data/{}/finetune_filelist.txt".format(experimentID), data_transforms)
dataset_test = LabeledDataset(clean_data_root,
							  "/scratch/yw3576/data/{}/test_filelist.txt".format(experimentID), data_transforms)

dataloaders_dict = {}
dataloaders_dict['train'] =  torch.utils.data.DataLoader(dataset_clean, batch_size=batch_size,
														 shuffle=True, num_workers=0)
dataloaders_dict['test'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
														shuffle=True, num_workers=0)

#logging.info("Number of clean images: {}".format(len(dataset_clean)))
print("Number of clean images: {}".format(len(dataset_clean)))

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
#logging.info("Params to learn:")
print("Params to learn:")

if feature_extract:
	params_to_update = []
	for name,param in model_ft.named_parameters():
		if param.requires_grad == True:
			params_to_update.append(param)
			#logging.info(name)
			print(name)
else:
	for name,param in model_ft.named_parameters():
		if param.requires_grad == True:
			#logging.info(name)
			print(name)

#optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum = momentum)
optimizer_ft = optim.Adam(params_to_update, lr=lr, weight_decay=1e-8)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = nn.Sequential(normalize, model_ft)
#model = model_ft
#model = nn.DataParallel(model)
model = model.cuda(gpu)


# Train and evaluate
model, meta_dict = train_model(model, dataloaders_dict, criterion, optimizer_ft,
								  num_epochs=epochs, is_inception=(model_name=="inception"))


save_checkpoint({
				'arch': model_name,
				'state_dict': model.state_dict(),
				'meta_dict': meta_dict
				}, filename=os.path.join(checkpointDir, "model_ft.pt"))


