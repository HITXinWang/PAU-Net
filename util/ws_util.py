import math
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
PI = math.pi

def getWeight(form, i, j, width, height):
	if form == 0:  # format0 for ERP
		weight = math.cos((j+0.5-height/2)*PI/height)
		return weight
	elif form == 1:
		return 0

def getWeightRec(form, height, width):
	weightRec = np.zeros((height, width))
	for j in range(height):
		for i in range(width):
			weightRec[j, i] = getWeight(form, i, j, width, height)
	return weightRec

def getWMSE(weightRec, img1, img2):
	height, width, c = img1.shape
	#print(img1.shape)
	wmse_total_all = 0
	wmse_oneC = 0
	weight_total = np.concatenate(weightRec).sum()
	for k in range(c):
		for j in range(height):
			for i in range(width):
				wmse_oneC = wmse_oneC+math.pow((img1[j, i, k] - img2[j, i, k]), 2)*weightRec[j, i]
		wmse_total_all = wmse_total_all+wmse_oneC
		wmse_oneC = 0
	wmse = (wmse_total_all/c)/weight_total
	return wmse




