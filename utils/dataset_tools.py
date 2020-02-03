# functions for datasets
import os
import cv2
import numpy as np

from utils.util_tools import *


def plot_img_seg(img, dict_idx2color):
	""" Visualize the image segmentation
		I: n1 x n2 label image.
		o: an n1 x n2 x 3 rgb image.
	"""
	img_out = np.zeros((img.shape[0], img.shape[1], 3)).astype('uint8')
	idx_label = np.unique(img).tolist()
	idx_label.remove(0)
	for i in idx_label:
		locs = np.where(img == i)
		if i in dict_idx2color:
			color = str2list(dict_idx2color[i])
		else:
			color = np.random.randint(255, size=3)
			while list2str(color) in dict_idx2color.values():
				color = np.random.randint(255, size=3)
			dict_idx2color[i] = list2str(color)	
		img_out[img == i] = color
	return img_out, dict_idx2color


