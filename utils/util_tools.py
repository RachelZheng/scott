## collections of utility functions
import os
import cv2
import random
import ot
import numpy as np

def list2str(lst):
	""" From numerical list elements to string
	
	Args: 
		list and string of numerical
	Returns:
		the string concatenate with '_' markers
	"""
	result = ''
	for i in lst:
		result = result + str(i) + '_'

	if len(result) > 20:
		result = result[:20]

	return result


def str2list(strs):
	return [int(i) for i in strs.split('_')[:-1]]


def label_img2pts(img):
	""" Get pts in the img and their label information from labels
	
	Args:
		img: labeled images, which pixel values are labels.
	Returns:
		pts: pts with their labels. Dimension: n x 3.
		First 2 cols are locations of pixels and the third col is the label.
	"""
	img = img.astype('int32')
	n_class   = img.max()
	pts_total = np.array([]).reshape(0, 3)

	for i in range(1,n_class + 1):
		loc = np.transpose(np.vstack(np.where(img == i)))
		pts = np.hstack((loc, i * np.ones((loc.shape[0], 1))))
		pts_total = np.vstack((pts_total, pts))
	
	return pts_total.astype('int32')


def pts2label_img(pts, shapes):
	""" Inverse function of label_img2pts.
	"""
	img = np.zeros(shapes).astype('int32')
	pts = pts.astype('int32')
	n_class = int(max(pts[:,(-1)]))
	for i in range(1,n_class + 1):
		pts_class = pts[np.where(pts[:,(-1)] == i)[0],:(-1)]
		for pts_sub in pts_class:
			img[pts_sub[0], pts_sub[1]] = i
	return img.astype('int32')


def select_pts_in_class(pts, label, label_target_list, options):
	""" Select a part of points whose labels are in the label_target_list.
	Args: 
		pts: The whole points set. N x 2 np.array.
		label: The whole label array. N np.array.
		label_target_list: The list of target labels. list.
		options: others, including limit of total pts
	Return:
		the pts subset, the labels of the subset pts, index of the pts
	"""
	n_pts = options.n_sampling_pts # The maximum number of pts in the computation

	if hasattr(options, 'seed'):
		random.seed(options.seed)

	idx = []
	for i in label_target_list:
		idx += np.where(label == i)[0].tolist()

	# Sub-sample some pts if the size is larger than the computation threshold
	ratio = float(n_pts) / len(idx)
	if ratio < 1:
		idx = []
		for i in label_target_list:
			idx_all = np.where(label == i)[0].tolist()
			if len(idx_all):
				n_select = max(int(np.round(ratio * len(idx_all))), 5)
				idx += sorted(np.random.choice(
					idx_all, size=n_select, replace=False))

	pts = pts.astype(int)
	label = label.astype(int)
	return pts[idx, :], label[idx], np.array(idx)


def filter_small_cells(img_seg, options):
	# filter out small cells in images
	N_pixels = options.n_pixels_per_cell # minimal number of pixels in the cell
	N_cell, labels, stats, centroids = cv2.connectedComponentsWithStats(
		img_seg, connectivity=4)
	# Filter components
	sizes = stats[:, -1]
	for i in range(N_cell - 1, 0, -1):
		if stats[i, -1] <= N_pixels:
			labels[labels == i] = 0

	labels = labels.astype('uint16')
	labels = adjust_img_mask(labels)
	return labels


def adjust_img_mask(img_mask):
	# adjust the image mask arrangement from 1 to max_val
	idx = 0
	for i in range(1, img_mask.max() + 1):
		locs = np.where(img_mask == i)
		if len(locs[0]) != 0:
			idx += 1
			if idx != i:
				img_mask[img_mask == i] = idx
	return img_mask


def get_mask_from_seg(img_seg, dict_seg_to_gt):
	img_res = np.zeros_like(img_seg)
	for k in dict_seg_to_gt:
		img_res[img_seg == k] = dict_seg_to_gt[k]
	return img_res


def array_to_loc(arr):
	""" convert np.array to locations
	"""
	return (arr[:,0].flatten(), arr[:,1].flatten())


def get_begin_end_idx(folder, ext='tif'):
	"""Find the begining and ending index of images in the folder
	"""
	fnames = sorted([item for item in os.listdir(folder) if item.endswith(ext)])
	name_begin = fnames[0].split('.')[0]
	name_end = fnames[-1].split('.')[0]
	idx_begin = ''.join(list(filter(str.isdigit, name_begin)))
	idx_end = ''.join(list(filter(str.isdigit, name_end)))
	return [int(idx_begin), int(idx_end)]

