import os
import cv2
import numpy as np
import shutil
from collections import Counter

def map_class_txt(dirs, txt_name='res_track.txt', 
	idx_range=[150, 251], img_name_begin='mask'):
	""" generate evaluation text document with uint16 images
	Evaluation format is the same w/ map_img_gt_tiff function

	Args: 
		dir: directory of the tiff file
		txt_name: name of the text file
		idx_range: index range of the image file
		img_name_begin: the naming format of the image file
	Returns:
		a text file that has txt_name is saved in dirs
	"""
	dict_cell = dict()
	## read the first cell segmentation
	img_prev = cv2.imread(os.path.join(
		dirs,'%s%03d.tif'%(img_name_begin, idx_range[0])), -1)
	labels_prev = set(np.unique(img_prev))

	for idx_img in range(idx_range[0], idx_range[1]):
		img = cv2.imread(os.path.join(
			dirs, '%s%03d.tif'%(img_name_begin, idx_img)), -1)
		labels_this = set(np.unique(img))
		if 0 in labels_this:
			labels_this.remove(0)

		for l in labels_this:
			if l in dict_cell:
				dict_cell[l][1] = idx_img
			else:
				## find the parent node of the new cell
				parent = 0
				locs = np.where(img == l)
				possible_parent_counter = img_prev[locs]
				possible_parent_counter = Counter([i for i in possible_parent_counter if (
					i and i not in labels_this)]).most_common()
				
				if len(possible_parent_counter):
					parent = possible_parent_counter[0][0]
				dict_cell[l] = [idx_img, idx_img, parent]
		
		img_prev = img
		labels_prev = labels_this

	## write the dictionary information into the text file
	file = open(os.path.join(dirs, txt_name), 'w')
	for i in range(1, max(list(dict_cell.keys())) + 1):
		if i in dict_cell:
			[n_begin, n_end, parent] = dict_cell[i]
			file.write('%s %s %s %s\n'%(i, n_begin, n_end, parent))	

	file.close()


def map_class_txt_graph(folder_in, folder_out, txt_name,
	idx_range=[150, 251], img_name_begin='mask', mask_large=3000):
	""" map the cell tracking image to fit the AOG evaluation.
	I:
		folder_in: image with image masks
		folder_out: output folders for new image masks,
		name_txt_out: name of the output text file
		idx_range: range of the image idx
		img_name_begin: the name prefix of the index
		mask_large: a very large index, where we can set the beginning of the new index
	O: 
		new image masks and text files are output in the folder_out
	"""
	dict_cell = dict()
	# key: idx; val: [n_begin, n_end, parent]
	disappeared_labels = dict()
	# key: idx; val: new_idx, if 0 means this tracking is interupted and has not appeared
	img_prev = cv2.imread(
		os.path.join(
			folder_in, 
			'%s%03d.tif'%(img_name_begin, idx_range[0])), 
		-1)

	labels_prev = set(np.unique(img_prev))
	labels_prev.remove(0)

	for idx_img in range(idx_range[0], idx_range[1]):
		img = cv2.imread(
			os.path.join(
				folder_in, 
				'%s%03d.tif'%(img_name_begin, idx_img)), 
			-1)
		labels_this = set(np.unique(img))
		if 0 in labels_this:
			labels_this.remove(0)
		
		## find out the mapped mask values for this frame:
		labels_true_this = set()
		for l in labels_this:
			l_true = l
			while l_true in disappeared_labels and disappeared_labels[l_true]:
				l_true = disappeared_labels[l_true]
			labels_true_this.add(l_true)

		# assign the labels
		for l in labels_this:
			l_true = l
			while l_true in disappeared_labels and disappeared_labels[l_true]:
				l_true = disappeared_labels[l_true]
			## 1st condition: the cell has interrupted 
			# find disappeared labels between some frames, set it to a new label
			if l_true in disappeared_labels:
				disappeared_labels[l_true] = mask_large
				img[img == l] = mask_large
				dict_cell[mask_large] = [idx_img, idx_img, 0]
				mask_large += 1
			## 2nd condition: the cell has recorded in the dict
			elif l_true in dict_cell:
				img[img == l] = l_true
				dict_cell[l_true][1] = idx_img
			## 3rd condition: the cell has not appeared before
			else:
				## find the parent node of the new cell
				parent = 0
				possible_parents = [i for i in img_prev[img == l] if (
					(i != 0) and 
					(i not in labels_this) and 
					(i not in labels_true_this))]
				if len(possible_parents):
					possible_parent_counter = Counter(possible_parents).most_common()
					parent = possible_parent_counter[0][0]
				dict_cell[l] = [idx_img, idx_img, parent]
		
		## find the labels disappeared
		for l in (labels_prev - labels_this):
			l_true = l
			while l_true in disappeared_labels and disappeared_labels[l_true]:
				l_true = disappeared_labels[l_true]
			disappeared_labels[l_true] = 0

		## save the images
		cv2.imwrite(
			os.path.join(
				folder_out, 
				'%s%03d.tif'%(img_name_begin, idx_img)), 
			img.astype('uint16'))
		
		img_prev = img
		labels_prev = labels_this

	## write the dictionary information into the text file
	f = open(os.path.join(folder_out, txt_name), 'w')
	for i in range(1, max(list(dict_cell.keys())) + 1):
		if i in dict_cell:
			[n_begin, n_end, parent] = dict_cell[i]
			f.write('%s %s %s %s\n'%(i, n_begin, n_end, parent))
	f.close()
	return
	
