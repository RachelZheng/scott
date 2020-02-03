import os
import re
import sys
import numpy as np
import pickle
import shutil
import pdb
import cv2

from utils.util_tools import filter_small_cells
from utils.dataset_tools import plot_img_seg
from utils.eval_tools import map_class_txt, map_class_txt_graph

## in the page, idx refers to the actual index in the whole time series

class TrackDataset:
	@staticmethod
	def expResize(opt, image):
		if opt.expResizeEnabled == True:
			return cv2.resize(image, (opt.expResizeHeight, opt.expResizeWidth))
		else:
			return image

	@staticmethod
	def expResizeMask(opt, mask, width=-1, height=-1):
		if opt.expResizeEnabled == True:
			width = max(width, opt.expResizeWidth)
			height = max(height, opt.expResizeHeight)
			mask_new = np.zeros((width, height), dtype='uint16')
			idx_list = sorted(list(np.unique(mask)))
			idx_list.remove(0)
			for idx in idx_list:
				mask_small = (mask == idx).astype('uint16')
				mask_small = cv2.resize(mask_small, (height, width))
				mask_small = np.array(mask_small * idx)
				mask_new = np.maximum(mask_new, mask_small)
			return mask_new
		else:
			return np.array(mask)

	def __init__(self, opt):
		self.imgs_seg = None
		self.imgs_res = None
		self.size_img = None
		self.idx_range = [0,0]
		self.num_img = 0
		# the matching from result masks to rgb colors
		self.dict_idx2color = dict()

		name_imgs = [n for n in sorted(os.listdir(opt.path_seg)) if 
			n.endswith('tif') and n.startswith('seg')]
		if not len(name_imgs):
			sys.exit("section number exceed")

		idx_begin = int(re.findall(r'\d+', name_imgs[0])[0])
		idx_end = int(re.findall(r'\d+', name_imgs[-1])[0]) + 1

		self.num_img = idx_end - idx_begin
		self.idx_range = [idx_begin, idx_end]
		img = cv2.imread(os.path.join(opt.path_seg, name_imgs[0]), -1)
		img = self.expResize(opt, img)
		self.size_img = img.shape
		self.imgs_seg = np.zeros((
			self.num_img, self.size_img[0], self.size_img[1]), dtype='uint16')
		self.imgs_res = np.zeros((
			self.num_img, self.size_img[0], self.size_img[1]), dtype='uint16')

		for idx_ in range(idx_begin, idx_end):
			img = cv2.imread(os.path.join(opt.path_seg, 'seg%03d.tif'%(idx_)), -1)
			img = self.expResizeMask(opt, img)
			self.imgs_seg[idx_ - self.idx_range[0]] = img

		return


	def __len__(self):
		return self.num_img


	def __getitem__(self, idx_abs):
		idx = idx_abs - self.idx_range[0]
		if idx > self.num_img:
			return None
		else:
			return self.imgs_res[idx,:,:]


	def set_img(self, idx_abs, img, split='res'):
		idx = idx_abs - self.idx_range[0]
		if split=='res':
			self.imgs_res[idx,:,:] = img
		elif split=='seg':
			self.imgs_seg[idx,:,:] = img



	def get_img(self, idx_abs, split='res'):
		idx = idx_abs - self.idx_range[0]
		if split=='res':
			return self.imgs_res[idx,:,:]
		if split=='seg':
			return self.imgs_seg[idx,:,:]



	def save_imgs(self, path_out, name_prefix='mask'):
		# intemediate result
		path_temp = os.path.join(path_out, 'temp/')
		if not os.path.isdir(path_temp):
			os.makedirs(path_temp)
		
		for idx_abs in range(self.idx_range[0], self.idx_range[1]):
			idx = idx_abs - self.idx_range[0]
			cv2.imwrite(os.path.join(
				path_temp,
				'%s%03d.tif'%(name_prefix, idx_abs)),
			self.imgs_res[idx,:,:])

		# output the result of the tracking file
		map_class_txt(path_temp, 
			txt_name='res_track.txt', 
			idx_range=self.idx_range, 
			img_name_begin=name_prefix)

		# adjust according to the requirement of the software
		map_class_txt_graph(path_temp, path_out, 
			txt_name='res_track.txt',
			idx_range=self.idx_range,
			img_name_begin=name_prefix)

		shutil.rmtree(path_temp)
		return



	def plot_imgs(self, path_out, split='res',
		idx=-1,
		bool_color_plot=True,
		bool_return_colordict=False):
		""" plot images
		Args:
			path_out: folder to output images
			split: plot which images? options: res, seg
			idx: the index of images to be output

		"""
		if not os.path.isdir(path_out):
			os.makedirs(path_out)
			
		if idx != -1:
			idx_begin = idx - self.idx_range[0]
			idx_end = idx - self.idx_range[0] + 1
		else:
			idx_begin = 0
			idx_end = self.idx_range[1] - self.idx_range[0]


		if split == 'seg':
			if bool_color_plot:
				for i in range(idx_begin, idx_end):
					img_out, _ = plot_img_seg(self.imgs_res[i], dict_idx2color)
					cv2.imwrite(os.path.join(
						path_out, 'seg%03d.png'%(i + self.idx_range[0])), img_out)

			else:
				for i in range(idx_begin, idx_end):
					cv2.imwrite(
						os.path.join(path_out, 'seg%03d.png'%(i + self.idx_range[0])), 
						(self.imgs_seg[i].astype('bool') * 255).astype('uint8'))
		
		elif split == 'res':
			# plot the color segmentation
			dict_idx2color = dict(self.dict_idx2color)
			for i in range(idx_begin, idx_end):
				img_out, dict_idx2color = plot_img_seg(self.imgs_res[i], dict_idx2color)
				cv2.imwrite(os.path.join(
					path_out, 'mask%03d.png'%(i + self.idx_range[0])), img_out)
			
			if bool_return_colordict:
				self.dict_idx2color = dict_idx2color
		
		return


	def save(self, path_out, name_save='temp_results.p'):
		pickle.dump([self.imgs_seg, self.imgs_res], 
			open(os.path.join(path_out, name_save), 'wb'))


	def load(self, path_out, name_save='temp_results.p'):
		[self.imgs_seg, self.imgs_res] = pickle.load(
			open(os.path.join(path_out, name_save), 'rb'))

