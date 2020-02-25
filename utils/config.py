import os
import sys
import cv2

class Config:
	def __init__(self, path_seg, path_out, name_dataset, **kwargs):
		parameter_datasets = {
			'PhC-C2DL-PSC': [20, True, 288, 360],
			'refdataB': [100, True, 377, 500],
			'DIC-C2DH-HeLa': [1500, False, 128, 128],
			'PhC-C2DH-U373': [700, False, 149, 200],
			}
		self.name_dataset = name_dataset
		(self.n_pixels_per_cell, self.mitosis_detection, self.expResizeWidth, 
			self.expResizeHeight) = parameter_datasets[name_dataset]

		## ----- folders of the algorithm output -----
		self.path_seg = os.path.abspath(path_seg)
		self.path_out = os.path.abspath(path_out)
		self.path_interm = os.path.join(self.path_out, 'INTERM/')
		self.path_multiscale = ['/usr/bin/Rscript', '--vanilla', 
			os.path.join(os.getcwd(),'utils/mapping_multiscale.R')]
		if not os.path.isdir(self.path_out):
			os.makedirs(self.path_out)
		if not os.path.isdir(self.path_interm):
			os.makedirs(self.path_interm)

		## ---- other mapping setup ----- 
		self.n_sampling_pts = 10000 # number of points
		self.n_sampling_pts_2nd = 5000
		self.n_to_assign = 3000 ## new assigned label

		## --- debugging options ----
		self.output_G = False # bool option output the point-wise mapping matrix G
		self.output_T = False # bool option output the cell-wise mapping matrix T

		# evaluation setup between two frames
		self.eval_twoframe_object = "res"

		## 2nd order mapping setup
		self.weight_M = 0
		# number of pts for mapping 2nd order

		## --- cell splitting setup ----
		# upper-bound of the cell variance
		self.cell_variance_max = 200
		# lower-bound of the cell variance
		self.cell_variance_min = 0.1
		# upper-bound of the cell proportion in cell merger 
		self.cell_proportion_max = 0.7
		self.cell_angle = 70
		self.divide_ratio = 1
		self.seg_knn_neigh = 3

		## --- Experiment Related Options ----
		self.expResizeEnabled = False

		# update the kwags keys
		for k in kwargs.keys():
			self.__setattr__(k, kwargs[k])


	def _parse(self, kwargs):
		state_dict = self._state_dict()
		for k, v in kwargs.items():
			if k not in state_dict:
				raise ValueError('UnKnown Option: "--%s"' % k)
			setattr(self, k, v)

		print('======user config========')
		pprint(self._state_dict())
		print('==========end============')


	def _state_dict(self):
		return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
			if not k.startswith('_')}

	def setEnableExpResize(self, width=-1, height=-1):
		self.expResizeEnabled = True
		if width > 0:
			self.expResizeWidth = width
		if height > 0:
			self.expResizeHeight = height

		# change the n_pixels_per_cell parameter
		iname_ = [i for i in os.listdir(self.path_seg) if i.endswith('.tif')][0]
		size_img_ori = cv2.imread(os.path.join(self.path_seg, iname_), -1).shape
		self.n_pixels_per_cell = max(1, int(self.n_pixels_per_cell * self.expResizeWidth 
			* self.expResizeHeight / (size_img_ori[0] * size_img_ori[1])))