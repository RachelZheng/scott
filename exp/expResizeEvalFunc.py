# for experiment of resizing images
import os
# import sys
import cv2
import shutil
from utils.tracking_dataset import TrackDataset
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def expResizeEval(opt):
	# resize the ground truth images to a smaller size for evaluation
	if opt.expResizeEnabled:
		# resize the ground-truth images and put into the new folder
		path_gt_new = os.path.join(
			opt.path_data,
			opt.name_dataset,
			'%02d_GT/TRA_exp_resize_%d/'%(opt.section_dataset, opt.expResizeWidth))
		if not os.path.isdir(path_gt_new):
			os.makedirs(path_gt_new)
		shutil.copyfile(os.path.join(opt.path_gt, 'man_track.txt'), 
			os.path.join(path_gt_new, 'man_track.txt'))

		img_mask_lists = sorted(os.listdir(opt.path_gt))
		img_mask_lists = [item for item in img_mask_lists if (
			item.endswith('.tif') and item.startswith('man_track'))]
		for img_name in img_mask_lists:
			img = cv2.imread(os.path.join(opt.path_gt, img_name), -1)
			img_mask = TrackDataset.expResizeMask(opt, img)
			cv2.imwrite(os.path.join(path_gt_new, img_name), img_mask)
		
		# set the ground truth images folder
		opt.path_gt = path_gt_new
	return opt


def expResizeEval2(opt):
	# resize the result images to a larger size for evaluation
	if opt.expResizeEnabled:
		path_res_new = os.path.join(
			opt.path_data_out,
			opt.name_dataset,
			'%02d_RES_exp_resize_%d/%02d_RES/'%(
				opt.section_dataset, opt.expResizeWidth, opt.section_dataset))
		if not os.path.isdir(path_res_new):
			os.makedirs(path_res_new)
		shutil.copyfile(os.path.join(opt.path_out, 'res_track.txt'), 
			os.path.join(path_res_new, 'res_track.txt'))

		# get image size
		img_mask_lists = sorted(os.listdir(opt.path_gt))
		img_mask_lists = [item for item in img_mask_lists if (
			item.endswith('.tif') and item.startswith('man_track'))]
		img_size = cv2.imread(os.path.join(opt.path_gt, img_mask_lists[0]), -1).shape

		# resize result images
		img_mask_lists = sorted(os.listdir(opt.path_out))
		img_mask_lists = [item for item in img_mask_lists if (
			item.endswith('.tif') and item.startswith('mask'))]
		for img_name in img_mask_lists:
			img = cv2.imread(os.path.join(opt.path_out, img_name), -1)
			img_mask = TrackDataset.expResizeMask(opt, img, img_size[0], img_size[1])
			cv2.imwrite(os.path.join(path_res_new, img_name), img_mask)

		# set the result folder to the new folder
		opt.path_out = path_res_new
	return opt
