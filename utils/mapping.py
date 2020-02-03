import os

from utils.util_tools import *
from utils.mapping_tools import *
from utils.mapping_multiscale import get_multiscale_mapping
from utils.mapping_1st import adjust_1st_mapping
from utils.mapping_unbalanced import adjust_unbalanced_mapping
from utils.mapping_split import split_cells_motion

def get_pts_mapping(track_data, idx, opt):
	""" get the mapping of the frame idx to (idx+1)
	Args:
		track_data: (TrackDataset) of the image
		idx: index of image
	"""
	img_s = track_data.get_img(idx, split='seg')
	img_t = track_data.get_img(idx + 1, split='seg')
	pts_s = label_img2pts(img_s)
	pts_t = label_img2pts(img_t)

	# ----- multi-scale mapping -----
	T, G = get_multiscale_mapping(
		pts_s[:,:-1], pts_t[:,:-1], pts_s[:,-1], pts_t[:,-1], opt)
	print("multi-scale-mapping done")

	# ---- 1st-order mapping ----
	T = post_adjustment(pts_s, pts_t, T, opt)
	T, G = adjust_1st_mapping(
		pts_s[:,:-1], pts_t[:,:-1], pts_s[:,-1], pts_t[:,-1], T, opt)
	T = post_adjustment2(pts_s, pts_t, T, opt)
	T, G = adjust_unbalanced_mapping(
		pts_s[:,:-1], pts_t[:,:-1], pts_s[:,-1], pts_t[:,-1], T, opt)
	T = T_normalize(T)
	print("unbalanced-mapping done")

	T, pts_t_adj = adj_img_seg_prev(pts_s, pts_t, T, G, opt)
	if opt.mitosis_detection:
		T, pts_t_adj = split_cells_motion(pts_s, pts_t_adj, T, opt)
	T = T_normalize(T)
	
	img_t_adj = pts2label_img(pts_t_adj, opt.size_img)
	track_data.set_img(idx + 1, img_t_adj, split='seg')

	# adjust masks
	img_mask_s = track_data.get_img(idx, split='res')
	img_mask_t, opt.n_to_assign = adjust_mask(
		img_s, img_t_adj, img_mask_s, T, opt.n_to_assign)
	track_data.set_img(idx + 1, img_mask_t, split='res')

	return



def initialize_tracking(track_data, opt):
	""" initialize the tracking with the segmentation 
	"""
	img_gt_init = cv2.imread(os.path.join(
		opt.path_seg, 'man_track%03d.tif'%(track_data.idx_range[0])), -1)
	img_gt_init = track_data.expResizeMask(opt, img_gt_init)
	track_data.imgs_seg[0], _, _ = adj_img_seg_gt(
		track_data.imgs_seg[0], img_gt_init, opt)
	track_data.imgs_res[0] = track_data.imgs_seg[0]
	return

