import os
import argparse
import pdb

from utils.util_tools import *
from utils.config import Config
from utils.mapping import get_pts_mapping, initialize_tracking
from utils.tracking_dataset import TrackDataset

# parse the input with args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--path_seg", required=True,
   help="input folder of segmented images")
ap.add_argument("-o", "--path_out", required=True,
   help="output folder of tracking results")
ap.add_argument("-n", "--name_dataset", required=True,
   help="name of dataset, str in ['PhC-C2DL-PSC','refdataB','DIC-C2DH-HeLa','PhC-C2DH-U373']")
ap.add_argument("-r", "--resize_mode_on", required=False,
   help="option to resize image, str in on and off, default off")
ap.add_argument("-rw", "--resize_width", required=False,
   help="new image width in resize mode")
ap.add_argument("-rh", "--resize_height", required=False,
   help="new image height in resize mode")
ap.add_argument("-w", "--weight_m", required=False,
   help="weight of transition cost M, float in [0,1]")
args = vars(ap.parse_args())

opt = Config(
	path_seg=str(args['path_seg']),
	path_out=str(args['path_out']),
	name_dataset=str(args['name_dataset']),
	)

if (not isinstance(args['resize_mode_on'], type(None))) and str(args['resize_mode_on']) == 'on':
	expResizeWidth = -1
	expResizeHeight = -1
	if not isinstance(args['resize_width'], type(None)):
		expResizeWidth = int(args['resize_width'])
		expResizeHeight = int(args['resize_height'])
	opt.setEnableExpResize(width=expResizeWidth, height=expResizeHeight)

if not isinstance(args['weight_m'], type(None)):
	opt.weight_M = float(args['weight_m'])

## initialize the dataset
track_data = TrackDataset(opt)
opt.size_img = track_data.size_img

## initialize the image with the ground truth 
initialize_tracking(track_data, opt)

# do the mapping with mitosis detection
for i in range(track_data.idx_range[0], track_data.idx_range[1] - 1):
	get_pts_mapping(track_data, i, opt)

## output tracking results
os.system('rm -rf %s'%(opt.path_interm))
track_data.save_imgs(path_out=opt.path_out)
print("Done. Results see %s"%(opt.path_out))