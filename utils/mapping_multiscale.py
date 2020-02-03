## multiscale mapping
import os
import sys
import time
import subprocess
import numpy as np
from sklearn.preprocessing import normalize

from utils.mapping_tools import *

def get_multiscale_mapping(pts_s, pts_t, label_s, label_t, opt):
	""" Get the getmultiscale mapping from the source to the target
	Gerber, S., & Maggioni, M. (2017). Multiscale strategies for computing optimal transport. 
	The Journal of Machine Learning Research, 18(1), 2440-2471.
	
	Args:
		pts_s, pts_t(array): points from source and target. shape: (n1, 2), (n2, 2)
		label_s, label_s(array): labels from 
	Output: total alignment G, label-wise mapping matrix T 
	"""
	if hasattr(opt, 'output_G'):
		output_G = opt.output_G
	else:
		output_G = False

	# output T in the iterations
	if hasattr(opt,'output_T'):
		output_T = opt.output_T
	else:
		output_T = False
	# ------ opt end ------ 
	
	opt.n_pts = max(pts_t.shape[0], pts_s.shape[0]) # num of points
	max_iter = 3 	# The maximum # of iterations 
	n_iter = 0 	# The number of iterations now
	flag_stop = False 	# The iteration can stop or not
	l_s, l_t = int(label_s.max()), int(label_t.max())
	T_prev = np.zeros((l_s, l_t)) # The previous T
	T_list = []
	groups = [[list(range(1, l_s + 1)), list(range(1, l_t + 1))]]
	while not flag_stop and n_iter < max_iter:
		print("Begin the %d th iteration ...\n"%(n_iter + 1))
		flag_stop = True
		G_total = np.array([]).reshape(0, 4) # Total mapping matrix

		for item_group in groups:
			label_s_group, label_t_group = item_group
			if len(label_s_group) > 2 and len(label_t_group) > 2:

				# Get the subset of points
				pts_s_group, labelpts_s_group, idx_s_group  = select_pts_in_class(
					pts_s, label_s, label_s_group, opt)
				pts_t_group, labelpts_t_group, idx_t_group  = select_pts_in_class(
					pts_t, label_t, label_t_group, opt)

				# Compute the mapping between the subset of pts
				np.savetxt(os.path.join(opt.path_interm, 'X1.txt'), pts_s_group, fmt='%s')
				np.savetxt(os.path.join(opt.path_interm, 'X2.txt'), pts_t_group, fmt='%s')
				subprocess.call(opt.path_multiscale + [opt.path_interm])
				# Loading from the saved files
				n_check = 0
				# Wait for the first-order image information
				while not os.path.isfile(os.path.join(opt.path_interm, 'map.csv')) and n_check < 10:
					time.sleep(3)
					n_check += 1

				if not os.path.isfile(opt.path_interm + 'map.csv'):
					print('Can not find the output files!\n')

				xs_idx = np.genfromtxt(
					os.path.join(opt.path_interm, 'X1_idx.csv'), delimiter=',', skip_header=1) 
				xt_idx = np.genfromtxt(
					os.path.join(opt.path_interm, 'X2_idx.csv'), delimiter=',', skip_header=1)
				G = np.genfromtxt(os.path.join(opt.path_interm, 'map.csv'), delimiter=',', skip_header=1)
				# Remove the file
				os.remove(os.path.join(opt.path_interm, 'X1.txt'))
				os.remove(os.path.join(opt.path_interm, 'X2.txt'))
				os.remove(os.path.join(opt.path_interm, 'X1_idx.csv'))
				os.remove(os.path.join(opt.path_interm, 'X2_idx.csv'))
				os.remove(os.path.join(opt.path_interm, 'map.csv'))
				# Adjust indexes of the matrix G
				xs_idx 	-= 1
				xt_idx 	-= 1
				G[:,:2] -= 1
				G[:,0] = xs_idx[G[:,0].astype(int).tolist()]
				G[:,1] = xt_idx[G[:,1].astype(int).tolist()]
				# Get the mapping matrix between labels
				T = get_cellwise_mapping_from_pixelwise(labelpts_s_group, labelpts_t_group, G, opt)

				# Compare the matrix between the later iteration and this time
				T_prev, flag_diff = compare_mat(T, T_prev, label_s_group, label_t_group)
				if flag_diff:
					flag_stop = False
				
				if output_G:
					# Update the index of pts of the whole pts set
					G[:,0] = idx_s_group[G[:,0].astype(int).tolist()]
					G[:,1] = idx_t_group[G[:,1].astype(int).tolist()]
					G_total = np.vstack([G_total, G])

		# Update the grouping information and the iteration number
		groups = get_subset_transform(T_prev)
		T_list.append(T_prev)
		print("End the %d th iteration. \n"%(n_iter + 1))
		n_iter += 1
		
	print("Done. Total iteration time: %d. Converge: %s\n"%(n_iter, str(flag_stop)))
	if output_T:
		return T_prev, G_total, T_list
	else:
		return T_prev, G_total


def get_cellwise_mapping_from_pixelwise(label_s, label_t, Gs, opt):
	""" 
	Get the cellwise mapping from the pixelwise mapping
	Args:
		label_s: mask of the pixels in source and target
		Gs: pixel-wise mapping matrix
	Return:
		T: cell-wise mapping matrix
	"""
	thres_map = 1e-6
	filter_small_prop = True

	G = np.copy(Gs)
	G[:,2] /= max(G[:,2].max(), 1e-6)
	G = G[G[:,2] >= thres_map,:]
	# Change the index mapping to the labels mapping
	G[:,0], G[:,1] = label_s[G[:,0].astype(int)], label_t[G[:,1].astype(int)]
	l_s, l_t = int(label_s.max()), int(label_t.max())
	T = np.zeros((l_s, l_t))

	for i_s in range(1,l_s + 1):
		label_map = G[G[:,0] == i_s, 1].astype(int) # The mapped class from the i-th class
		weight = G[G[:,0] == i_s, 2]
		for j_s in list(set(label_map.tolist())):
			idx_j = np.where(label_map == j_s)[0].tolist()
			T[i_s - 1, j_s - 1] = sum(weight[idx_j])
	
	T = normalize(T, axis=1, norm='l1')

	if filter_small_prop:
		thres = 0.05
		T[T < thres] = 0
		T = normalize(T, axis=1, norm='l1')

	return T


