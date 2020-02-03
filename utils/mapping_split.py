# define the cell split
import os
import pdb
import numpy as np
from sklearn.mixture import GaussianMixture

from utils.util_tools import *
from utils.mapping_tools import *
from utils.mapping_1st import get_row_mapping_from_mat
from utils.mapping_2nd import get_2nd_mapping

def get_degree(pts_s, pts_t):
	## get the degree of
	dif = pts_t - pts_s
	comp = np.array(dif[:,0]) + np.array(dif[:,1]) * complex(0,1)
	return np.angle(comp, deg=True)


def get_degree_dist(pts_s, pts_t):
	## get both degree and distance of the point movement
	dif = pts_t - pts_s
	dist_st = np.linalg.norm(dif, axis=1)
	deg_st = np.angle(
		np.array(dif[:,0]) + np.array(dif[:,1]) * complex(0,1), deg=True)
	return dist_st, deg_st


def get_cell_segment_partial(pts, pts_assigned, options):
	""" get the cell segmentation based on the partially assigned labels
	Some possible methods: 
		use the previous segmentation and the pixel-wise mapping
		use the previous image segmentation
		use the original img to do a histogram equalization
		statistical methods like KNN and GMM
		watershed segmentation
		keep the boundary consistent
	I: 
		pts: n x 2 pts to be assigned
		pts_assigned: n x 3 pts that has been assigned
		options fields:
			img_ori: original images
			img_seg_prev: previous segmentation
	O:
		pts_with_new_labels: n x 3 pts
	"""
	## -------- step 1: training with knn --------
	clf = KNeighborsClassifier(n_neighbors=options.seg_knn_neigh)
	clf.fit(pts_assigned[:,:(-1)], pts_assigned[:,(-1)].flatten())
	label_te = clf.predict(pts)
	ratio = len(label_te[label_te > 0])/len(label_te)
	ratio_thres = 0.2
	if ratio < ratio_thres or ratio > 1 - ratio_thres:
		prob = clf.predict_proba(pts)[:,1]
		label_te = (prob >= np.percentile(prob, 50)).astype(int)
	return np.hstack((pts, label_te.reshape(-1,1)))



def split_cells_motion(pts_s, pts_t, T, options):
	""" split cells according to the pixel motion information
	"""
	list_subset = get_subset_transform(T)
	n_row, _ = T.shape
	T_adj = np.copy(T)
	clf1 = GaussianMixture(n_components=1)
	clf2 = GaussianMixture(n_components=2)
	n = max(10, options.n_pixels_per_cell // 3)
	n2 = max(10, options.n_pixels_per_cell // 10)
	img_t_adj = pts2label_img(pts_t, options.size_img)

	for lst_s, lst_t in list_subset:
		if not (len(lst_s) == 1 and len(lst_t) == 1):
			continue		
		## select pts
		pts_s_group, _, _ = select_pts_in_class(pts_s[:,:(-1)], 
			pts_s[:,(-1)], lst_s, options)
		pts_t_group, _, _ = select_pts_in_class(pts_t[:,:(-1)], 
			pts_t[:,(-1)], lst_t, options)

		len_s, len_t = len(pts_s_group), len(pts_t_group)
		
		if not (len_s > n and len_t > n and (len_t/len_s > options.divide_ratio)):
			continue
		# compute GW distance
		gw  = get_row_mapping_from_mat(get_2nd_mapping(pts_s_group, pts_t_group, options)[0])
		# select maximum mapping proportion for each point
		gw2 = np.flipud(np.sort(gw.view('f8,f8,f8'), order=['f2'], axis=0).view(np.float))
		gw3 = gw2[np.unique(gw2[:, 1].astype(int), return_index=True)[1]]

		## get the movement direction, delete points with zero movement
		"""
		degree = get_degree(pts_s_group[gw3[:,0].astype(int)], pts_t_group[gw3[:,1].astype(int)])
		degree_no_zero = [i for i in degree.tolist() if (i > 0.1 or i < -0.1)]
		idx = np.union1d(np.where(degree > 0.1)[0], np.where(degree < -0.1)[0])
		"""
		dist_st, degree_st = get_degree_dist(
			pts_s_group[gw3[:,0].astype(int)], pts_t_group[gw3[:,1].astype(int)])
		degree_no_zero = degree_st[dist_st > 0]
		idx = np.where(dist_st > 0)[0]
		if len(degree_no_zero) <= n2:
			continue
		
		## add a virtual row to fit the Gaussian Mixtures
		X = np.transpose(np.vstack((degree_no_zero, np.zeros_like(degree_no_zero))))
		## shift 360 degrees to avoid +- 180 degree cases
		X_shift = np.copy(X)
		X_shift[X_shift[:,0] < 0, 0] += 360
		## fit normal distributions
		clf1.fit(X_shift)
		clf2.fit(X_shift)

		bool_fit1 = (clf1.bic(X_shift) > clf2.bic(X_shift))
		clf1.fit(X)
		clf2.fit(X)
		v1, v2 = clf2.covariances_[0,0,0], clf2.covariances_[1,0,0]
		mean_dif = min(clf2.means_[1,0] - clf2.means_[0,0], 360 - abs(clf2.means_[0,0] - clf2.means_[1,0]))
		label_pred = clf2.predict(X)

		if ((bool_fit1 and clf1.bic(X) > clf2.bic(X)) 
			and (v1 > options.cell_variance_min and v2 > options.cell_variance_min) 
			and (v1 < options.cell_variance_max and v2 < options.cell_variance_max) 
			and (mean_dif > options.cell_angle) 
			and (sum(label_pred) > n2 and sum(label_pred) < len(label_pred) - n2)):
			pts_t_assigned = np.hstack((pts_t_group[idx], label_pred.reshape(-1,1)))
			pts_t_group_adj = get_cell_segment_partial(pts_t_group, pts_t_assigned, options)
			n0 = len(pts_t_group_adj[pts_t_group_adj[:,2] == 0,2])
			n1 = len(pts_t_group_adj[pts_t_group_adj[:,2] == 1,2])
			r = n0/(n0 + n1)

			if r < options.cell_proportion_max and r > (1.0 - options.cell_proportion_max):
				## update pts_t
				new_pts = array_to_loc(pts_t_group_adj[pts_t_group_adj[:,2] == 1,:2])
				n_to_assign = img_t_adj.max() + 1
				img_t_adj[new_pts] = n_to_assign
				## update mappings T
				l_s, l_t = lst_s[0] - 1, lst_t[0] - 1
				p = len(new_pts[0])/len(pts_s_group)
				T_adj = np.hstack((T_adj, np.zeros((n_row, 1))))
				T_adj[l_s, -1], T_adj[l_s, l_t] = p, 1 - p

	pts_t_adj = label_img2pts(img_t_adj)
	return T_adj, pts_t_adj