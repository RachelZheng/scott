## the first order mapping 
import ot
import warnings
import numpy as np
from sklearn.preprocessing import normalize

from utils.mapping_tools import *

def get_1st_mapping(pts_s, pts_t, options):
	M = ot.dist(pts_s, pts_t, metric ='euclidean')
	M /= max(M.max(),1e-6)
	# Point-wise probability
	p_s = np.ones((len(pts_s),))/len(pts_s)
	p_t = np.ones((len(pts_t),))/len(pts_t)
	if hasattr(options, 'eps_0'):
		G, eps = get_sinkhorn_eps(p_s, p_t, M, 
			eps_0=options.eps_0, bool_fix_eps=options.bool_fix_eps)
	else:
		G, eps = get_sinkhorn_eps(p_s, p_t, M)
	return G, eps


def adjust_1st_mapping(pts_s, pts_t, label_s, label_t, T, options):
	""" Adjust the 1st order mapping of the data using sinkhorn method
	"""
	if hasattr(options, 'output_G'):
		output_G = options.output_G # Update mapping G on every iteration
	else:
		output_G = False

	if hasattr(options, 'output_eps'):
		output_eps = options.output_eps
	else:
		output_eps = False

	max_iter = 2
	options.n_pts = 5000

	for n_iter in range(max_iter):
		eps_list = []
		print("Begin the %d th iteration in the adjustment ...\n"%(n_iter + 1))
		list_subset = get_subset_transform(T)
		G_total = np.array([]).reshape(0, 3)
		for item_group in list_subset:
			label_s_group, label_t_group = item_group

			bool_compute = (output_G or output_eps) and (len(label_s_group) * len(label_t_group) > 1)
			if not bool_compute:
				continue
			# Get the subset of points
			pts_s_group, labelpts_s_group, idx_s_group = select_pts_in_class(
				pts_s, label_s, label_s_group, options)
			pts_t_group, labelpts_t_group, idx_t_group = select_pts_in_class(
				pts_t, label_t, label_t_group, options)
			# Compute the mapping between the subset of pts using sinkhorn method
			G_map, eps = get_1st_mapping(pts_s_group, pts_t_group, options)
			T_new = get_label_mapping(labelpts_s_group, labelpts_t_group, G_map)
			T, _  = compare_mat(T_new, T, label_s_group, label_t_group)
			eps_list.append(eps)

			if not output_G:
				continue
			# Update the idx row-wise alignment
			G_map_row = get_row_mapping_from_mat(G_map)
			# Update the idx to all the pts
			G_map_row[:,0] = idx_s_group[G_map_row[:,0].astype(int).tolist()]
			G_map_row[:,1] = idx_t_group[G_map_row[:,1].astype(int).tolist()]
			G_total = np.vstack([G_total, G_map_row])

		# Adjust mapping 
		thres = 0.05
		T[T < thres] = 0
		T = normalize(T, axis=1, norm='l1')

	# Compute the cost per point
	if output_eps:
		return T, G_total, eps_list
	return T, G_total


def get_sinkhorn_eps(p_s, p_t, M, eps_0=1e-3, bool_fix_eps=False):
	# Get the suitable epsilon value in the computation
	if bool_fix_eps:
		eps = eps_0
	else:
		flag_stop = False
		eps = eps_0
		for i in range(2):
			if not flag_stop:
				try:
					with warnings.catch_warnings(record=True) as w:
						warnings.simplefilter("always")
						G1 = ot.sinkhorn(p_s, p_t, M, eps)
					if not len(w):
						eps /= 2
					else:
						flag_stop = True
				except:
					flag_stop = True
					eps *= 2

	G = ot.sinkhorn(p_s, p_t, M, eps)
	return G, eps



def get_row_mapping_from_mat(G, thres = 1e-6):
	""" Get the row-wise alignment from the matrix G
	"""
	G_total = np.array([]).reshape(0, 3)
	n_row, n_col = G.shape
	for i in range(n_row):
		if np.any(G[i, :] > thres):
			for j in range(n_col):
				if G[i, j] <= thres:
					continue
				addon = np.array([i, j, G[i,j]])
				G_total = np.vstack([G_total, addon])
	return G_total