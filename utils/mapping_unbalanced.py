import ot
import numpy as np
from sklearn.preprocessing import normalize

from utils.mapping_tools import *
from utils.mapping_1st import get_sinkhorn_eps, get_row_mapping_from_mat

def get_max_transcost_between_cells(label_s, label_t, M, T,
	C_cost=0.5, max_cost_lowerbd=2):
	"""Dynamically compute the distance between two cells in the image.

	Args:
		max_cost_lowerbd: lower bound of the transition cost between virtual
			and actual points
	"""
	max_cost = M.max()
	for item_s in np.unique(label_s):
		for item_t in np.unique(label_t):
			if T[item_s - 1, item_t - 1] > 0:
				M_st = M[label_s == item_s, :]
				M_st = M_st[:, label_t == item_t]
				min_cost = np.min(M_st) # The minimal cost of transfering between two cells
				if min_cost > max_cost_lowerbd and min_cost < max_cost: 
					max_cost = min_cost
	# make the max_cost of virtual pts a little less than the minimal transportation cost
	max_cost = max(C_cost * max_cost, max_cost_lowerbd)
	return max_cost


def get_unbalanced_mapping(pts_s, pts_t, label_s, label_t, T, options):
	""" Get the unbalanced mapping between 2 point sets
	This process adds virtual points, and then use the first-order information
	to estimate the OT.
	"""
	# get # of pts to add to the both sides of pts
	n_s_new, n_t_new = add_virtual_pts(pts_s, pts_t)
	n_s, n_t = pts_s.shape[0], pts_t.shape[0]
	# Get the boundary constraints
	p_s, p_t = np.ones((n_s_new,))/n_s_new, np.ones((n_t_new,))/n_t_new

	M = ot.dist(pts_s, pts_t, metric='euclidean') # cost matrix
	max_cost = get_max_transcost_between_cells(label_s, label_t, M, T)
	
	# expand the cost matrix 	
	M = np.vstack([M, np.ones((n_s_new - n_s, n_t)) * max_cost])
	M = np.hstack([M, np.ones((n_s_new, n_t_new - n_t)) * max_cost])
	M[n_s:n_s_new,n_t:n_t_new] = 0
	# matrix normalization
	M /= max(M.max(),1e-6)

	if hasattr(options, 'eps_0'):
		G, eps = get_sinkhorn_eps(p_s, p_t, M,
			eps_0=options.eps_0, bool_fix_eps=options.bool_fix_eps)
	else:
		G, eps = get_sinkhorn_eps(p_s, p_t, M)

	return G[:n_s, :n_t], eps



def add_virtual_pts(pts_s, pts_t):
	""" Get the # of pts added in unbalanced mapping.
	Now we use the max # of source pts and target pts.
	"""
	pts_all  = np.concatenate((pts_s, pts_t), axis=0)
	n_unique = np.unique(pts_all, axis=0).shape[0]
	return n_unique, n_unique



def adjust_unbalanced_mapping(pts_s, pts_t, label_s, label_t, T, options, max_iter=4):
	""" Apply the unbalanced mapping to all the groups.
	"""
	## option of output mapping matrix or not
	if hasattr(options, 'output_eps'):
		output_eps = options.output_eps
	else:
		output_eps = False

	eps_list = []
	n_iter = 0
	bool_converge = False
	output_G = False

	while n_iter <= max_iter:
		# if the final round is done and G is output, break the iterations
		if output_G:
			break

		if bool_converge or n_iter == max_iter:
			output_G = True			
		
		list_subset = get_subset_transform(T)
		G_total = np.array([]).reshape(0, 3)
		n_iter += 1
		bool_converge = True
		
		for i_group in range(len(list_subset)):
			label_s_group, label_t_group = list_subset[i_group]
			if (len(label_s_group) * len(label_t_group) > 1):
				pts_s_group, labelpts_s_group, idx_s_group = select_pts_in_class(
					pts_s, label_s, label_s_group, options)
				pts_t_group, labelpts_t_group, idx_t_group = select_pts_in_class(
					pts_t, label_t, label_t_group, options)
								
				G, eps = get_unbalanced_mapping(
					pts_s_group, pts_t_group, labelpts_s_group, labelpts_t_group, T, options)
				T_new = get_label_mapping(labelpts_s_group.astype(int), labelpts_t_group.astype(int), G)
				T, bool_diff = compare_mat(T_new, T, label_s_group, label_t_group)
				bool_converge = min(1 - bool_diff, bool_converge)

				eps_list.append(eps)

			elif len(label_s_group) and len(label_t_group) and (output_G or output_eps):
				pts_s_group, labelpts_s_group, idx_s_group = select_pts_in_class(
					pts_s, label_s, label_s_group, options)
				pts_t_group, labelpts_t_group, idx_t_group = select_pts_in_class(
					pts_t, label_t, label_t_group, options)
				# Compute the general mapping
				M = ot.dist(pts_s_group, pts_t_group, metric='euclidean')
				M /= max(M.max(),1e-6)
				# Point-wise probability
				p_s = np.ones((len(pts_s_group),)) / len(pts_s_group)
				p_t = np.ones((len(pts_t_group),)) / len(pts_t_group)

				if hasattr(options, 'eps_0'):
					G, eps = get_sinkhorn_eps(p_s, p_t, M,
						eps_0=options.eps_0, bool_fix_eps=options.bool_fix_eps)
				else:
					G, eps = get_sinkhorn_eps(p_s, p_t, M)

				eps_list.append(eps)
			else:
				continue

			if output_G:
				# Update the idx row-wise alignment
				G_map_row = get_row_mapping_from_mat(G)
				# Update the idx to all the pts
				G_map_row[:,0] = idx_s_group[G_map_row[:,0].astype(int).tolist()]
				G_map_row[:,1] = idx_t_group[G_map_row[:,1].astype(int).tolist()]
				G_total = np.vstack([G_total, G_map_row])
		
		# Adjust mapping 
		thres = 0.05
		T[T < thres] = 0
		T = normalize(T, axis=1, norm='l1')
		
	if output_eps:
		return T, G_total, eps_list
	return T, G_total