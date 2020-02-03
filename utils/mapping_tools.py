## collections of mapping functions

import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from utils.util_tools import *


def compare_mat(T1, T2, label_s, label_t, thr=0.1):
	""" Compare the cell-wise mappingmatrix between T1 and T2
	in label_s rows and label_t columns
	"""
	label_s_adj,label_t_adj = [i - 1 for i in label_s], [i - 1 for i in label_t]
	T1_sub, T2_sub = T1[label_s_adj,:], T2[label_s_adj,:]
	T1_sub, T2_sub = T1_sub[:,label_t_adj], T2_sub[:,label_t_adj]
	diff = abs(T1_sub - T2_sub)
	if np.linalg.norm(diff) > thr:
		bool_diff = True
	else:
		bool_diff = False
	n_row2, n_col2 = T2.shape
	n_row1, n_col1 = T1.shape
	# Update T2 w/ T1
	for i in label_s_adj:
		for j in label_t_adj:
			T2[i,j] = T1[i,j]
			if n_row2 > n_row1:
				T2[n_row1:, j] = 0
		if n_col2 > n_col1:
			T2[i, n_col1:] = 0
	return T2, bool_diff



def post_adjustment(pts_s, pts_t, T, opt):
	"""
	function for handling some error in cell grouping with multiscale OT
	I:
		pts_s, pts_t: n_s/n_t x 3
		T: transition matrix
	O: 
		T_adj: adjusted transition matrix
	"""
	img_s, img_t = pts2label_img(pts_s, opt.size_img), pts2label_img(pts_t, opt.size_img)
	list_subset = get_subset_transform(T)
	T_adj = np.copy(T)
	for groups in list_subset:
		label_s_group, label_t_group = groups
		## check the target image have corresponding cells or not
		if len(label_s_group):
			xs_group, labelpts_s_group, _ = select_pts_in_class(
				pts_s[:,:(-1)], pts_s[:,(-1)], label_s_group, opt)
			img_t_labels = img_t[(xs_group[:,0], xs_group[:,1])]
			img_t_labels = [i for i in img_t_labels if (i not in set(label_t_group) and i)]
			img_t_labels = Counter(img_t_labels).most_common()
			## check the labels overlap with the source pts
			for j_t in img_t_labels:
				if j_t[1] < 20:
					break
				l_t = j_t[0]
				n_l_t = max(len(np.where(pts_t[:,(-1)] == l_t)[0]), 1)
				if j_t[1]/n_l_t > 0.2:
					for l_s in label_s_group:
						T_adj[l_s - 1, l_t - 1] = 1

		if len(label_t_group):
			xt_group, labelpts_t_group, _ = select_pts_in_class(
				pts_t[:,:(-1)], pts_t[:,(-1)], label_t_group, opt)
			img_s_labels = img_s[(xt_group[:,0], xt_group[:,1])]
			img_s_labels = [i for i in img_s_labels if (i not in set(label_s_group) and i)]
			img_s_labels = Counter(img_s_labels).most_common()
			## check the labels overlap with the target pts
			for i_s in img_s_labels:
				if i_s[1] < 20:
					break
				l_s = i_s[0]
				n_l_s = max(len(np.where(pts_s[:,(-1)] == l_s)[0]), 1)
				if i_s[1] / n_l_s > 0.2:
					for l_t in label_t_group:
						T_adj[l_s - 1, l_t - 1] = 1

	return T_adj



def post_adjustment2(pts_s, pts_t, T, opt):
	img_s, img_t = pts2label_img(pts_s, opt.size_img), pts2label_img(pts_t, opt.size_img)
	list_subset = get_subset_transform(T)
	T_adj = np.copy(T)
	idx_s = np.where(np.max(T, axis=1) == 0)[0]
	idx_t = np.where(np.max(T, axis=0) == 0)[0]
	for i_s in idx_s:
		cnt = Counter(img_t[img_s == i_s + 1]).most_common()
		if cnt[0][0] > 0:
			T_adj[i_s, cnt[0][0] - 1] = 1
		elif len(cnt) > 1 and cnt[1][1] > opt.n_pixels_per_cell:
			T_adj[i_s, cnt[1][0] - 1] = 1
	for i_t in idx_t:
		cnt = Counter(img_s[img_t == i_t + 1]).most_common()
		if cnt[0][0] > 0:
			T_adj[cnt[0][0] - 1, i_t] = 1
		elif len(cnt) > 1 and cnt[1][1] > opt.n_pixels_per_cell:
			T_adj[cnt[1][0] - 1, i_t] = 1
	return T_adj


def get_subset_transform(T):
	""" Get the sub-group cell transform matrix T
	Input: l1 * l2 transform matrix T matrix
	Output: list of indexes, like [[[1,2],[2]], [[3, 4], [1]]]
		indicates classes 1, 2 in source transfers to class 2 in target
		classes 3,4 in source transfers to class 1 in target
	"""
	idx_s_0, idx_t_0 = set(range(T.shape[0])), set(range(T.shape[1])) 
	list_result = []
	flag_stop   = False # Flag indicating all the cells are classified	
	while not flag_stop:
		# Initialization
		if len(idx_s_0):
			idx_s   = set([idx_s_0.pop()])
			idx_t   = set(np.nonzero(T[list(idx_s),:][0,:])[0])
			idx_t_0 = idx_t_0 - idx_t
			flag_add_finish = False   # If it is True then stops
			flag_s  = 's' # The next element to add. Options: s and t
		else:
			idx_t   = set([idx_t_0.pop()])
			idx_s   = set(np.nonzero(T[:,list(idx_t)][:,0])[0])
			idx_s_0 = idx_s_0 - idx_s
			flag_add_finish = False   # If it is True then stops
			flag_s  = 't' # The next element to add. Options: s and t
		while not flag_add_finish:
			# It is term to add source classes
			if flag_s == 's':
				idx_s_new = []
				for ele_t in list(idx_t):
					idx_s_new += (np.nonzero(T[:,ele_t])[0]).tolist()
				idx_s_new = set(idx_s_new) # The new source classes
				if idx_s_new == idx_s: # If we get same classes
					flag_add_finish = True # Then finish adding this round
				elif len(idx_s_new) > len(idx_s): # Add more classes
					flag_s  = 't'
					idx_s   = idx_s_new
					idx_s_0 = idx_s_0 - idx_s
				else:
					# Some classes may disappear
					flag_add_finish = True
			else:
				# It is term to add target classes
				idx_t_new = []
				for ele_s in list(idx_s):
					idx_t_new += (np.nonzero(T[ele_s,:])[0]).tolist()
				idx_t_new = set(idx_t_new) # The new target classes
				if idx_t_new == idx_t: # If we get same classes
					flag_add_finish = True # Then finish adding this round
				elif len(idx_t_new) > len(idx_t):  # Add more classes
					flag_s = 's'
					idx_t  = idx_t_new
					idx_t_0 = idx_t_0 - idx_t
				else:
					flag_add_finish = True
		# Add 1 for the source and target classes
		if len(idx_s) != 0 or len(idx_t) != 0:
			list_result.append([[(s + 1) for s in list(idx_s)],[(t + 1) for t in list(idx_t)]])
		
		if len(idx_s_0) == 0 and len(idx_t_0) == 0:
			flag_stop = True
	# Add one for every class
	return list_result



def adjust_mask(img_s, img_t, img_mask_s, T, n_to_assign):
	""" adjust target image segmentation according to the previous mask adjustment
	"""
	lst = get_subset_transform(T)
	img_mask_t = np.zeros_like(img_mask_s)
	for l_s_lst, l_t_lst in lst:
		## appeared cells
		if not len(l_s_lst):
			img_mask_t[img_t == l_t_lst[0]] = n_to_assign
			n_to_assign += 1
		## non-disappeared cells
		elif len(l_t_lst):
			l_total = np.array([])
			for l_s in l_s_lst:
				l_total = np.concatenate((l_total, img_mask_s[img_s == l_s]))
			n_s = Counter(l_total).most_common()[0][0]
			## 1-to-1 cell mapping
			if len(l_t_lst) == 1:
				img_mask_t[img_t == l_t_lst[0]] = n_s
			## 1-to-many cell mapping
			else:
				for l_t in l_t_lst:
					img_mask_t[img_t == l_t] = n_to_assign
					n_to_assign += 1
	return img_mask_t, n_to_assign



def adj_img_seg_prev(pts_s, pts_t, T, G, options):
	""" Adjust image segmentation with the mapping from the previous frame.
	Assumption: cell will not merge.
	Input:
		pts_s, pts_t: positions and labels. Dimension: n x 3.
			First 2 cols are locations of pixels and the third col is the label.
		T: label-wise mapping matrix, where row in the source label and col is target label.
		G: total alignment from source to target.
	Output:
		pts_t_adj: positions and labels of the target frame.
		T_adj: The adjusted mapping matrix T
	"""
	idxes2seg_class_t = np.where(np.sum(T, axis = 0) > 1.5)[0]
	# If more than 1.5 cells merged together, then
	# There would be 2 cells mistakely merge to 1 cell
	G_copy = _delete_duplicate_trans_t(G) # The label change indexes
	G_label   = np.copy(G_copy)
	pts_t_adj = np.copy(pts_t) # pts after adjustment
	N_class_t = int(max(pts_t_adj[:,-1])) # Total # of the clusters
	label_s, label_t = pts_s[:,-1], pts_t[:,-1] # Labels of target
	G_label[:,0] = label_s[G_label[:,0].astype(int)]
	G_label[:,1] = label_t[G_label[:,1].astype(int)]
	G_label[:,:2] -= 1 

	# Build a knn model to assign those pts
	neigh = KNeighborsClassifier(n_neighbors=3)
	for idx_class_t in idxes2seg_class_t:
		idx_map_t = np.where(G_label[:,1] == idx_class_t)[0] # Row index of row-wise mapping
		idx_class_s_list = np.where(T[:, idx_class_t] != 0)[0]
		idx_pts_t = np.where(pts_t[:,-1] == (idx_class_t + 1))[0]
		idx_pts_t_left = set(np.copy(idx_pts_t))
		pts_assigned = np.array([]).reshape(0, 2)
		label_assigned = []
		N_class_t_sub = 0
		idx_t_class_list = [] # Class-wise index in pts_t
		for idx_class_s in idx_class_s_list:
			idx_map_s = np.where(G_label[:,0] == idx_class_s)[0]
			idx_map_st = np.intersect1d(idx_map_s, idx_map_t) # Row index of mapping from S to T
			if len(idx_map_st) < 5:
				# if mapping points pairs < 5, we consider these 2 components have no mappings
				T[idx_class_s, idx_class_t] = 0
			else:
				idx_pts_from_s = G_copy[idx_map_st,1].astype(int)
				pts_assigned   = np.vstack([pts_assigned, pts_t[idx_pts_from_s, :(-1)]])
				label_assigned += [N_class_t_sub] * len(idx_pts_from_s)
				N_class_t_sub  += 1
				idx_t_class_list.append(idx_pts_from_s)
				idx_pts_t_left -= set(G_copy[idx_map_st,1].tolist())
		# --- Step 1: find the pts from class T that has not been aligned ---
		# and assign them to the new position
		if len(idx_pts_t_left) and len(pts_assigned):
			# print('idx_class_t: %d\n'%(idx_class_t))
			neigh.fit(pts_assigned, label_assigned)  # Fit knn model
			idx_pts_t_left = np.array(list(idx_pts_t_left)) # Indexes of pts that have not been assigned
			pts_left = pts_t[idx_pts_t_left, :(-1)]
			classes_left = neigh.predict(pts_left)
			# Re-arrange The pts
			for idx_class in range(N_class_t_sub):
				# Update the lists
				idx_old = idx_t_class_list[idx_class]
				idx2add = idx_pts_t_left[np.where(classes_left == idx_class)[0]]
				idx_t_class_list[idx_class] = np.append(idx_old, idx2add)
		# --- Step 2: Assign these classes to the new label ---
		for i in range(1, len(idx_t_class_list)):
			idx2change = idx_t_class_list[i]
			N_class_t  = T.shape[1] + 1
			# Modify the labels 
			pts_t_adj[idx2change, -1] = N_class_t
			# Modify the mapping matrix T
			idx_s_change = idx_class_s_list[i]
			T = np.hstack([T, np.zeros((T.shape[0],1))])
			T[idx_s_change, -1] = T[idx_s_change, idx_class_t]
			T[idx_s_change, idx_class_t] = 0

	return T, pts_t_adj


def _delete_duplicate_trans_t(G):
	""" Delete duplicated transition of T.
	"""
	if G.shape[1] == 4:
		G_single_t = np.sort(G.view('i8,i8,i8,i8'), order=['f1','f2'], axis=0).view(np.float)
	elif G.shape[1] == 3:
		G_single_t = np.sort(G.view('i8,i8,i8'), order=['f1','f2'], axis=0).view(np.float)
	else:
		G_single_t = G
		print('Dimension incorrect!!')
	idx_dup  = G_single_t[:, 1]
	idx_diff = np.append(idx_dup[1:] - idx_dup[:-1], np.array([1]))
	idx_keep = np.where(idx_diff)[0]
	G_single_t = G_single_t[idx_keep, :]
	return G_single_t


def get_label_mapping(label_s, label_t, Gs, options=dict()):
	""" Get the mapping between source and target cells
	Input: 
		n1 * 1 labels for source, n2 * 1 labels for target, 
		n1 * n2 transfer matrix
	Output: l1 * l2 matrix with relative transformation, which is row-normalized
	### PLEASE BE CAREFUL: OUR LABELS BEGIN WITH 1
	"""
	label_s = np.array(label_s).astype(int)
	label_t = np.array(label_t).astype(int)
	l_s_max = int(label_s.max())
	l_s_range = np.unique(label_s).tolist()
	l_t = int(label_t.max())
	G = Gs / max(Gs.max(), 1e-6)
	T_label = np.zeros((l_s_max, l_t))
	thres = 1e-6
	G[G < thres] = 0
	if hasattr(options, 'row_normalize') and options.row_normalize:
		ROW_NORMALIZE_THRES = 0.2
		ROW_NORMALIZE = True
	else:
		ROW_NORMALIZE = False
		
	for i_s in l_s_range:
		idx_class  = np.where(label_s == i_s)[0] # The index of class i_s from the source
		G_class    = G[idx_class,:] # Sub-transformation matrix
		idx_target = np.nonzero(G_class)[1].tolist() # matched points idxes
		idx_target = sorted(set(idx_target), key=idx_target.index) # non-duplicated sorted matched points idxes
		weight_M   = np.sum(G_class[:,idx_target], axis=0) # Take the sum of every column
		weight_M   /= max(np.sum(weight_M), 1e-6) # Normalize
		label_matched = label_t[idx_target]
		label_matched_unique = np.unique(label_matched).tolist()
		if ROW_NORMALIZE: 
			# if we choose that option, then target group that only small partial be hit should be deleted
			for i_t in label_matched_unique:
				hit_mat   = G_class[:,label_t == i_t]
				pts_total = hit_mat.shape[1]
				hit_pts   = np.unique(np.where(hit_mat > 0.1)[1])
				ratio     = float(len(hit_pts)) / pts_total
				# if less than 0.2 part of the cell get transfer from that cell 
				if ratio < ROW_NORMALIZE_THRES:
					label_matched_unique.remove(i_t)
		for i_t in label_matched_unique:
			idx_class_target = np.where(label_matched == i_t)[0]
			T_label[i_s - 1, i_t - 1] = np.sum(weight_M[idx_class_target])
	return T_label



def adj_img_seg_gt(img, img_gt, options):
	"""adjust the image segmentation according to the ground truth
	In:
		img, 16-bit mask of segmentation
		img, 16-bit mask of ground truth

	Out:
		img_new. 16-bit mask of adjusted segmentation  
	"""
	dict_gt_to_seg = dict()
	dict_seg_to_gt = dict()
	unassigned = [] # unassigned segmentation
	img_new = np.copy(img)
	shapes = options.size_img

	## state 1: cell and ground-truth has overlaps
	for l in range(1, img.max() + 1):
		if l not in dict_seg_to_gt:
			msks = [i for i in img_gt[img == l] if i]
			n_msks = len(np.unique(msks))

			if n_msks == 0:
				unassigned.append(l)
			elif n_msks == 1:
				# 1 target cell, further discuss according to the source cell number
				l_gt = msks[0]
				msks_seg = [i for i in img[img_gt == l_gt] if i]

				if len(np.unique(msks)) == 1:
					# 1 cell to 1 ground-truth cell
					dict_gt_to_seg[l_gt] = l
					dict_seg_to_gt[l] = l_gt
				else:
					# 2 cells to 1 ground-truth cells: assign to the largest proportion of the cell
					l_assign = Counter(msks_seg).most_common()[0][0]
					dict_gt_to_seg[l_gt] = l_assign
					dict_seg_to_gt[l_assign] = l_gt
					if (l_assign != l):
						unassigned.append(l)
				
			else:
				## 1 cell to 2+ gt cells
				img_new, dict_gt_to_seg, dict_seg_to_gt = _adj_cell_seg(
					img_new, img_gt, l, dict_gt_to_seg, dict_seg_to_gt, options)
	
	## state 2: cell and ground-truth has no overlaps. 
	# Try to match unassigned ground truth cells with the unassigned cells
	unassigned_gt = set(np.unique(img_gt)) - set(dict_gt_to_seg.keys()) - {0}
	thres_dist = 5
	for l in unassigned:
		pts = np.where(img == l)
		rng = [[max(pts[0].min() - thres_dist, 0), min(pts[0].max() + thres_dist, shapes[0])],\
			[max(pts[1].min() - thres_dist, 0), min(pts[1].max() + thres_dist, shapes[1])]]
		msks = [i for i in img_gt[rng[0][0]:rng[0][1], rng[1][0]:rng[1][1]].flatten() if (i and i in unassigned_gt)]
		l_gt_candidates = Counter(msks).most_common()
		if len(l_gt_candidates) >= 1:
			l_gt = l_gt_candidates[0][0]
			dict_gt_to_seg[l_gt] = l
			dict_seg_to_gt[l] = l_gt

	## Add other cells into the images
	return img_new, dict_gt_to_seg, dict_seg_to_gt



def _adj_cell_seg(img, img_gt, idx_to_adj, dict_gt_to_seg, dict_seg_to_gt, options):
	""" adjust cell-part segmentation with ground-truth information
	I: 
		img: n1 x n2 np.array with value to be the segmentation index
		img_gt: n1 x n2 np.array with value to be the ground-truth index in the text record
		idx_to_adj: index to be adjusted
		options: custom selection with methods
	O: 
		img: new img after adjustment
	"""
	method = 'knn' # or kmeans, or other methods
	## prepare training and testing samples
	# testing samples
	locs_te = np.where(img == idx_to_adj)
	locs_te_arr = np.transpose(np.vstack((locs_te[0], locs_te[1])))
	idx_max = img.max()
	labels = np.unique(img_gt[locs_te]).tolist()
	labels.remove(0)
	# training samples
	locs_tr_arr = np.empty((0,2))
	label_tr = []
	for l in labels:
		locs_tr = np.where(img_gt == l)
		locs_tr_arr_add = np.transpose(np.vstack((locs_tr[0], locs_tr[1])))
		locs_tr_arr = np.vstack((locs_tr_arr, locs_tr_arr_add))
		label_tr += [l] * len(locs_tr[0])
	## training process
	## for now only knn and GMM methods are supported
	if method == 'knn':
		n_neighbors = min(3, len(locs_tr_arr))
		clf = KNeighborsClassifier(n_neighbors=n_neighbors)
		clf.fit(locs_tr_arr, label_tr)
		label_te = clf.predict(locs_te_arr)
	elif method == 'GMM':
		clf = GaussianMixture(n_components=len(labels), covariance_type='full')
		# re-assign gmm labels
		clf.fit(locs_tr_arr, label_tr)
		idx_te = clf.predict(locs_te_arr)
		labels2 = []
		for j in range(idx_te.max() + 1):
			locs_te_arr_group = locs_te_arr[np.where(idx_te == j)[0]]
			label_gt_item = img_gt[array_to_loc(locs_te_arr_group[:,:2])]
			## ground-truth index
			labels2.append(Counter([k for k in label_gt_item if k]).most_common()[0][0])
		labels2 = np.array(labels2)
		label_te = labels2[idx_te]
	## assign new labels
	for l in labels:
		if l != labels[0]:
			n_to_assign = img.max() + 1
			locs_to_assign = locs_te_arr[label_te == l,:]
			locs_to_assign = (locs_to_assign[:,0], locs_to_assign[:,1])
			img[locs_to_assign] = n_to_assign
		else:
			n_to_assign = idx_to_adj
		# update the mapping dictionary
		dict_seg_to_gt[n_to_assign] = l
		"""
		if l in dict_gt_to_seg:
			dict_gt_to_seg[l] += [n_to_assign]
		else:
			dict_gt_to_seg[l] = [n_to_assign]
		"""
		dict_gt_to_seg[l] = n_to_assign
	return img, dict_gt_to_seg, dict_seg_to_gt


def T_normalize(T, thres=0.2):
	T = normalize(T, axis=1, norm='l1')
	T[T < thres] = 0
	T = normalize(T, axis=1, norm='l1')
	lst = get_subset_transform(T)
	for item in lst:
		nrow_arr, ncol_arr = item
		if len(nrow_arr) == 2 and len(ncol_arr) == 2:
			nrow_arr, ncol_arr = np.array(nrow_arr)-1, np.array(ncol_arr) - 1
			T_sub = T[nrow_arr[:, None], ncol_arr]
			T[nrow_arr[:, None], ncol_arr] = _update_T(T_sub)
	return T


def _update_T(T):
	""" update matrix T
	"""
	## get row max
	bool_stop = False
	while not bool_stop:
		ind = np.where(T == T.max())
		ind = [ind[0][0], ind[1][0]]
		c, r = np.where(T[:,ind[1]])[0], np.where(T[ind[0]])[0]
		if len(c) <= 1 and len(r) <= 1:
			bool_stop = True
		else:
			T[:,ind[1]] = 0
			T[ind[0]] = 0
			T[ind[0], ind[1]] = 1
		T = normalize(T, axis=1, norm='l1')
	return T


def update_T_with_adj_img(T, img_old, img_adj):
	""" update transfer matrix T with adjusted point masks in source
	previous update_T_with_gt function
	
	Args: 
		T: original transfer matrix
		img_old: n1 x n2 index array
		img_adj: adjusted n1 x n2 index array
	
	Returns: 
		T_adj: adjusted transfer matrix
	"""
	img_old, img_adj = img_old.astype(int), img_adj.astype(int)
	idx_old, idx_adj = set(np.unique(img_old)), set(np.unique(img_adj))
	n_row_adj = int(img_adj.max())
	n_row_old, n_col_old = T.shape
	T_adj = np.zeros((n_row_adj, n_col_old))
	T_adj[:n_row_old,:] = T
	## delete not labeled cells
	for i in (idx_old - idx_adj):
		T_adj[i - 1, :] = 0
	## re-assign split cells
	for j in (idx_adj - idx_old):
		loc = np.where(img_adj == j)
		j_old = img_old[loc][0]
		T_adj[j - 1, :] = T_adj[j_old - 1, :]
	return T_adj