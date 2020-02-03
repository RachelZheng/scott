import ot
import numpy as np
import scipy as sp

from utils.mapping_tools import *
from utils.mapping_1st import get_sinkhorn_eps

def get_2nd_mapping(pts_s, pts_t, options):
	""" Get the second-order transform matrix. 
	Input: source/target samples, options
	Output: alignment matrix G, the subset of samples from source/target
	"""
	if hasattr(options, 'weight_M'):
		weight_M = options.weight_M
	else:
		weight_M = 0.5

	if hasattr(options, 'n_sampling_pts_2nd'):
		n_pts = options.n_sampling_pts_2nd
	else:
		n_pts = 5000

	if len(pts_s) > n_pts:
		idx_s     = np.random.randint(len(pts_s), size=n_pts)
		pts_s_sub = pts_s[idx_s,:]
	else:
		idx_s 	  = range(len(pts_s)) 
		pts_s_sub = pts_s

	if len(pts_t) > n_pts:
		idx_t     = np.random.randint(len(pts_t), size=n_pts)
		pts_t_sub = pts_t[idx_t,:]
	else:
		idx_t 	  = range(len(pts_t)) 
		pts_t_sub = pts_t

	# Compute first-order information
	M = ot.dist(pts_s_sub, pts_t_sub, metric='euclidean')
	M /= max(M.max(), 1e-6)

	# Compute second-order information
	C_s = sp.spatial.distance.cdist(pts_s_sub, pts_s_sub)
	C_t = sp.spatial.distance.cdist(pts_t_sub, pts_t_sub)
	C_s /= max(C_s.max(), 1e-6)
	C_t /= max(C_t.max(), 1e-6)

	# Point-wise probability
	p_s = np.ones((len(pts_s_sub),))/len(pts_s_sub)
	p_t = np.ones((len(pts_t_sub),))/len(pts_t_sub)
	gw, eps = get_gw_eps(C_s, C_t, p_s, p_t, weight_M=weight_M, M=M)
	
	# Just second-order information
	return gw, pts_s_sub, pts_t_sub, idx_s, idx_t, eps


def get_gw_eps(C1, C2, a, b, eps_0=5e-3, weight_M=0, M=0):
	""" Get the suitable epsilon value in the ot.gromov_wasserstein function.
	"""
	eps = eps_0
	if weight_M == 0:
		# Purely L1 normalization
		gw, eps = get_sinkhorn_eps(a, b, M, bool_fix_eps=True)
	else:
		gw = mix_gromov_wasserstein(C1, C2, a, b, epsilon=eps, weight_M=weight_M, M=M)
	return gw, eps



def mix_gromov_wasserstein(C1, C2, p, q, epsilon, weight_M, M,
    max_iter=1000, tol=1e-8, verbose=False, log=False, init_method='avg',
    **kwargs):
    """ Get the mixture order wasserstein distance.
    """
    C1, C2 = np.asarray(C1, dtype=np.float64), np.asarray(C2, dtype=np.float64)

    # initialize matrix T with different initialization
    if init_method == 'avg':
       T = np.outer(p, q)  # average initialization
    elif init_method == 'random':
        T = get_random_T(p, q)
    elif init_method == 'sequence':
        T = get_sequence_aligned_T(p, q)
    elif init_method == 'wasserstein':
        T = ot.emd(p, q, M)

    cpt = 0
    err = 1
    while (err > tol and cpt < max_iter):
        Tprev = T
        tens = weight_M * tensor_square_loss(C1, C2, T) + (1 - weight_M) * M
        T = ot.sinkhorn(p, q, tens, epsilon)
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(T - Tprev)
            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt += 1
    if log:
        return T, log
    else:
        return T


def tensor_square_loss(C1, C2, T):
    tens = -np.dot(C1, T).dot(C2.T)
    tens -= tens.min()
    return tens


def get_random_T(p, q):
    n1, n2 = len(p), len(q)
    T = np.zeros((n1, n2), dtype=float)
    T[:n1-1, :n2-1] = np.random.uniform(
        low=0, high=max(np.max(p), np.max(q)), size=(n1-1, n2-1))

    for i in range(n1 - 1):
        s = np.sum(T[i])
        if s > p[i]:
            ratio = np.random.uniform(low=0, high=p[i]/s)
            T[i] = T[i] * ratio

    for j in range(n2 - 1):
        s = np.sum(T[:,j])
        if s > q[j]:
            ratio = np.random.uniform(low=0, high=q[j]/s)
            T[:,j] = T[:,j] * ratio

    T[-1] = q - np.sum(T, axis=0)
    T[:,-1] = p - np.sum(T, axis=1)
    return T


def get_sequence_aligned_T(p, q):
    """Sequentially align the transition weights according to the pts sequence.
    Boundary condition:
        T * 1 = p; T' * 1 = q; |p|_1 == |q|_1 == 1
    """
    n1, n2 = len(p), len(q)
    T = np.zeros((n1, n2), dtype=float)

    i = 0
    j = 0
    p_to_assign = p[i]
    q_to_assign = q[j]

    while(i < n1 and j < n2):
        amt = min(p_to_assign, q_to_assign)
        T[i,j] = amt
        p_to_assign -= amt
        q_to_assign -= amt
        while p_to_assign == 0:
            i += 1
            if i < n1:
                p_to_assign = p[i]
            else:
                break

        while q_to_assign == 0:
            j += 1
            if j < n2:
                q_to_assign = q[j]
            else:
                break
    return T

