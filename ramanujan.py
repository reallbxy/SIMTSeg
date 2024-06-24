# %%
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from scipy.ndimage import gaussian_filter
import cupy as cp
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

# small_basis = {
#     'p2': np.array([1,-1]),
#     'p3': np.array([2, -1, -1]),
#     'p4': np.array([2, 0, -2, 0]),
#     'p5': np.array([4, -1, -1, -1, -1]),
#     'p6': np.array([2, 1, -1, -2, -1, 1]),
#     'p7': np.array([6, -1, -1, -1, -1, -1, -1]),
#     'p8': np.array([4, 0, 0, 0, -4, 0, 0, 0]),
#     'p9': np.array([6, 0, 0, -3, 0, 0, -3, 0, 0]),
#     'p10': np.array([4, 1, -1, 1, -1, -4, -1, 1, -1, 1])
# }

def basis_cal(p):
    basis = cp.zeros(p, dtype=complex)
    k_list = []
    for i in range(p):
        k = i + 1
        if cp.gcd(k, p) == 1:
            k_list.append(k)
    for n in range(p):
        basis_k = 0
        for kk in k_list:
            basis_k += np.exp(1j * 2 * np.pi * kk * n / p)
        basis[n] = basis_k
    return cp.real(basis)


def Pp_cal(p, M):
    # if p <= 10:
    #     basis = small_basis['p%d' %p]
    # else:
    #     basis = basis_cal(p)
    basis = basis_cal(p)
    Bp = cp.zeros((p,p))
    for i in range(p):
        Bp[i] = cp.roll(basis, i)
    P = Bp / p
    Pp = cp.tile(P, (M, M))
    return Pp


def proj(x, p):
    if len(x) % p != 0:
        length = ((len(x) // p) + 1) * p
        M = (len(x) // p) + 1
        padding = x[:length - len(x)]
        x = cp.hstack((x, padding))
    else:
        x = x
        M = len(x) // p
    Pp = Pp_cal(p, M)
    proj_vec = cp.dot(Pp,x)
    return x, proj_vec


def find_p(x, p_min=200):
    # dominant periodicity searching
    p_max = int(len(x) // 3)
    # p_max = int(len(x) // 20)
    dist = []
    # p_max = 500
    for p in range(p_min, p_max + 1, 50):
        x_padding, proj_v = proj(x, p)
        a = x_padding.reshape((1, -1))
        b = proj_v.reshape((1, -1))
        d = cdist(a.get(), b.get(), 'cosine')
        dist.append(d)
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    period = (dist.index(min(dist)) + 2) * 50
    return period


def time_fold(x, p):
    if len(x[0]) % p != 0:
        length = ((len(x[0]) // p) + 1) * p
        padding = cp.zeros((len(x), length - len(x[0])))
        x = cp.hstack((x, padding))
    else:
        x = x
    fold = cp.zeros((len(x), p, len(x[0])//p))
    for i in range(len(x)):
        for j in range(len(x[0])//p):
            fold[i][:, j : j + 1] = x[i][p * j : p * (j+1)].reshape((-1, 1))
    return fold


def dataset_process(data):
    #  0-1 normalization
    content = np.zeros(data.shape)
    for i in range(len(data[0])):
        mm = MinMaxScaler()
        content[:, i:i+1] = mm.fit_transform(data[:, i:i+1])
    # series for ramanujan projection
    content = cp.array(content)
    x_sum = cp.sum(content, axis=1)
    if len(x_sum) <= 800:
        p_min = int(len(x_sum)/5)
    else:
        p_min = 300
    period = find_p(x_sum, p_min=p_min)
    content_trans = content.T
    fold = time_fold(content_trans, period)
    fold_x = fold.transpose((1, 2, 0))
    return content, period, fold_x


def euc_dist(pt1, pt2):
    return cp.sqrt(cp.square(pt2[0] - pt1[0]) + cp.square(pt2[1] - pt1[1]))


def Frechet_distance(exp_data,num_data):
    P=exp_data
    Q=num_data
    p_length = len(P)
    q_length = len(Q)
    distance_matrix = cp.ones((p_length, q_length)) * -1

    # fill the first value with the distance between
    # the first two points in P and Q
    # distance_matrix[0, 0] = euclidean(P[0], Q[0])
    distance_matrix[0, 0] = euc_dist(P[0], Q[0])

    # load the first column and first row with distances (memorize)
    for i in range(1, p_length):
        distance_matrix[i, 0] = max(distance_matrix[i - 1, 0], euc_dist(P[i], Q[0]))
    for j in range(1, q_length):
        distance_matrix[0, j] = max(distance_matrix[0, j - 1], euc_dist(P[0], Q[j]))

    for i in range(1, p_length):
        for j in range(1, q_length):
            distance_matrix[i, j] = max(
                min(distance_matrix[i - 1, j], distance_matrix[i, j - 1], distance_matrix[i - 1, j - 1]),
                euc_dist(P[i], Q[j]))
    # distance_matrix[p_length - 1, q_length - 1]
    return distance_matrix[p_length-1,q_length-1]


def mask_process(hidden, height, width, period, T, smooth):
    # unfold hidden variable
    h = tf.reduce_mean(hidden, axis=3, keepdims=True)
    h = np.array(h)[0]
    m = h.reshape(height, width, 1)
    m_t = m.transpose((2, 0, 1))
    value = np.zeros(m_t.shape[1] * m_t.shape[2])
    for i in range(m_t.shape[2]):
        value[period * i:period * (i + 1)] = m_t[0][:, i:i + 1].reshape((period,))
    value = value[:T].reshape(-1, 1)
    value = gaussian_filter(value, smooth)
    mm = MinMaxScaler()
    value = mm.fit_transform(value)
    m_value = np.array(value)
    return m_value

