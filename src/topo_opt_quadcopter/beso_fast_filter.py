# Vectorized replacements for BESO's "simple" filter (beso_filters.prepare2s / run2), and
# further down for its "casting" filter.
#
# beso_runner appends this file to the private copy of beso_filters.py, so these definitions
# override the originals. The results are the same: each element's sensitivity becomes the
# average of its neighbours' within r_min, weighted by (r_min - distance), leaving out the
# element itself (as BESO does). The originals are pure-Python loops taking ~8 s per
# iteration at 36k elements; these take well under a second.

import numpy as _np
import scipy.sparse as _sp
from scipy.spatial import cKDTree as _cKDTree

_simple_filter_matrices = {}  # id(weight_factor2) → (element ids, sparse weights, row sums)


def prepare2s(cg, cg_min, cg_max, r_min, opt_domains, weight_factor2, near_elm):
    if id(weight_factor2) in _simple_filter_matrices:
        raise NotImplementedError("fast simple filter: only one 'simple' filter per run is supported")
    ids =_np.array(sorted(set(opt_domains)))
    points = _np.array([cg[en] for en in ids])
    pairs = _cKDTree(points).query_pairs(r_min, output_type="ndarray")
    d = _np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    keep = d < r_min  # query_pairs includes d == r_min; BESO does not
    pairs, w = pairs[keep], r_min - d[keep]
    # Keep BESO's data structures filled in, for other filters that share near_elm.
    a, b = ids[pairs[:, 0]].tolist(), ids[pairs[:, 1]].tolist()
    weight_factor2.update(zip(zip(a, b), w.tolist()))
    for en in ids.tolist():
        near_elm.setdefault(en, [])
    for en, en2 in zip(a, b):
        near_elm[en].append(en2)
        near_elm[en2].append(en)
    n = len(ids)
    W = _sp.coo_matrix((_np.concatenate([w, w]), (_np.concatenate([pairs[:, 0], pairs[:, 1]]),
                                                  _np.concatenate([pairs[:, 1], pairs[:, 0]]))),
                       shape=(n, n)).tocsr()
    _simple_filter_matrices[id(weight_factor2)] = (ids, W, _np.asarray(W.sum(axis=1)).ravel())
    return weight_factor2, near_elm


def run2(file_name, sensitivity_number, weight_factor2, near_elm, opt_domains):
    ids, W, row_sum = _simple_filter_matrices[id(weight_factor2)]
    if set(opt_domains) != set(ids.tolist()):
        raise ValueError("fast simple filter: domains differ from those given to prepare2s")
    if (row_sum == 0).any():
        msg = "\nERROR: simple filter failed due to division by 0." \
              "Some element has not a near element in distance <= r_min.\n"
        print(msg)
        beso_lib.write_to_log(file_name, msg)
        return sensitivity_number
    s = _np.array([sensitivity_number[en] for en in ids.tolist()])
    filtered = sensitivity_number.copy()
    filtered.update(zip(ids.tolist(), (W @ s / row_sum).tolist()))
    return filtered


# Vectorized replacements for BESO's "casting" filter (prepare2s_casting / run2_casting).
#
# BESO's filter: each element's sensitivity is first averaged with those of the elements
# "below" it, then replaced by the maximum over itself and the elements "above" it, so a solid
# element keeps everything below it solid (material fills columns from the low end of the
# casting vector). "Above" means within r_min of it measured across the casting direction
# and higher along it. The original sorts elements into square sectors of side r_min and
# compares pairs in Python (~65 s per iteration at 20k elements).
#
# _CASTING_FIX = False reproduces the original exactly, including a bug in the neighbouring
# sectors: the height test there is reversed (`if z_en <= z_en2: break` over the neighbour
# sorted from the top), so an element only gets neighbours from the adjacent sectors when it
# is higher than every element in that sector, and then gets the lower ones as "above".
# True uses the intended rule everywhere: above = within r_min across, strictly higher.
_CASTING_FIX = False
_casting_matrices = {}  # id(above_elm) → (element ids, above CSR incl. self, below CSR, below counts)


def _casting_frame(casting_vector):
    """BESO's transformation matrix to casting coordinates (rows ex, ey, casting vector)."""
    v = _np.asarray(casting_vector, dtype=float)
    v = v / _np.linalg.norm(v)
    ex = _np.array([-v[2], 0., v[0]])
    ex /= _np.linalg.norm(ex)
    return _np.array([ex, _np.cross(ex, v), v])


def prepare2s_casting(cg, r_min, opt_domains, above_elm, below_elm, casting_vector):
    if _casting_matrices:
        raise NotImplementedError("fast casting filter: only one 'casting' filter per run is supported")
    ids = _np.array(sorted(set(opt_domains)))
    c = _np.array([cg[en] for en in ids]) @ _casting_frame(casting_vector).T
    xy, z = c[:, :2], c[:, 2]
    pairs = _cKDTree(xy).query_pairs(r_min, output_type="ndarray")
    d = _np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    pairs = pairs[d <= r_min]
    # orient each pair as (lower, upper) in BESO's sort order: z, then element id
    a, b = pairs[:, 0], pairs[:, 1]
    b_up = (z[b] > z[a]) | ((z[b] == z[a]) & (ids[b] > ids[a]))
    lo, hi = _np.where(b_up, a, b), _np.where(b_up, b, a)
    if _CASTING_FIX:
        keep = z[hi] > z[lo]
        el, above = lo[keep], hi[keep]
    else:
        sector = _np.floor((xy - xy.min(axis=0)) / r_min).astype(_np.int64)
        same = (sector[lo] == sector[hi]).all(axis=1)
        # adjacent sectors: element e takes the lower element f of the pair as "above" when e
        # is higher than every element of f's sector
        key = sector[:, 0] * (sector[:, 1].max() + 1) + sector[:, 1]
        _, inv = _np.unique(key, return_inverse=True)
        top = _np.full(inv.max() + 1, -_np.inf)
        _np.maximum.at(top, inv, z)
        cross = ~same & (z[hi] > top[inv[lo]])
        el = _np.concatenate([lo[same], hi[cross]])
        above = _np.concatenate([hi[same], lo[cross]])
    n = len(ids)
    ones = _np.ones(len(el), dtype=bool)
    A = _sp.coo_matrix((ones, (el, above)), shape=(n, n)).tocsr()  # row e: elements above e
    A = (A + _sp.identity(n, dtype=bool, format="csr")).tocsr()
    A.sort_indices()
    B = _sp.coo_matrix((_np.ones(len(el)), (above, el)), shape=(n, n)).tocsr()  # row f: elements below f
    _casting_matrices[id(above_elm)] = (ids, A, B, _np.asarray(B.sum(axis=1)).ravel())
    return above_elm, below_elm


def run2_casting(sensitivity_number, above_elm, below_elm, opt_domains):
    ids, A, B, n_below = _casting_matrices[id(above_elm)]
    if set(opt_domains) != set(ids.tolist()):
        raise ValueError("fast casting filter: domains differ from those given to prepare2s_casting")
    s = _np.array([sensitivity_number[en] for en in ids.tolist()])
    averaged = (s + B @ s) / (1 + n_below)
    filtered = _np.maximum.reduceat(averaged[A.indices], A.indptr[:-1])
    out = sensitivity_number.copy()
    out.update(zip(ids.tolist(), filtered.tolist()))
    return out
