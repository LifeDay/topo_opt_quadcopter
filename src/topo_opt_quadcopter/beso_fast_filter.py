# Vectorized replacements for BESO's "simple" filter (beso_filters.prepare2s / run2).
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
