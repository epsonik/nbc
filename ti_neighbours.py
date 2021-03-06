import numpy as np

from point import Point

from math import sqrt
from typing import Set, Dict, Tuple, List

from sortedcontainers import SortedSet
import time
VECTOR_ID = int
KNB = Dict[VECTOR_ID, Set]
NDF = Dict[VECTOR_ID, float]
R_KNB = Dict[VECTOR_ID, Set]

def ti_k_neighbourhood(vectors: np.ndarray, k: int, reference_point: np.ndarray) -> Tuple[KNB, R_KNB]:
    knb, r_knb = _init(vectors)

    points = _ti(vectors, reference_point)
    for point in points:
        idx = point.idx
        neighbours = _ti_neighbours(point, k)
        _fill(knb, r_knb, idx, neighbours)

    return knb, r_knb

def _ti_neighbours(point: Point, k):
    bp = point
    fp = point
    backward_search = bp.preceding
    bp = bp.preceding
    forward_search = fp.following
    fp = fp.following

    neighbour_candidates = SortedSet(key=lambda x: x[1])

    bp, fp, backward_search, forward_search = _candidate_nbs(backward_search,
                                                             forward_search,
                                                             neighbour_candidates,
                                                             k,
                                                             point,
                                                             bp,
                                                             fp)
    eps = neighbour_candidates[-1][1]
    eps = _verify_backward(point, bp, backward_search, neighbour_candidates, k, eps)
    eps = _verify_forward(point, fp, forward_search, neighbour_candidates, k, eps)
    return [n[0].idx for n in neighbour_candidates]


def _candidate_nbs(backward_search: bool,
                   forward_search: bool,
                   neighbour_candidates: SortedSet,
                   k: int,
                   p: Point,
                   bp: Point,
                   fp: Point):
    i = 0
    while forward_search and backward_search and i < k:
        if p.dist - bp.dist < fp.dist - p.dist:
            dist = distance(bp.vector, p.vector)
            neighbour_candidates.add((bp, dist))
            backward_search = bp.preceding
            bp = bp.preceding
        else:
            dist = distance(fp.vector, p.vector)
            neighbour_candidates.add((fp, dist))
            forward_search = fp.following
            fp = fp.following
        i += 1
    while forward_search and i < k:
        dist = distance(fp.vector, p.vector)
        i += 1
        neighbour_candidates.add((fp, dist))
        forward_search = fp.following
        fp = fp.following
    while backward_search and i < k:
        dist = distance(bp.vector, p.vector)
        i += 1
        neighbour_candidates.add((bp, dist))
        backward_search = bp.preceding
        bp = bp.preceding
    return bp, fp, backward_search, forward_search


def _ti(vectors: np.ndarray,
        reference_point: np.ndarray):
    rp_dist = []
    for idx, v in enumerate(vectors):
        dist = distance(v, reference_point)
        rp_dist.append(dist)
    sort_start_time = time.time()
    arg_sorted_rp_dist = np.argsort(rp_dist)
    print("Sorting time TI--- %s seconds ---" % (time.time() - sort_start_time))
    points = []
    for i, vector_id in enumerate(arg_sorted_rp_dist):
        if i == 0:
            points.append(Point(vector_id, vectors[vector_id], rp_dist[vector_id]))
        else:
            point = Point(vector_id, vectors[vector_id], rp_dist[vector_id], preceding=points[i - 1])
            points.append(point)
            points[i - 1].following = point
    return points


def _fill(knb, r_knb, vector_idx, neighbours):
    knb[vector_idx] = neighbours
    for n in neighbours:
        r_knb[n].add(vector_idx)


def distance(v1, v2) -> float:
    n = v1.shape[0]
    dist = 0.0
    for i in range(n):
        dist += (v1[i] - v2[i])**2.0
    result = sqrt(dist)
    return result

def _init(vectors):
    knb = {}
    r_knb = {}
    for i in range(vectors.shape[0]):
        r_knb[i] = set()
    return knb, r_knb

def _verify_forward(p: Point, fp: Point, forward_search: bool, neighbour_candidates: SortedSet, k: int, eps: float):
    while forward_search and (p.dist - fp.dist) <= eps:
        dist = distance(fp.vector, p.vector)
        if dist < eps:
            i = len([e for e in neighbour_candidates if e[1] == eps])
            if len(neighbour_candidates) - i >= k - 1:
                for e in neighbour_candidates:
                    if e[1] == eps:
                        neighbour_candidates.remove(e)
                neighbour_candidates.add((fp, dist))
                eps = neighbour_candidates[-1][1]
            else:
                neighbour_candidates.add((fp, dist))
        elif dist == eps:
            neighbour_candidates.add((fp, dist))
        forward_search = fp.following
        fp = fp.following
    return eps


def _verify_backward(p: Point, bp: Point, backward_search: bool, neighbour_candidates: SortedSet, k: int, eps: float):
    while backward_search and (p.dist - bp.dist) <= eps:
        dist = distance(bp.vector, p.vector)
        if dist < eps:
            i = len([e for e in neighbour_candidates if e[1] == eps])
            if len(neighbour_candidates) - i >= k - 1:
                for e in neighbour_candidates:
                    if e[1] == eps:
                        neighbour_candidates.remove(e)
                neighbour_candidates.add((bp, dist))
                eps = neighbour_candidates[-1][1]
            else:
                neighbour_candidates.add((bp, dist))
        elif dist == eps:
            neighbour_candidates.add((bp, dist))
        backward_search = bp.preceding
        bp = bp.preceding
    return eps
