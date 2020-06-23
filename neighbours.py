from math import sqrt
from typing import Set, Dict, Tuple

import numpy as np

VECTOR_ID = int
KNB = Dict[VECTOR_ID, Set]
NDF = Dict[VECTOR_ID, float]
R_KNB = Dict[VECTOR_ID, Set]


def distance(v1, v2) -> float:
    n = v1.shape[0]
    dist = 0.0
    for i in range(n):
        dist += (v1[i] - v2[i])**2.0
    return sqrt(dist)


def k_neighbourhood(vectors: np.ndarray, k: int) -> Tuple[KNB, R_KNB]:
    knb, r_knb = _init(vectors)

    for idx1, v1 in enumerate(vectors):
        neighbour_candidates = []
        for idx2, v2 in enumerate(vectors):
            if idx1 != idx2:
                dist = distance(v1, v2)
                neighbour_candidates.append((idx2, dist))
        neighbour_candidates.sort(key=lambda t: t[1])
        eps = neighbour_candidates[:k][-1][1]

        neighbours = set()
        for (i, d) in neighbour_candidates:
            if d > eps:
                break
            neighbours.add(i)
        _fill(knb, r_knb, idx1, neighbours)
    return knb, r_knb


def ndf(knb: KNB, r_knb: R_KNB) -> NDF:
    ndfs = {}
    for k in knb.keys():
        ndfs[k] = len(r_knb[k]) / len(knb[k])
    return ndfs

def _init(vectors):
    knb = {}
    r_knb = {}
    for i in range(vectors.shape[0]):
        r_knb[i] = set()
    return knb, r_knb

def _fill(knb, r_knb, vector_idx, neighbours):
    knb[vector_idx] = neighbours
    for n in neighbours:
        r_knb[n].add(vector_idx)
