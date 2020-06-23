from math import sqrt
import time



def distance(v1, v2) :
    n = v1.shape[0]
    dist = 0.0
    for i in range(n):
        dist += (v1[i] - v2[i])**2.0
    result = sqrt(dist)
    print(result)
    return result


def k_neighbourhood(vectors, k):
    knb, r_knb = _init(vectors)
    sort_time = 0
    for idx1, v1 in enumerate(vectors):
        neighbour_candidates = []
        for idx2, v2 in enumerate(vectors):
            if idx1 != idx2:
                dist = distance(v1, v2)
                neighbour_candidates.append((idx2, dist))
        sort_start_time = time.time()
        neighbour_candidates.sort(key=lambda t: t[1])
        sort_time += (time.time() - sort_start_time)

        eps = neighbour_candidates[:k][-1][1]

        neighbours = set()
        for (i, d) in neighbour_candidates:
            if d > eps:
                break
            neighbours.add(i)
        _fill(knb, r_knb, idx1, neighbours)
    print("Sorting time --- %s seconds ---" % (sort_time))
    return knb, r_knb


def ndf(knb, r_knb):
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
