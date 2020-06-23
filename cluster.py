from typing import Dict, Set, Union

import numpy as np

import neighbours
import ti_neighbours

CLUSTER_ID = int
VECTOR_ID = int
CLUSTERS = Dict[VECTOR_ID, CLUSTER_ID]
KNB = Dict[VECTOR_ID, Set]
NDF = Dict[VECTOR_ID, float]
R_KNB = Dict[VECTOR_ID, Set]


def save_to_file(output_path, clusters):
    noise_points_count = 0
    with open(output_path, "w") as csv:
        for _, value in sorted(clusters.items(), key=lambda kv: kv[0]):
            if value == -1:
                noise_points_count+=1
            csv.write(f"{value}\n")
    print("Number of noise point {}".format(noise_points_count))

def nbc(vectors: np.array, k: int, reference_point: Union[None, np.array] = None) -> CLUSTERS:
    clusters = {}
    for idx, _ in enumerate(vectors):
        clusters[idx] = -1

    if reference_point is not None:
        knb, r_knb = ti_neighbours.ti_k_neighbourhood(vectors, k, reference_point)
    else:
        knb, r_knb = neighbours.k_neighbourhood(vectors, k)

    ndf = neighbours.ndf(knb, r_knb)

    current_cluster_id = 0
    for idx, v in enumerate(vectors):
        if _has_cluster(idx, clusters) or not _is_dense_point(idx, ndf):
            continue
        clusters[idx] = current_cluster_id
        dense_points = set()

        for n_idx in knb[idx]:
            clusters[n_idx] = current_cluster_id
            if _is_dense_point(n_idx, ndf):
                dense_points.add(n_idx)

        while dense_points:
            dp = dense_points.pop()
            for n_idx in knb[dp]:
                if _has_cluster(n_idx, clusters):
                    continue
                clusters[n_idx] = current_cluster_id
                if _is_dense_point(n_idx, ndf):
                    dense_points.add(n_idx)

        current_cluster_id += 1

    return clusters


def _is_dense_point(idx: VECTOR_ID, ndf: NDF) -> bool:
    return ndf[idx] >= 1


def _has_cluster(idx: VECTOR_ID, clusters: CLUSTERS) -> bool:
    return clusters[idx] != -1