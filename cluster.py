import time

import neighbours
import ti_neighbours


def save_to_file(output_path, clusters):
    noise_points_count = 0
    with open(output_path, "w") as csv:
        for _, value in sorted(clusters.items(), key=lambda kv: kv[0]):
            if value == -1:
                noise_points_count+=1
            csv.write(f"{value}\n")
    print("Number of noise point {}".format(noise_points_count))

def nbc(vectors, k, reference_point) :
    clusters = {}
    for idx, _ in enumerate(vectors):
        clusters[idx] = -1
    if reference_point is not None:
        group_start_time = time.time()
        knb, r_knb = ti_neighbours.ti_k_neighbourhood(vectors, k, reference_point)
        grouping_time = (time.time() - group_start_time)
    else:
        group_start_time = time.time()
        knb, r_knb = neighbours.k_neighbourhood(vectors, k)
        grouping_time = (time.time() - group_start_time)
    print("Grouping time --- %s seconds ---" % (grouping_time))

    ndf = neighbours.ndf(knb, r_knb)

    current_cluster_id = 0
    for idx, v in enumerate(vectors):
        if _has_cluster(idx, clusters) or not _check_is_dp(idx, ndf):
            continue
        clusters[idx] = current_cluster_id
        dense_points = set()

        for n_idx in knb[idx]:
            clusters[n_idx] = current_cluster_id
            if _check_is_dp(n_idx, ndf):
                dense_points.add(n_idx)

        while dense_points:
            dp = dense_points.pop()
            for n_idx in knb[dp]:
                if _has_cluster(n_idx, clusters):
                    continue
                clusters[n_idx] = current_cluster_id
                if _check_is_dp(n_idx, ndf):
                    dense_points.add(n_idx)

        current_cluster_id += 1

    return clusters

# check if point is dense point
def _check_is_dp(idx, ndf):
    return ndf[idx] >= 1


def _has_cluster(idx, clusters):
    return clusters[idx] != -1