import numpy as np
import pandas as pd
from absl import app
from absl import flags

import cluster

FLAGS = flags.FLAGS
flags.DEFINE_integer("k", default=5, help="k parameter for nearest neighbours")
flags.DEFINE_string("path", default=None, short_name="p", help="Input csv.")
flags.DEFINE_boolean("use_ti", default=False, short_name="ti",
                     help="Use Triangle Inequality (TI).")
flags.DEFINE_list("reference_point", default=[], short_name="rp",
                  help="Reference point for TI")
flags.DEFINE_string("output_path", default="clusters.csv", short_name="o", help="Output path with")
flags.mark_flag_as_required("path")


def run(_):
    df = pd.read_csv(FLAGS.path, header=None, delimiter=",")
    points = df.astype(np.float64)
    if FLAGS.use_ti:
        reference_point = (np.array(FLAGS.reference_point).astype(np.float64) if FLAGS.reference_point
                           else np.array(points.values.min(axis=0)))
    else:
        reference_point = None
    clusters = cluster.nbc(points.values, FLAGS.k, reference_point=reference_point)
    cluster.save_to_file(output_path=FLAGS.output_path, clusters = clusters)


def main():
    app.run(run)

if __name__ == "__main__":
    main()