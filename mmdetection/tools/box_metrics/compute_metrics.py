import argparse
import cProfile as profile
import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

from metrics import get_dice_1, get_fast_aji, get_fast_aji_plus, get_fast_pq, remap_label


def run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=False, ext=".npy"):
    # print stats of each image
    print(pred_dir)

    file_list = glob.glob("%s/*%s" % (pred_dir, ext))
    file_list.sort()  # ensure same order

    metrics = [[], [], [], [], [], []]
    for filename in file_list[:]:
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        #true = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        #true = (true["inst_map"]).astype("int32")
        true = np.load(os.path.join(true_dir, basename + ".npy")).astype("int32")

        #pred = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
        #pred = (pred["inst_map"]).astype("int32")
        pred = np.load(os.path.join(pred_dir, basename + ".npy")).astype("int32")

        # to ensure that the instance numbering is contiguous 确保实例（每个mask）之间的编码是连续的
        pred = remap_label(pred, by_size=False)
        true = remap_label(true, by_size=False)

        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(get_fast_aji(true, pred))
        metrics[2].append(pq_info[0])  # dq
        metrics[3].append(pq_info[1])  # sq
        metrics[4].append(pq_info[2])  # pq
        metrics[5].append(get_fast_aji_plus(true, pred))

        if print_img_stats:
            print(basename, end="\t")
            for scores in metrics:
                print("%f " % scores[-1], end="  ")
            print()
    ####
    metrics = np.array(metrics)
    metrics_avg = np.mean(metrics, axis=-1)
    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print(metrics_avg)
    metrics_avg = list(metrics_avg)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="mode to run the measurement,"
        "`type` for nuclei instance type classification or"
        "`instance` for nuclei instance segmentation",
        nargs="?",
        default="instance",
        const="instance",
    )
    parser.add_argument(
        "--pred_dir", help="point to output dir", nargs="?", default="", const=""
    )
    parser.add_argument(
        "--true_dir", help="point to ground truth dir", nargs="?", default="", const=""
    )
    args = parser.parse_args()
    
    if args.mode == "instance":
        run_nuclei_inst_stat(args.pred_dir, args.true_dir, print_img_stats=False)