#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np


def main():
    if args.log is not None:
        with open(args.log) as f:
            logs = json.load(f)
        val_scores = []
        for log in logs:
            if "validation/main/acc" in log.keys():
                val_scores += [[log["epoch"], log["validation/main/acc"]]]
            elif "val_perplexity" in log.keys():
                val_scores += [[log["epoch"], 1 / log["val_perplexity"]]]
            elif "validation/main/loss" in log.keys():
                val_scores += [[log["epoch"], -log["validation/main/loss"]]]
        if len(val_scores) == 0:
            raise ValueError(
                "`validation/main/acc` or `val_perplexity` is not found in log."
            )
        val_scores = np.array(val_scores)
        sort_idx = np.argsort(val_scores[:, -1])
        sorted_val_scores = val_scores[sort_idx][::-1]
        print("best val scores = " + str(sorted_val_scores[: args.num, 1]))
        print(
            "selected epochs = "
            + str(sorted_val_scores[: args.num, 0].astype(np.int64))
        )
        last = [
            os.path.dirname(args.snapshots[0]) + "/snapshot.ep.%d" % (int(epoch))
            for epoch in sorted_val_scores[: args.num, 0]
        ]
    else:
        last = sorted(args.snapshots, key=os.path.getmtime)
        last = last[-args.num :]
    print("average over", last)
    avg = None

    import torch

    # sum
    for path in last:
        states = torch.load(path, map_location=torch.device("cpu"))
        if 'state_dict' in states:
            states = states['state_dict']
        if 'model' in states:
            states = states['model']
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]

    # average
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] /= args.num

    torch.save(avg, args.out)


def get_parser():
    parser = argparse.ArgumentParser(description="average models from snapshot")
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--backend", default="chainer", type=str)
    parser.add_argument("--log", default=None, type=str, nargs="?")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
