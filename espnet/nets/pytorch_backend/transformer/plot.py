#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import matplotlib.pyplot as plt
import numpy

from espnet.asr import asr_utils


def _plot_and_save_attention(att_w, filename, xtokens=None, ytokens=None):
    # dynamically import matplotlib due to not found error
    from matplotlib.ticker import MaxNLocator
    import os

    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)
    w, h = plt.figaspect(1.0 / len(att_w))
    fig = plt.Figure(figsize=(w * 2, h * 2))
    axes = fig.subplots(1, len(att_w))
    if len(att_w) == 1:
        axes = [axes]
    for ax, aw in zip(axes, att_w):
        # plt.subplot(1, len(att_w), h)
        ax.imshow(aw.astype(numpy.float32), aspect="auto")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Labels for major ticks
        if xtokens is not None:
            ax.set_xticks(numpy.linspace(0, len(xtokens) - 1, len(xtokens)))
            ax.set_xticks(numpy.linspace(0, len(xtokens) - 1, 1), minor=True)
            ax.set_xticklabels(xtokens + [""], rotation=40)
        if ytokens is not None:
            ax.set_yticks(numpy.linspace(0, len(ytokens) - 1, len(ytokens)))
            ax.set_yticks(numpy.linspace(0, len(ytokens) - 1, 1), minor=True)
            ax.set_yticklabels(ytokens + [""])
    fig.tight_layout()
    return fig


def savefig(plot, filename):
    plot.savefig(filename)
    plt.clf()


