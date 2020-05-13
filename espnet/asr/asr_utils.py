#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import copy
import json
import logging

# matplotlib related
import os
import shutil
import tempfile

# chainer related
import chainer

from chainer.training import extension

from chainer.serializers.npz import DictionarySerializer
from chainer.serializers.npz import NpzDeserializer

# io related
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")


# * -------------------- training iterator related -------------------- *


def torch_snapshot(savefun=torch.save, filename="snapshot.ep.{.updater.epoch}"):
    """Extension to take snapshot of the trainer for pytorch.

    Returns:
        An extension function.

    """

    @extension.make_extension(trigger=(1, "epoch"), priority=-100)
    def torch_snapshot(trainer):
        _torch_snapshot_object(trainer, trainer, filename.format(trainer), savefun)

    return torch_snapshot


def _torch_snapshot_object(trainer, target, filename, savefun):
    # make snapshot_dict dictionary
    s = DictionarySerializer()
    s.save(trainer)
    if hasattr(trainer.updater.model, "model"):
        # (for TTS)
        if hasattr(trainer.updater.model.model, "module"):
            model_state_dict = trainer.updater.model.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.model.state_dict()
    else:
        # (for ASR)
        if hasattr(trainer.updater.model, "module"):
            model_state_dict = trainer.updater.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.state_dict()
    snapshot_dict = {
        "trainer": s.target,
        "model": model_state_dict,
        "optimizer": trainer.updater.get_optimizer("main").state_dict(),
    }

    # save snapshot dictionary
    fn = filename.format(trainer)
    prefix = "tmp" + fn
    tmpdir = tempfile.mkdtemp(prefix=prefix, dir=trainer.out)
    tmppath = os.path.join(tmpdir, fn)
    try:
        savefun(snapshot_dict, tmppath)
        shutil.move(tmppath, os.path.join(trainer.out, fn))
    finally:
        shutil.rmtree(tmpdir)

# * -------------------- general -------------------- *
def get_model_conf(model_path, conf_path=None):
    """Get model config information by reading a model config file (model.json).

    Args:
        model_path (str): Model path.
        conf_path (str): Optional model config path.

    Returns:
        list[int, int, dict[str, Any]]: Config information loaded from json file.

    """
    if conf_path is None:
        model_conf = os.path.dirname(model_path) + "/model.json"
    else:
        model_conf = conf_path
    with open(model_conf, "rb") as f:
        logging.info("reading a config file from " + model_conf)
        confs = json.load(f)
    if isinstance(confs, dict):
        # for lm
        args = confs
        return argparse.Namespace(**args)
    else:
        # for asr, tts, mt
        idim, odim, args = confs
        return idim, odim, argparse.Namespace(**args)


def torch_save(path, model):
    """Save torch model states.

    Args:
        path (str): Model path to be saved.
        model (torch.nn.Module): Torch model.

    """
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def snapshot_object(target, filename):
    """Returns a trainer extension to take snapshots of a given object.

    Args:
        target (model): Object to serialize.
        filename (str): Name of the file into which the object is serialized.It can
            be a format string, where the trainer object is passed to
            the :meth: `str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.

    Returns:
        An extension function.

    """

    @extension.make_extension(trigger=(1, "epoch"), priority=-100)
    def snapshot_object(trainer):
        torch_save(os.path.join(trainer.out, filename.format(trainer)), target)

    return snapshot_object


def torch_load(path, model):
    """Load torch model states.

    Args:
        path (str): Model path or snapshot file path to be loaded.
        model (torch.nn.Module): Torch model.

    """
    if "snapshot" in os.path.basename(path):
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)[
            "model"
        ]
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)

    if hasattr(model, "module"):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict


def torch_resume(snapshot_path, trainer):
    """Resume from snapshot for pytorch.

    Args:
        snapshot_path (str): Snapshot file path.
        trainer (chainer.training.Trainer): Chainer's trainer instance.

    """
    # load snapshot
    snapshot_dict = torch.load(snapshot_path, map_location=lambda storage, loc: storage)

    # restore trainer states
    d = NpzDeserializer(snapshot_dict["trainer"])
    d.load(trainer)

    # restore model states
    if hasattr(trainer.updater.model, "model"):
        # (for TTS model)
        if hasattr(trainer.updater.model.model, "module"):
            trainer.updater.model.model.module.load_state_dict(snapshot_dict["model"])
        else:
            trainer.updater.model.model.load_state_dict(snapshot_dict["model"])
    else:
        # (for ASR model)
        if hasattr(trainer.updater.model, "module"):
            trainer.updater.model.module.load_state_dict(snapshot_dict["model"])
        else:
            trainer.updater.model.load_state_dict(snapshot_dict["model"])

    # retore optimizer states
    trainer.updater.get_optimizer("main").load_state_dict(snapshot_dict["optimizer"])

    # delete opened snapshot
    del snapshot_dict


# * ------------------ recognition related ------------------ *
def parse_hypothesis(hyp, char_list):
    """Parse hypothesis.

    Args:
        hyp (list[dict[str, Any]]): Recognition hypothesis.
        char_list (list[str]): List of characters.

    Returns:
        tuple(str, str, str, float)

    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp["yseq"][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp["score"])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace("<space>", " ")

    return text, token, tokenid, score

def add_single_results(js, best, best_token, score):
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    text = best.replace('<eos>', '')
    rec_text = text
    rec_tokenid = best_token[1:]
    out_dic = dict(js['output'][0].items())

    # update name
    out_dic['name'] += '[0]'

        # add recognition results
    out_dic['rec_text'] = rec_text
    out_dic['rec_token'] = rec_text
    out_dic['rec_tokenid'] = ' '.join(map(str, rec_tokenid))
    out_dic['score'] = float(score)

    # add to list of N-best result dicts
    new_js['output'].append(out_dic)
    if 'text' in out_dic.keys():
        logging.info('groundtruth: %s' % out_dic['text'])
    logging.info('prediction : %s' % out_dic['rec_text'])
    return new_js


def add_results_to_json(js, nbest_hyps, char_list):
    """Add N-best results to json.

    Args:
        js (dict[str, Any]): Groundtruth utterance dict.
        nbest_hyps_sd (list[dict[str, Any]]):
            List of hypothesis for multi_speakers: nutts x nspkrs.
        char_list (list[str]): List of characters.

    Returns:
        dict[str, Any]: N-best results added utterance dict.

    """
    # copy old json info
    new_js = dict()
    new_js["utt2spk"] = js["utt2spk"]
    new_js["output"] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

        # copy ground-truth
        if len(js["output"]) > 0:
            out_dic = dict(js["output"][0].items())
        else:
            # for no reference case (e.g., speech translation)
            out_dic = {"name": ""}

        # update name
        out_dic["name"] += "[%d]" % n

        # add recognition results
        out_dic["rec_text"] = rec_text
        out_dic["rec_token"] = rec_token
        out_dic["rec_tokenid"] = rec_tokenid
        out_dic["score"] = score

        # add to list of N-best result dicts
        new_js["output"].append(out_dic)

        # show 1-best result
        if n == 1:
            if "text" in out_dic.keys():
                logging.info("groundtruth: %s" % out_dic["text"])
            logging.info("prediction : %s" % out_dic["rec_text"])

    return new_js

