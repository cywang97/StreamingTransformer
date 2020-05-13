#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the speech recognition task."""

import json
import logging
import os
import sys
import pdb

import numpy as np
import torch

from espnet.asr.asr_utils import add_results_to_json, add_single_results
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest


def _recursive_to(xs, device):
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple):
        return tuple(_recursive_to(x, device) for x in xs)
    return xs


def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    model.recog_args = args


    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError(
                "use '--api v2' option to decode with non-default language model"
            )
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None


    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            feat = load_inputs_and_targets(batch)
            feat = feat[0][0]
            pdb.set_trace()
            if args.prefix_decode:
                best, ids, score = model.prefix_recognize(feat, args, train_args, train_args.char_list, rnnlm)
                new_js[name] = add_single_results(js[name], best, ids, score)
            else:
                nbest_hyps = model.recognize(
                    feat, args, train_args.char_list, rnnlm
                )
                new_js[name] = add_results_to_json(
                    js[name], nbest_hyps, train_args.char_list
                )


    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )

def viterbi_decode(args):
    set_deterministic_pytorch(args)
    idim, odim, train_args = get_model_conf(
        args.model, os.path.join(os.path.dirname(args.model), 'model.json'))
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    if args.model is not None:
        load_params = dict(torch.load(args.model)['state_dict'])
        model_params = dict(model.named_parameters())
        for k, v in load_params.items():
            k = k.replace('module.', '')
            if k in model_params and v.size() == model_params[k].size():
                model_params[k].data = v.data
                logging.warning('load parameters {}'.format(k))
    model.recog_args = args

    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()

    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}


    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})


    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            feat = load_inputs_and_targets(batch)
            y = np.fromiter(map(int, batch[0][1]['output'][0]['tokenid'].split()), dtype=np.int64)
            align = model.viterbi_decode(feat[0][0], y)
            assert len(align) == len(y)
            new_js[name] = js[name]
            new_js[name]['output'][0]['align'] = ' '.join([str(i) for i in list(align)])

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
