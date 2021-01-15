#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from collections import defaultdict, Counter
from distutils.util import strtobool
import time

import logging
import math
import numpy as np

import torch

from espnet.nets.viterbi_align import viterbi_align
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import adaptive_enc_mask, turncated_mask, trigger_mask
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling, EncoderConv2d
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.scorers.ctc import CTCPrefixScorer

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5

class E2E(torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed", "custom"],
                           help='transformer input layer type')
        group.add_argument("--transformer-output-layer", type=str, default='embed',
                           choices=['conv', 'embed', 'linear'])
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')

        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        # Decoder
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')

        # Streaming params
        group.add_argument('--chunk', default=True, type=strtobool,
                           help='streaming mode, set True for chunk-encoder, False for look-ahead encoder')
        group.add_argument('--chunk-size', default=16, type=int,
                           help='chunk size for chunk-based encoder')
        group.add_argument('--left-window', default=1000, type=int,
                           help='left window size for look-ahead based encoder')
        group.add_argument('--right-window', default=1000, type=int,
                           help='right window size for look-ahead based encoder')
        group.add_argument('--dec-left-window', default=0, type=int,
                           help='left window size for decoder (look-ahead based method)')
        group.add_argument('--dec-right-window', default=6, type=int,
                           help='right window size for decoder (look-ahead based method)')
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        self.decoder = Decoder(
            odim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            input_layer=args.transformer_output_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = [1]

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
        # self.verbose = args.verbose
        self.reset_parameters(args)
        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
        else:
            self.ctc = None

        self.rnnlm = None
        self.left_window = args.dec_left_window
        self.right_window = args.dec_right_window

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, ys_pad, enc_mask=None, dec_mask=None):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        batch_size = xs_pad.shape[0]
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        if isinstance(self.encoder.embed, EncoderConv2d):
            xs, hs_mask = self.encoder.embed(xs_pad, torch.sum(src_mask, 2).squeeze())
            hs_mask = hs_mask.unsqueeze(1)
        else:
            xs, hs_mask = self.encoder.embed(xs_pad, src_mask)

        if enc_mask is not None:
            enc_mask = enc_mask[:, :hs_mask.shape[2], :hs_mask.shape[2]]
        enc_mask = enc_mask & hs_mask if enc_mask is not None else hs_mask
        hs_pad, _ = self.encoder.encoders(xs, enc_mask)
        if self.encoder.normalize_before:
            hs_pad = self.encoder.after_norm(hs_pad)


        # CTC forward
        ys = [y[y != self.ignore_id] for y in ys_pad]
        y_len = max([len(y) for y in ys])
        ys_pad = ys_pad[:, :y_len]
        if dec_mask is not None:
            dec_mask = dec_mask[:, :y_len+1, :hs_pad.shape[1]]
        self.hs_pad = hs_pad
        batch_size = xs_pad.size(0)
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)

        # trigger mask
        hs_mask = hs_mask & dec_mask if dec_mask is not None else hs_mask
        # 2. forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        self.pred_pad = pred_pad

        # 3. compute attention loss
        loss_att = self.criterion(pred_pad, ys_out_pad)
        self.acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                               ignore_label=self.ignore_id)


        # copyied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        return self.loss, loss_ctc_data, loss_att_data, self.acc

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x, mask=None):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0).cuda()
        if mask is not None:
            mask = mask.cuda()
        if isinstance(self.encoder.embed, EncoderConv2d):
            hs, _ = self.encoder.embed(x, torch.Tensor([float(x.shape[1])]).cuda())
        else:
            hs, _ = self.encoder.embed(x, None)
        hs, _ = self.encoder.encoders(hs, mask)
        if self.encoder.normalize_before:
            hs = self.encoder.after_norm(hs)
        return hs.squeeze(0)

    def viterbi_decode(self, x, y, mask=None):
        enc_output = self.encode(x, mask)
        logits = self.ctc.ctc_lo(enc_output).detach().data
        logit = np.array(logits.cpu().data).T
        align = viterbi_align(logit, y)[0]
        return align

    def ctc_decode(self, x, mask=None):
        enc_output = self.encode(x, mask)
        logits = self.ctc.argmax(enc_output.view(1, -1, 512)).detach().data
        path = np.array(logits.cpu()[0])
        return path

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        enc_output = self.encode(x).unsqueeze(0)
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info('input lengths: ' + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}
        if lpz is not None:
            import numpy

            from espnet.nets.ctc_prefix_score import CTCPrefixScore

            ctc_prefix_score = CTCPrefixScore(lpz.cpu().detach().numpy(), 0, self.eos, numpy)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0).cuda()
                ys = torch.tensor(hyp['yseq']).unsqueeze(0).cuda()
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.forward_one_step,
                                                         (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(ys, ys_mask, enc_output)[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0].cpu(), hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]].cpu() \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]].cpu()
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            from espnet.nets.e2e_asr_common import end_detect
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps

    def prefix_recognize(self, x, recog_args, train_args, char_list=None, rnnlm=None):
        '''recognize feat

        :param ndnarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list

        TODO(karita): do not recompute previous attention for faster decoding
        '''
        pad_len = self.eos - len(char_list) + 1
        for i in range(pad_len):
            char_list.append('<eos>')
        if isinstance(self.encoder.embed, EncoderConv2d):
            seq_len = ((x.shape[0]+1)//2+1)//2
        else:
            seq_len = ((x.shape[0]-1)//2-1)//2

        if train_args.chunk:
            s = np.arange(0, seq_len, train_args.chunk_size)
            mask = adaptive_enc_mask(seq_len, s).unsqueeze(0)
        else:
            mask = turncated_mask(1, seq_len, train_args.left_window, train_args.right_window)
        enc_output = self.encode(x, mask).unsqueeze(0)
        lpz = torch.nn.functional.softmax(self.ctc.ctc_lo(enc_output), dim=-1)
        lpz = lpz.squeeze(0)

        h = enc_output.squeeze(0)

        logging.info('input lengths: ' + str(h.size(0)))
        h_len = h.size(0)
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None, 'seq': char_list[y],
               'last_time': [], "ctc_score": 0.0,  "rnnlm_score": 0.0, "att_score": 0.0,
               "cache": None, "precache": None, "preatt_score": 0.0, "prev_score":0.0}

        hyps = {char_list[y]: hyp}
        hyps_att = {char_list[y]: hyp}
        Pb_prev, Pnb_prev = Counter(), Counter()
        Pb, Pnb = Counter(), Counter()
        Pjoint = Counter()
        lpz = lpz.cpu().detach().numpy()
        vocab_size = lpz.shape[1]
        r = np.ndarray((vocab_size), dtype=np.float32)
        l = char_list[y]
        Pb_prev[l] = 1
        Pnb_prev[l] = 0
        A_prev = [l]
        A_prev_id = [[y]]
        vy.unsqueeze(1)
        total_copy = time.time() - time.time()
        samelen = 0
        hat_att = {}
        if mask is not None:
            chunk_pos = set(np.array(mask.sum(dim=-1))[0])
            for i in chunk_pos:
                hat_att[i] = {}
        else:
            hat_att[enc_output.shape[1]] = {}

        for i in range(h_len):
            hyps_ctc = {}
            threshold = recog_args.threshold#self.threshold #np.percentile(r, 98)
            pos_ctc = np.where(lpz[i] > threshold)[0]
            #self.removeIlegal(hyps)
            if mask is not None:
                chunk_index = mask[0][i].sum().item()
            else:
                chunk_index = h_len
            hyps_res = {}
            for l, hyp in hyps.items():
                if l in hat_att[chunk_index]:
                    hyp['tmp_cache'] = hat_att[chunk_index][l]['cache']
                    hyp['tmp_att'] = hat_att[chunk_index][l]['att_scores']
                else:
                    hyps_res[l] = hyp
            tmp = self.clusterbyLength(hyps_res) # This step clusters hyps according to length dict:{length,hyps}
            start = time.time()

            # pre-compute beam
            self.compute_hyps(tmp,i,h_len,enc_output, hat_att[chunk_index], mask, train_args.chunk)
            total_copy += time.time()-start
            # Assign score and tokens to hyps
            #print(hyps.keys())
            for l, hyp in hyps.items():
                if 'tmp_att' not in hyp:
                    continue #Todo check why
                local_att_scores = hyp['tmp_att']
                local_best_scores, local_best_ids = torch.topk(local_att_scores, 5, dim=1)
                pos_att = np.array(local_best_ids[0].cpu())
                pos = np.union1d(pos_ctc, pos_att)
                hyp['pos'] = pos

            # pre-compute ctc beam
            hyps_ctc_compute = self.get_ctchyps2compute(hyps,hyps_ctc,i)
            hyps_res2 = {}
            for l, hyp in hyps_ctc_compute.items():
                l_minus = ' '.join(l.split()[:-1])
                if l_minus in hat_att[chunk_index]:
                    hyp['tmp_cur_new_cache'] = hat_att[chunk_index][l_minus]['cache']
                    hyp['tmp_cur_att_scores'] = hat_att[chunk_index][l_minus]['att_scores']
                else:
                    hyps_res2[l] = hyp
            tmp2_cluster = self.clusterbyLength(hyps_res2)
            self.compute_hyps_ctc(tmp2_cluster,h_len,enc_output, hat_att[chunk_index], mask, train_args.chunk)

            for l, hyp in hyps.items():
                start = time.time()
                l_id = hyp['yseq']
                l_end = l_id[-1]
                vy[0] = l_end
                prefix_len = len(l_id)
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                else:
                    rnnlm_state = None
                    local_lm_scores = torch.zeros([1, len(char_list)])

                r = lpz[i] * (Pb_prev[l] + Pnb_prev[l])

                start = time.time()
                if 'tmp_att' not in hyp:
                    continue #Todo check why
                local_att_scores = hyp['tmp_att']
                new_cache = hyp['tmp_cache']
                align = [0] * prefix_len
                align[:prefix_len - 1] = hyp['last_time'][:]
                align[-1] = i
                pos = hyp['pos']
                if 0 in pos or l_end in pos:
                    if l not in hyps_ctc:
                        hyps_ctc[l] = {'yseq': l_id}
                        hyps_ctc[l]['rnnlm_prev'] = hyp['rnnlm_prev']
                        hyps_ctc[l]['rnnlm_score'] = hyp['rnnlm_score']
                        if l_end != self.eos:
                            hyps_ctc[l]['last_time'] = [0] * prefix_len
                            hyps_ctc[l]['last_time'][:] = hyp['last_time'][:]
                            hyps_ctc[l]['last_time'][-1] = i
                            cur_att_scores = hyps_ctc_compute[l]["tmp_cur_att_scores"]
                            cur_new_cache = hyps_ctc_compute[l]["tmp_cur_new_cache"]
                            hyps_ctc[l]['att_score'] = hyp['preatt_score'] + \
                                                       float(cur_att_scores[0, l_end].data)
                            hyps_ctc[l]['cur_att'] = float(cur_att_scores[0, l_end].data)
                            hyps_ctc[l]['cache'] = cur_new_cache
                        else:
                            if len(hyps_ctc[l]["yseq"]) > 1:
                                hyps_ctc[l]["end"] = True
                            hyps_ctc[l]['last_time'] = []
                            hyps_ctc[l]['att_score'] = hyp['att_score']
                            hyps_ctc[l]['cur_att'] = 0
                            hyps_ctc[l]['cache'] = hyp['cache']

                        hyps_ctc[l]['prev_score'] = hyp['prev_score']
                        hyps_ctc[l]['preatt_score'] = hyp['preatt_score']
                        hyps_ctc[l]['precache'] = hyp['precache']
                        hyps_ctc[l]['seq'] = hyp['seq']


                for c in list(pos):
                    if c == 0:
                        Pb[l] += lpz[i][0] * (Pb_prev[l] + Pnb_prev[l])
                    else:
                        l_plus = l+ " " +char_list[c]
                        if l_plus not in hyps_ctc:
                            hyps_ctc[l_plus] = {}
                            if "end" in hyp:
                                hyps_ctc[l_plus]['yseq'] = True
                            hyps_ctc[l_plus]['yseq'] = [0] * (prefix_len + 1)
                            hyps_ctc[l_plus]['yseq'][:len(hyp['yseq'])] = l_id
                            hyps_ctc[l_plus]['yseq'][-1] = int(c)
                            hyps_ctc[l_plus]['rnnlm_prev'] = rnnlm_state
                            hyps_ctc[l_plus]['rnnlm_score'] = hyp['rnnlm_score'] + float(local_lm_scores[0, c].data)
                            hyps_ctc[l_plus]['att_score'] = hyp['att_score'] \
                                                            + float(local_att_scores[0, c].data)
                            hyps_ctc[l_plus]['cur_att'] = float(local_att_scores[0, c].data)
                            hyps_ctc[l_plus]['cache'] = new_cache
                            hyps_ctc[l_plus]['precache'] = hyp['cache']
                            hyps_ctc[l_plus]['preatt_score'] = hyp['att_score']
                            hyps_ctc[l_plus]['prev_score'] = hyp['score']
                            hyps_ctc[l_plus]['last_time'] = align
                            hyps_ctc[l_plus]['rule_penalty'] = 0
                            hyps_ctc[l_plus]['seq'] = l_plus
                        if l_end != self.eos and c == l_end:
                            Pnb[l_plus] += lpz[i][l_end] * Pb_prev[l]
                            Pnb[l] += lpz[i][l_end] * Pnb_prev[l]
                        else:
                            Pnb[l_plus] += r[c]


                        if l_plus not in hyps:
                            Pb[l_plus] += lpz[i][0] * (Pb_prev[l_plus] + Pnb_prev[l_plus])
                            Pnb[l_plus] += lpz[i][c] * Pnb_prev[l_plus]
            #total_copy += time.time() - start
            for l in hyps_ctc.keys():
                if Pb[l] != 0 or Pnb[l] != 0:
                    hyps_ctc[l]['ctc_score'] = np.log(Pb[l] + Pnb[l])
                else:
                    hyps_ctc[l]['ctc_score'] = float('-inf')
                local_score = hyps_ctc[l]['ctc_score'] + recog_args.ctc_lm_weight * hyps_ctc[l]['rnnlm_score'] + \
                             recog_args.penalty * (len(hyps_ctc[l]['yseq']))
                hyps_ctc[l]['local_score'] = local_score
                hyps_ctc[l]['score'] = (1-recog_args.ctc_weight) * hyps_ctc[l]['att_score'] \
                                       + recog_args.ctc_weight * hyps_ctc[l]['ctc_score'] + \
                                       recog_args.penalty * (len(hyps_ctc[l]['yseq'])) + \
                                       recog_args.lm_weight * hyps_ctc[l]['rnnlm_score']
            Pb_prev = Pb
            Pnb_prev = Pnb
            Pb = Counter()
            Pnb = Counter()
            hyps1 = sorted(hyps_ctc.items(), key=lambda x: x[1]['local_score'], reverse=True)[:beam]
            hyps1 = dict(hyps1)
            hyps2 = sorted(hyps_ctc.items(), key=lambda x: x[1]['att_score'], reverse=True)[:beam]
            hyps2 = dict(hyps2)
            hyps = sorted(hyps_ctc.items(), key=lambda x: x[1]['score'], reverse=True)[:beam]
            hyps = dict(hyps)
            for key in hyps1.keys():
                if key not in hyps:
                    hyps[key] = hyps1[key]
            for key in hyps2.keys():
                if key not in hyps:
                    hyps[key] = hyps2[key]
        hyps = sorted(hyps.items() , key=lambda x: x[1]['score'], reverse=True)[:beam]
        hyps = dict(hyps)
        logging.info('input lengths: ' + str(h.size(0)))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))
        if "<eos>" in hyps.keys():
            del hyps["<eos>"]
        #for key in hyps.keys():
        #    logging.info("{0}\tctc:{1}\tatt:{2}\trnnlm:{3}\tscore:{4}".format(key,hyps[key]["ctc_score"],hyps[key]['att_score'],
        #                                        hyps[key]['rnnlm_score'], hyps[key]['score']))
        #     print("!!!","Decoding None")
        best = list(hyps.keys())[0]
        ids = hyps[best]['yseq']
        score = hyps[best]['score']
        logging.info('score: ' + str(score))
        #if l in hyps.keys():
        #    logging.info(l)

        #print(samelen,h_len)
        return best, ids, score

    def removeIlegal(self,hyps):
        max_y = max([len(hyp['yseq']) for l, hyp in hyps.items()])
        for_remove = []
        for l, hyp in hyps.items():
            if max_y - len(hyp['yseq']) > 4:
                for_remove.append(l)
        for cur_str in for_remove:
            del hyps[cur_str]

    def clusterbyLength(self,hyps):
        tmp = {}
        for l, hyp in hyps.items():
            prefix_len = len(hyp['yseq'])
            if prefix_len > 1 and hyp['yseq'][-1] == self.eos:
                continue
            else:
                if prefix_len not in tmp:
                    tmp[prefix_len] = []
                tmp[prefix_len].append(hyp)
        return tmp


    def compute_hyps(self, current_hyps, curren_frame,total_frame,enc_output, hat_att, enc_mask, chunk=True):
        for length, hyps_t in current_hyps.items():
            ys_mask = subsequent_mask(length).unsqueeze(0).cuda()
            ys_mask4use = ys_mask.repeat(len(hyps_t), 1, 1)

            # print(ys_mask4use.shape)
            l_id = [hyp_t['yseq'] for hyp_t in hyps_t]
            ys4use = torch.tensor(l_id).cuda()
            enc_output4use = enc_output.repeat(len(hyps_t), 1, 1)
            if hyps_t[0]["cache"] is None:
                cache4use = None
            else:
                cache4use = []
                for decode_num in range(len(hyps_t[0]["cache"])):
                    current_cache = []
                    for hyp_t in hyps_t:
                        current_cache.append(hyp_t["cache"][decode_num].squeeze(0))
                    # print( torch.stack(current_cache).shape)

                    current_cache = torch.stack(current_cache)
                    cache4use.append(current_cache)

            partial_mask4use = []
            for hyp_t in hyps_t:
                #partial_mask4use.append(torch.ones([1, len(hyp_t['last_time'])+1, enc_mask.shape[1]]).byte())
                align = [0] * length
                align[:length - 1] = hyp_t['last_time'][:]
                align[-1] = curren_frame
                align_tensor = torch.tensor(align).unsqueeze(0)
                if chunk:
                    partial_mask = enc_mask[0][align_tensor]
                else:
                    right_window = self.right_window
                    partial_mask = trigger_mask(1, total_frame, align_tensor,
                                            self.left_window, right_window)
                partial_mask4use.append(partial_mask)

            partial_mask4use = torch.stack(partial_mask4use).cuda().squeeze(1)
            local_att_scores_b, new_cache_b = self.decoder.forward_one_step(ys4use, ys_mask4use,
                                                                            enc_output4use, partial_mask4use, cache4use)
            for idx, hyp_t in enumerate(hyps_t):
                hyp_t['tmp_cache'] = [new_cache_b[decode_num][idx].unsqueeze(0)
                                      for decode_num in range(len(new_cache_b))]
                hyp_t['tmp_att'] = local_att_scores_b[idx].unsqueeze(0)
                hat_att[hyp_t['seq']] = {}
                hat_att[hyp_t['seq']]['cache'] = hyp_t['tmp_cache']
                hat_att[hyp_t['seq']]['att_scores'] = hyp_t['tmp_att']

    def get_ctchyps2compute(self,hyps,hyps_ctc,current_frame):
        tmp2 = {}
        for l, hyp in hyps.items():
            l_id = hyp['yseq']
            l_end = l_id[-1]
            if "pos" not in hyp:
                continue
            if 0 in hyp['pos'] or l_end in hyp['pos']:
                #l_minus = ' '.join(l.split()[:-1])
                #if l_minus in hat_att:
                #    hyps[l]['tmp_cur_new_cache'] = hat_att[l_minus]['cache']
                #    hyps[l]['tmp_cur_att_scores'] = hat_att[l_minus]['att_scores']
                #    continue
                if l not in hyps_ctc and l_end != self.eos:
                    tmp2[l] = {'yseq': l_id}
                    tmp2[l]['seq'] = l
                    tmp2[l]['rnnlm_prev'] = hyp['rnnlm_prev']
                    tmp2[l]['rnnlm_score'] = hyp['rnnlm_score']
                    if l_end != self.eos:
                        tmp2[l]['last_time'] = [0] * len(l_id)
                        tmp2[l]['last_time'][:] = hyp['last_time'][:]
                        tmp2[l]['last_time'][-1] = current_frame
        return tmp2

    def compute_hyps_ctc(self,hyps_ctc_cluster,total_frame,enc_output, hat_att, enc_mask, chunk=True):
        for length, hyps_t in hyps_ctc_cluster.items():
            ys_mask = subsequent_mask(length - 1).unsqueeze(0).cuda()
            ys_mask4use = ys_mask.repeat(len(hyps_t), 1, 1)
            l_id = [hyp_t['yseq'][:-1] for hyp_t in hyps_t]
            ys4use = torch.tensor(l_id).cuda()
            enc_output4use = enc_output.repeat(len(hyps_t), 1, 1)
            if "precache" not in hyps_t[0] or hyps_t[0]["precache"] is None:
                cache4use = None
            else:
                cache4use = []
                for decode_num in range(len(hyps_t[0]["precache"])):
                    current_cache = []
                    for hyp_t in hyps_t:
                        # print(length, hyp_t["yseq"], hyp_t["cache"][0].shape,
                        #       hyp_t["cache"][2].shape, hyp_t["cache"][4].shape)
                        current_cache.append(hyp_t["precache"][decode_num].squeeze(0))
                    current_cache = torch.stack(current_cache)
                    cache4use.append(current_cache)
            partial_mask4use = []
            for hyp_t in hyps_t:
                #partial_mask4use.append(torch.ones([1, len(hyp_t['last_time']), enc_mask.shape[1]]).byte())
                align = hyp_t['last_time']
                align_tensor = torch.tensor(align).unsqueeze(0)
                if chunk:
                    partial_mask = enc_mask[0][align_tensor]
                else:
                    right_window = self.right_window
                    partial_mask = trigger_mask(1, total_frame, align_tensor, self.left_window, right_window)
                partial_mask4use.append(partial_mask)

            partial_mask4use = torch.stack(partial_mask4use).cuda().squeeze(1)

            local_att_scores_b, new_cache_b = \
                self.decoder.forward_one_step(ys4use, ys_mask4use,
                                              enc_output4use, partial_mask4use, cache4use)
            for idx, hyp_t in enumerate(hyps_t):
                hyp_t['tmp_cur_new_cache'] = [new_cache_b[decode_num][idx].unsqueeze(0)
                                              for decode_num in range(len(new_cache_b))]
                hyp_t['tmp_cur_att_scores'] = local_att_scores_b[idx].unsqueeze(0)
                l_minus = ' '.join(hyp_t['seq'].split()[:-1])
                hat_att[l_minus] = {}
                hat_att[l_minus]['att_scores'] = hyp_t['tmp_cur_att_scores']
                hat_att[l_minus]['cache'] = hyp_t['tmp_cur_new_cache']

