"""Transducer speech recognition model (pytorch)."""

from distutils.util import strtobool

import logging

import math
import numpy as np

import torch
import time
import pdb


from espnet.nets.asr_interface import ASRInterface

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor

from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter, pad_list

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, turncated_mask, trigger_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.scorers.ctc import CTCPrefixScorer

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (Namespace): argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Extend arguments for transducer models.

        Both Transformer and RNN modules are supported.
        General options encapsulate both modules options.

        """
        group = parser.add_argument_group("transformer model setting")

        # Encoder - general
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--kernel-size', default=32, type=int)
        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder - RNN
        group.add_argument('--eprojs', default=320, type=int,
                           help='Number of encoder projection units')
        group.add_argument('--subsample', default="1", type=str,
                           help='Subsample input frames x_y_z means subsample every x frame \
                           at 1st layer, every y frame at 2nd layer etc.')
        # Attention - general
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')

        # Decoder - general
        group.add_argument('--dtype', default='lstm', type=str,
                           choices=['lstm', 'gru', 'transformer'],
                           help='Type of decoder to use.')
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                           help='Dropout rate for the decoder')
        group.add_argument('--relative-v', default=False, type=strtobool)
        # Decoder - RNN
        group.add_argument('--dec-embed-dim', default=320, type=int,
                           help='Number of decoder embeddings dimensions')
        group.add_argument('--dropout-rate-embed-decoder', default=0.0, type=float,
                           help='Dropout rate for the decoder embeddings')
        # Transformer
        group.add_argument("--input-layer", type=str, default="conv2d",
                           choices=["conv2d", "vgg2l", "linear", "embed", "custom"],
                           help='transformer encoder input layer type')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=False, type=strtobool,
                           help='normalize loss by length')
        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument('--causal', type=strtobool, default=False)


        return parser

    def __init__(self, idim, odim, args, ignore_id=-1, blank_id=0):
        """Construct an E2E object for transducer model.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs
            args (Namespace): argument Namespace containing options

        """
        torch.nn.Module.__init__(self)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(
            idim=idim,
            d_model=args.adim,
            n_heads=args.aheads,
            d_ffn=args.eunits,
            layers=args.elayers,
            kernel_size=args.kernel_size,
            input_layer=args.input_layer,
            dropout_rate=args.dropout_rate,
            causal=args.causal)

        args.eprojs = args.adim
        if args.mtlalpha < 1.0:
            self.decoder = Decoder(
                odim=odim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate
            )


        self.sos = odim - 1
        self.eos = odim - 1
        self.ignore_id = ignore_id

        self.odim = odim
        self.adim = args.adim

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

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    
    def forward(self, xs_pad, ilens, ys_pad, enc_mask=None):
        """E2E forward.

        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)

        Returns:
            loss (torch.Tensor): transducer loss value

        """
        # 1. encoder
        xs_pad = xs_pad[:, :max(ilens)]
        masks = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, masks)

        self.hs_pad = hs_pad


        # CTC forward
        ys = [y[y != self.ignore_id] for y in ys_pad]
        y_len = max([len(y) for y in ys])
        ys_pad = ys_pad[:, :y_len]
        self.hs_pad = hs_pad
        cer_ctc = None
        batch_size = xs_pad.size(0)
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)

        # trigger mask
        start_time = time.time()
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

        return self.loss, loss_att_data, loss_ctc_data, self.acc

    def encode(self, x, streaming_mask=None):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, attention_dim)
        """
        self.eval()

        x = torch.as_tensor(x).unsqueeze(0).cuda()
        enc_output, _ = self.encoder(x, None, streaming_mask)

        return enc_output.squeeze(0)


    def greedy_recognize(self, x, recog_args, streaming_mask=None):
        h = self.encode(x, streaming_mask).unsqueeze(0)
        lpz = self.ctc.log_softmax(h)
        ys_hat = lpz.argmax(dim=-1)[0].cpu()
        ids = []
        for i in range(len(ys_hat)):
            if ys_hat[i].item() != 0 and ys_hat[i].item() != ys_hat[i-1].item():
                ids.append(ys_hat[i].item())
        rets = [{'score': 0.0, 'yseq': ids}]
        return rets

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, streaming_mask=None):
        """
        Recognize input features.

        Args:
            x (ndarray): input acoustic feature (T, D)
            recog_args (namespace): argument Namespace containing options
            char_list (list): list of characters
            rnnlm (torch.nn.Module): language model module

        Returns:
            y (list): n-best decoding results

        """
        h = self.encode(x, streaming_mask)
        params = [h, recog_args]

        if recog_args.beam_size == 1:
            nbest_hyps = self.decoder.recognize(*params)
        else:
            params.append(rnnlm)
            start_time = time.time() 
            nbest_hyps = self.decoder.recognize_beam(*params)
            #nbest_hyps, decoder_time = self.decoder.beam_search(h, recog_args, prefix=False)
            end_time = time.time()
            decode_time = end_time - start_time

        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded character id sequence tensor (B, Lmax)

        Returns:
            ret (ndarray): attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).

        """
        if self.etype == 'transformer' and self.dtype != 'transformer' and \
           self.rnnt_mode == 'rnnt-att':
            raise NotImplementedError("Transformer encoder with rnn attention decoder"
                                      "is not supported yet.")
        elif self.etype != 'transformer' and self.dtype != 'transformer':
            if self.rnnt_mode == 'rnnt':
                return []
            else:
                with torch.no_grad():
                    hs_pad, hlens = xs_pad, ilens
                    hpad, hlens, _ = self.encoder(hs_pad, hlens)

                    ret = self.decoder.calculate_all_attentions(hpad, hlens, ys_pad)
        else:
            with torch.no_grad():
                self.forward(xs_pad, ilens, ys_pad)

                ret = dict()
                for name, m in self.named_modules():
                    if isinstance(m, MultiHeadedAttention):
                        ret[name] = m.attn.cpu().numpy()

        return ret
