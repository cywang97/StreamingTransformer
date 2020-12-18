#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math
import logging

import numpy
import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.embedding import RelativePositionalEncoding
from espnet.nets.pytorch_backend.transformer.subsamplingV2 import Conv2dSubsampling, EncoderConv2d
from espnet.nets.pytorch_backend.transformer.repeat import repeat



class MultiHeadedAttention_XL(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_r = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def _rel_shift(self, x):
        zero_pad = torch.zeros(*x.size()[0:-1], 1).to(x.device)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[0:2], x.size()[3]+1, x.size()[2])
        x = x_padded[:, :, 1:].view_as(x)
        return x

    def forward(self, query, key, value, r_emb, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)   #(b, t, d)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        r = self.linear_r(r_emb).view(x_len, -1, self.h, self.d_k)   # (t, t, h, d)

        rw_head_q = q + r_w_bias
        rr_head_q = q + r_r_bias

        rw_head_q = rw_head_q.transpose(1,2)
        rr_head_q = rr_head_q.permute([1, 2, 0, 3])   #(b, tq, h, d)  -> (tq, h, b, d)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        r = r.transpose(1,2)           #(tq, tk, h, d)  -> (tq, h, tk, d)

        AC = torch.matmul(rw_head_q, k.transpose(-2, -1))

        BD = torch.matmul(rr_head_q, r.transpose(-2, -1))
        #BD = self._rel_shift(BD)
        BD = BD.permute([2, 1, 0, 3])

        scores = (AC + BD) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)

        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.layer_norm = nn.LayerNorm(n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, pos_k, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)   #(b, t, d)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        A = torch.matmul(q, k.transpose(-2, -1))
        reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0,1)
        B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
        B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
        scores = (A + B) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.dropout(self.linear_out(x))  # (batch, time1, d_model)

class ConvModule(nn.Module):
    def __init__(self, input_dim, kernel_size, dropout_rate, causal=False):
        super(ConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

        self.pw_conv_1 = nn.Conv2d(1, 2, 1, 1, 0)
        self.glu_act = torch.nn.Sigmoid()
        self.causal = causal
        self.kernel_size = kernel_size
        if causal:
            self.dw_conv_1d = nn.Conv1d(input_dim, input_dim, kernel_size, 1, padding=(kernel_size-1), groups=input_dim)
        else:
            self.dw_conv_1d = nn.Conv1d(input_dim, input_dim, kernel_size, 1, padding=(kernel_size-1)//2, groups=input_dim)
        #self.BN = nn.BatchNorm1d(input_dim)
        self.act = nn.ReLU()
        self.pw_conv_2 = nn.Conv2d(1, 1, 1, 1, 0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = x[:, 0] * self.glu_act(x[:, 1])
        x = x.permute([0, 2, 1])
        x = self.dw_conv_1d(x)
        if self.causal:
            x = x[:, :, :-(self.kernel_size-1)]
        #x = self.BN(x)
        x = self.act(x)
        x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.pw_conv_2(x)
        x = self.dropout(x).squeeze(1)

        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout_rate):
        super(FeedForward, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        self.layer_norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout_rate)
        )


    def forward(self, x):
        x = self.layer_norm(x)
        out = self.net(x)

        return out


class EncoderLayer(nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(self, d_model, n_head, d_ffn, kernel_size, dropout_rate, causal=False):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.feed_forward_in = FeedForward(d_model, d_ffn, dropout_rate)
        self.self_attn = MultiHeadedAttention(n_head, d_model, dropout_rate)
        self.conv = ConvModule(d_model, kernel_size, dropout_rate, causal)
        self.feed_forward_out = FeedForward(d_model, d_ffn, dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_k, mask):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x = x + 0.5 * self.feed_forward_in(x)
        x = x + self.self_attn(x, pos_k, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)

        out = self.layer_norm(x)

        return out, pos_k, mask

class Encoder(nn.Module):
    def __init__(self,
                 idim=83,
                 d_model=256,
                 n_heads=4,
                 d_ffn=2048,
                 layers=6,
                 kernel_size=32,
                 dropout_rate=0.1,
                 input_layer="conv2d",
                 padding_idx=-1,
                 causal=False
                 ):
        super(Encoder, self).__init__()


        if input_layer == "custom":
            self.embed = EncoderConv2d(idim, d_model)
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, d_model, dropout_rate)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.pos_emb = RelativePositionalEncoding(d_model // n_heads, 1000, False)
        self.encoders = repeat(
            layers,
            lambda: EncoderLayer(
                d_model,
                n_heads,
                d_ffn,
                kernel_size,
                dropout_rate,
                causal
            )
        )

    def forward(self, xs, masks, streaming_mask=None):
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        elif isinstance(self.embed, EncoderConv2d):
            if masks is None:
                xs, masks = self.embed(xs, torch.Tensor([float(xs.shape[1])]).cuda())
            else:
                xs, masks = self.embed(xs, torch.sum(masks,2).squeeze())
            masks = torch.unsqueeze(masks,1)
        else:
            xs = self.embed(xs)

        if streaming_mask is not None and masks is not None:
            hs_mask = masks & streaming_mask
        elif masks is not None:
            hs_mask = masks
        else:
            hs_mask = streaming_mask

        x_len = xs.shape[1]
        pos_seq = torch.arange(0, x_len).long().to(xs.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k, _ = self.pos_emb(pos_seq)

        xs, _, _= self.encoders(xs, pos_k, hs_mask)

        return xs, masks



