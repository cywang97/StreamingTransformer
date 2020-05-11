#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""

import math
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

def make_pad_mask(lens):
    """Function to make mask tensor containing indices of padded part

        e.g.: lengths = [5, 3, 2]
              mask = [[1, 1, 1, 1 ,1],
                      [1, 1, 1, 0, 0],
                      [1, 1, 0, 0, 0]]

        :param torch.Tensor lengths: (B)
        :return: mask tensor containing indices of padded part (B, Tmax)
        :rtype: torch.Tensor
        """
    if len(lens.shape) > 0:
        pass
    else:
        lens = lens.unsqueeze(0)
    # return torch.arange(lens.max(),
    #                     device=lens.device).repeat(size, 1) < lens.unsqueeze(-1)
    return torch.arange(lens.max(),
                        device=lens.device).repeat(lens.size(0), 1) < lens.unsqueeze(-1)


class DecoderConv1dPosition(torch.nn.Module):
    def __init__(self, odim, dim):
        super(DecoderConv1dPosition, self).__init__()

        self.embed = torch.nn.Embedding(odim, 512)
        self.position = PositionalEncoding(odim, 1)
        self.conv1 = torch.nn.Conv1d(512, 512, 3, stride=1)
        self.norm1 = torch.nn.LayerNorm(512)
        self.conv2 = torch.nn.Conv1d(512, 512, 3, stride=1)
        self.norm2 = torch.nn.LayerNorm(512)
        self.conv3 = torch.nn.Conv1d(512, 512, 3, stride=1)
        self.norm3 = torch.nn.LayerNorm(512)

        self.out = torch.nn.Linear(512, dim)

    def forward(self, ys_pad, mask):
        '''
        :param torch.Tensor ys_pad: batch of padded input sequence ids (B, Tmax)
        :param torch.Tensor mask: batch of input length mask (B, Tmax)
        :return: batch of padded hidden state sequences (B, Tmax, 512)
        :rtype: torch.Tensor
        '''

        ys_pad = self.embed(ys_pad)
        ys_pad = self.position(ys_pad)
        mask = ~mask.unsqueeze(-1)

        ys_pad = F.pad(ys_pad, (0, 0, 2, 0))
        ys_pad = self.conv1(ys_pad.transpose(1, 2)).transpose(1, 2)
        ys_pad = self.norm1(ys_pad)
        ys_pad = F.relu(ys_pad)
        ys_pad = ys_pad.masked_fill(mask, 0.0)

        ys_pad = F.pad(ys_pad, (0, 0, 2, 0))
        ys_pad = self.conv2(ys_pad.transpose(1, 2)).transpose(1, 2)
        ys_pad = self.norm2(ys_pad)
        ys_pad = F.relu(ys_pad)
        ys_pad = ys_pad.masked_fill(mask, 0.0)

        ys_pad = F.pad(ys_pad, (0, 0, 2, 0))
        ys_pad = self.conv3(ys_pad.transpose(1, 2)).transpose(1, 2)
        ys_pad = self.norm3(ys_pad)
        ys_pad = F.relu(ys_pad)
        ys_pad = ys_pad.masked_fill(mask, 0.0)

        ys_pad = self.out(ys_pad)
        ys_pad = ys_pad.masked_fill(mask, 0.0)

        return ys_pad


class DecoderConv1d(torch.nn.Module):
    def __init__(self, odim, dim):
        super(DecoderConv1d, self).__init__()

        self.embed = torch.nn.Embedding(odim, 512)
        self.conv1 = torch.nn.Conv1d(512, 512, 3, stride=1)
        self.norm1 = torch.nn.LayerNorm(512)
        self.conv2 = torch.nn.Conv1d(512, 512, 3, stride=1)
        self.norm2 = torch.nn.LayerNorm(512)
        self.conv3 = torch.nn.Conv1d(512, 512, 3, stride=1)
        self.norm3 = torch.nn.LayerNorm(512)

        self.out = torch.nn.Linear(512, dim)

    def forward(self, ys_pad, mask):
        '''
        :param torch.Tensor ys_pad: batch of padded input sequence ids (B, Tmax)
        :param torch.Tensor mask: batch of input length mask (B, Tmax)
        :return: batch of padded hidden state sequences (B, Tmax, 512)
        :rtype: torch.Tensor
        '''

        ys_pad = self.embed(ys_pad)
        mask = ~mask.unsqueeze(-1)

        ys_pad = F.pad(ys_pad, (0, 0, 2, 0))
        ys_pad = self.conv1(ys_pad.transpose(1, 2)).transpose(1, 2)
        ys_pad = self.norm1(ys_pad)
        ys_pad = F.relu(ys_pad)
        ys_pad = ys_pad.masked_fill(mask, 0.0)

        ys_pad = F.pad(ys_pad, (0, 0, 2, 0))
        ys_pad = self.conv2(ys_pad.transpose(1, 2)).transpose(1, 2)
        ys_pad = self.norm2(ys_pad)
        ys_pad = F.relu(ys_pad)
        ys_pad = ys_pad.masked_fill(mask, 0.0)

        ys_pad = F.pad(ys_pad, (0, 0, 2, 0))
        ys_pad = self.conv3(ys_pad.transpose(1, 2)).transpose(1, 2)
        ys_pad = self.norm3(ys_pad)
        ys_pad = F.relu(ys_pad)
        ys_pad = ys_pad.masked_fill(mask, 0.0)

        ys_pad = self.out(ys_pad)
        ys_pad = ys_pad.masked_fill(mask, 0.0)

        return ys_pad

class EncoderConv2d(torch.nn.Module):
    """VGG-like module

    :param int dim: output dim for next step
    """

    def __init__(self, idim, dim):
        super(EncoderConv2d, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.norm1_1 = torch.nn.LayerNorm([64, idim])
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.norm1_2 = torch.nn.LayerNorm([64, idim])
        idim = int(math.ceil(float(idim) / 2))

        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.norm2_1 = torch.nn.LayerNorm([128, idim])
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.norm2_2 = torch.nn.LayerNorm([128, idim])
        idim = int(math.ceil(float(idim) / 2))

        self.out = torch.nn.Linear(idim * 128, dim)

    def forward(self, xs_pad, ilens):
        '''
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of input length (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128)
        :rtype: torch.Tensor
        '''

        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.unsqueeze(1)

        mask = ~make_pad_mask(ilens).unsqueeze(1).unsqueeze(-1)
        xs_pad = self.conv1_1(xs_pad)
        xs_pad = self.norm1_1(xs_pad.transpose(1, 2)).transpose(1, 2)
        xs_pad = F.relu(xs_pad)
        xs_pad = xs_pad.masked_fill(mask, 0.0)

        xs_pad = self.conv1_2(xs_pad).masked_fill(mask, 0.0)
        xs_pad = self.norm1_2(xs_pad.transpose(1, 2)).transpose(1, 2)
        xs_pad = F.relu(xs_pad)
        xs_pad = xs_pad.masked_fill(mask, 0.0)

        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        ilens = torch.ceil(ilens.float() / 2).long()

        mask = ~make_pad_mask(ilens).unsqueeze(1).unsqueeze(-1)
        xs_pad = self.conv2_1(xs_pad)
        xs_pad = self.norm2_1(xs_pad.transpose(1, 2)).transpose(1, 2)
        xs_pad = F.relu(xs_pad)
        xs_pad = xs_pad.masked_fill(mask, 0.0)

        xs_pad = self.conv2_2(xs_pad)
        xs_pad = self.norm2_2(xs_pad.transpose(1, 2)).transpose(1, 2)
        xs_pad = F.relu(xs_pad)
        xs_pad = xs_pad.masked_fill(mask, 0.0)

        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        ilens = torch.ceil(ilens.float() / 2).long()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        mask = ~make_pad_mask(ilens).unsqueeze(-1)
        xs_pad = xs_pad.transpose(1, 2).contiguous()
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), -1)
        xs_pad = self.out(xs_pad)
        xs_pad = xs_pad.masked_fill(mask, 0.0)

        return xs_pad, ~mask.squeeze(-1)
