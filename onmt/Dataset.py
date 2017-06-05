from __future__ import division

import math
import random
from itertools import product

import numpy as np
from scipy.misc import comb
import torch
from torch.autograd import Variable

import onmt


class Dataset(object):

    def __init__(self, srcData, tgtData, batchSize, cuda, volatile=False):
        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src)/batchSize)
        self.volatile = volatile

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index*self.batchSize:(index+1)*self.batchSize])
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch = zip(*batch)
        else:
            indices, srcBatch, tgtBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        return (wrap(srcBatch), lengths), wrap(tgtBatch), indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])


class ISDataset(object):
    """Dataset for importance sampling."""

    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.sampler = sampler

        self.src = self.dataset.src
        if self.dataset.tgt:
            self.tgt = self.dataset.tgt
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = self.dataset.cuda

        self.batchSize = self.dataset.batchSize
        self.numBatches = self.dataset.numBatches
        self.volatile = self.dataset.volatile

    def __getitem__(self, index):
        """Return a batch.

        EXAMPLE
        -------
        >>> src = [torch.LongTensor([0, 1, 2]), torch.LongTensor([3, 4, 5]), torch.LongTensor([6, 7, 8])]
        >>> tgt = [torch.LongTensor([0, 1, 2]), torch.LongTensor([3, 4, 5]), torch.LongTensor([6, 7, 8])]
        >>> dataset = Dataset(src, tgt, 2, [])
        >>> sampler = onmt.EditDistanceSampler([0])
        >>> aug_dataset = ISDataset(dataset, sampler)
        >>> src, tgt, indices, rewards, proposed_weights = aug_dataset[0]
        """
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self.dataset._batchify(
            self.src[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch, rewards, proposed_weights = self.sampler.sample(self.tgt[index*self.batchSize:(index+1)*self.batchSize])
            tgtBatch = self.dataset._batchify(tgtBatch)
        else:
            tgtBatch = None
            rewards = None
            proposed_weights = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        batch = zip(indices, srcBatch) if tgtBatch is None\
            else zip(indices, srcBatch, tgtBatch, rewards, proposed_weights)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch = zip(*batch)
        else:
            indices, srcBatch, tgtBatch, rewards, proposed_weights = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        return (wrap(srcBatch), lengths), wrap(tgtBatch), indices, rewards, proposed_weights

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        self.dataset.shuffle()
