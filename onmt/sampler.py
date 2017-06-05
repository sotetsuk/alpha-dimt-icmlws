from __future__ import division

import numpy as np
import scipy.misc as misc
import nltk
import torch


class Sampler(object):

    def __init__(self):
        pass

    def sample(self, tgt_data):
        """Return augmented tgt_data, rewards, and weights of proposed distribution"""
        pass

    def reward(self, y, y_ast, score=None):
        """Return reward of (y, y*)"""
        pass


class HammingDistanceSampler(Sampler):

    def __init__(self, temperature, max_len, voc_min, voc_max):
        """Sampling augmented target sentences and importance weights.

        EXAMPLE
        -------
        >>> edit_sampler = HammingDistanceSampler(0.5, 10, 0, 9)
        >>> tgt_data = [torch.LongTensor([0, 3, 5, 4, 3]), torch.LongTensor([2, 5, 1, 3, 9]), torch.LongTensor([2, 0, 1, 4, 5])]
        >>> augmented_tgt_data, rewards, proposed_weights = edit_sampler.sample(tgt_data, [2, 3, 4])
        >>> for s in tgt_data:
        ...     print(s)
        >>> for s in augmented_tgt_data:
        ...     print(s)
        >>> for r in rewards:
        ...     print(r)
        >>> for p in proposed_weights:
        ...     print(p)

        ARGUMENTS
        ---------
        voc_min: minimum index of vocabulary (inclusive)
        voc_max: maximum index of vocabulary (inclusive)

        """
        super(HammingDistanceSampler, self).__init__()
        self.temperature = temperature
        self.max_len = max_len
        self.voc_min = voc_min
        self.voc_max = voc_max
        self.voc_size = voc_max - voc_min + 1
        self.edit_frac = 0.2

        p = np.ones((self.max_len + 1, self.max_len))  # p[len_target, n_edit] = values of proposal distributions
        self.p = p

    def sample(self, tgt_data, edit_list=None):
        """Augment tgt_data and return importance sampling weight

        ARGS
        ----
        tgt_data: list of Torch.LongTensor
        edit_list: specify the edit length for each sample (especially for debug)

        Types of augmentation include: - deletion
        """

        if edit_list is not None:
            assert len(edit_list) == len(tgt_data)

        new_tgt_data = []
        rewards = []
        proposal_weights = []
        for i, tgt in enumerate(tgt_data):
            len_tgt = len(tgt)
            max_edit = int(len_tgt * self.edit_frac)  # in this study, we define max_edit as length of sentence

            # define edit distance for this tgt sentence
            if edit_list is not None:
                e = edit_list[i]
            else:
                e = np.random.choice(max_edit + 1, 1)[0]  # choose from {0, ..., max_edit}

            # get proposal weights
            proposal_weights.append(self.p[len_tgt, e])

            # execute augmentation
            n_substitutions = e
            # substitutions
            substituted_ixs = list(np.random.choice(len_tgt, n_substitutions, replace=False))
            new_tgt = []
            for j, t in enumerate(tgt):
                if j in substituted_ixs:
                    new_token = t
                    while new_token == t:
                        new_token = int(np.random.choice(self.voc_size, 1)[0]) + self.voc_min
                    new_tgt.append(new_token)
                else:
                    new_tgt.append(t)

            # store the corresponding reward
            reward = float(self.reward(tgt, new_tgt, e))
            rewards.append(reward)

            new_tgt_data.append(torch.LongTensor(new_tgt))

        return new_tgt_data, rewards, proposal_weights

    def reward(self, y, y_ast, score=None):
        """Return minus of edit distance.

        :param y:
        :param y_ast:
        :param score: edit distance
        :return:
        """
        if score is not None:
            return - score

            # TODO: calc and return edit distance of (y, y*)


class BLEUSampler(Sampler):

    def __init__(self, edit_cand=[0, 1, 2, 3, 4, 5]):
        """Sampling augmented target sentences and importance weights.

        EXAMPLE
        -------
        >>> sampler = BLEUSampler()
        >>> tgt_data = [torch.LongTensor([0, 3, 5, 4, 3]), torch.LongTensor([2, 5, 1, 3, 9, 3]), torch.LongTensor([2, 0, 1, 4, 5])]
        >>> augmented_tgt_data, rewards, weight = sampler.sample(tgt_data, [2, 2, 1])
        >>> for s in tgt_data:
        ...     print(s.size())
        torch.Size([5])
        torch.Size([6])
        torch.Size([5])
        >>> for s in augmented_tgt_data:
        ...     print(s.size())
        torch.Size([3])
        torch.Size([4])
        torch.Size([4])
        >>> for r in rewards:
        ...     print(r)
        >>> for w in weight:
        ...     print(w)
        1.0
        1.0
        1.0

        """
        super(BLEUSampler, self).__init__()
        self.edit_cand = edit_cand

    def sample(self, tgt_data, edit_list=None):
        """Augment tgt_data and return importance sampling weight

        ARGS
        ----
        tgt_data: list of Torch.LongTensor
        edit_list: specify the edit length for each sample (especially for debug)

        Types of augmentation include:
          - deletion
        """

        if edit_list is not None:
            assert len(edit_list) == len(tgt_data)

        new_tgt_data = []
        rewards = []
        for i, tgt in enumerate(tgt_data):
            # Define edit distance for this tgt sentence
            if edit_list is not None:
                e = edit_list[i]
            else:
                e = np.random.choice(self.edit_cand, 1)[0]

            # Execute augmentation
            len_seq = len(tgt)
            e = max(min(len_seq - 1, e), 0)
            deleted_ixs = np.random.choice(len_seq, e, replace=False)
            new_tgt = []
            for j, t in enumerate(tgt):
                if j in deleted_ixs:
                    continue
                new_tgt.append(t)

            # Store the corresponding reward
            reward = float(self.reward(tgt, new_tgt))
            rewards.append(reward)

            new_tgt_data.append(torch.LongTensor(new_tgt))

        return new_tgt_data, rewards, [1. for _ in range(len(rewards))]

    def reward(self, y, y_ast):
        """return BLEU score

        EXAMPLE
        -------
        >>> sampler = BLEUSampler()
        >>> sampler.reward(["This", "is", "a", "pen", "."], ["This", "are", "a", "pen", "."])
        0.6042750794713536

        """
        blue_score = nltk.translate.bleu_score.sentence_bleu([y_ast], y)
        return blue_score
