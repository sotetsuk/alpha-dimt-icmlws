import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import Constants


class AlphaLoss(nn.Module):

    def __init__(self, weight=None):
        super(AlphaLoss, self).__init__()
        self.weight = weight  # weight for imbalance # of class or PAD class
        self.use_cuda = False

    def forward(self, log_outputs, targets, proposed_weights, log_q_weights, alpha, rewards):
        """Return weighted negative log likelihood loss

        :param log_outputs: seq_len * batch_size x voc_size
        :param targets: seq_len * batch_size
        :param q_weights: batch_size
        """
        batch_size = log_q_weights.size()[0]

        mask = np.zeros(log_outputs.size())
        mask[np.arange(mask.shape[0]), targets.cpu().data.numpy().astype(np.int32)] = 1.
        mask = Variable(torch.from_numpy(mask).float())
        cls_weight = Variable(self.weight.unsqueeze(0).expand(log_outputs.size()).float())

        # cuda TODO(sotestuk): fix to use more elegant way
        if self.use_cuda:
            mask = mask.cuda()
            cls_weight = cls_weight.cuda()

        nll = - log_outputs * cls_weight * mask  # seq_len * batch_size x voc_size
        nll = nll.sum(dim=1).squeeze()  # seq_len * batch_size
        nll = nll.view(-1, batch_size)  # seq_len x batch_size
        seq_len = nll.size(0)
        nll = nll.sum(dim=0).squeeze()  # batch_size

        log_p_weights = - nll.data.cpu()
        p = np.exp(log_p_weights / seq_len)
        p = p / p.sum()
        q = np.exp(log_q_weights)
        q = q / q.sum()
        alpha_weights = np.power(p, alpha) * np.power(q, (1 -alpha))
        proposed_weights = proposed_weights / proposed_weights.sum()  # proposal weight is normalized for mini-batch
        alpha_weights = proposed_weights * alpha_weights 
        alpha_weights = alpha_weights / alpha_weights.sum()
        alpha_weights = Variable(alpha_weights, requires_grad=False)

        if self.use_cuda:
            alpha_weights = alpha_weights.cuda()

        weighted_log_likelihood = nll * alpha_weights * batch_size  # rescale to the same scale as ML loss by * batch_size
        loss = weighted_log_likelihood.sum()

        # calculate sample efficiency
        p_sample_efficiency_list = list(p * torch.FloatTensor(rewards))
        q_sample_efficiency_list = list(q * torch.FloatTensor(rewards))
        pq_sample_efficiency_list = list(alpha_weights.data.cpu() * torch.FloatTensor(rewards))

        return loss, list(torch.exp(log_p_weights - log_q_weights)), p_sample_efficiency_list, q_sample_efficiency_list, pq_sample_efficiency_list


def AlphaCriterion(vocabSize, gpus):
    weight = torch.ones(vocabSize)
    weight[Constants.PAD] = 0
    crit = AlphaLoss(weight)
    if gpus:
        crit.cuda()
        crit.use_cuda = True
    return crit


def NMTCriterion(vocabSize, gpus):
    weight = torch.ones(vocabSize)
    weight[Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if gpus:
        crit.cuda()
    return crit


def memoryEfficientLoss(outputs, targets, generator, crit, max_generator_batches, eval=False):
    """Memory efficient loss.

    :param outputs: seq_len x batch_size x logits_size
    :param targets: seq_len x batch_size
    :param generator:
    :param crit:
    :param max_generator_batches:
    :param eval:
    :return:
    """
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)  # seq_len x batch_size x logits_size

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, max_generator_batches)
    targets_split = torch.split(targets, max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        # out_t = seq_len x batch_size x logits_size
        # targ_t = seq_len x batch_size

        out_t = out_t.view(-1, out_t.size(2))  # seq_len * batch_size x logits_size
        scores_t = generator(out_t)  # seq_len * batch_size x voc_size

        loss_t = crit(scores_t, targ_t.view(-1))  # scholar (1-d)

        pred_t = scores_t.max(1)[1]  # seq_len * batch_size x 1

        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(Constants.PAD).data).sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, num_correct


def alpha_loss(outputs, targets, generator, crit, max_generator_batches, rewards, proposed_weights, tau, alpha, eval=False):
    """Loss function of proposed method.

    :param outputs: seq_len x batch_size x logits_size
    :param targets: seq_len x batch_size
    :param generator:
    :param crit:
    :param max_generator_batches:
    :param eval:
    :return:
    """
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)  # seq_len x batch_size x logits_size

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, max_generator_batches)
    targets_split = torch.split(targets, max_generator_batches)

    # TODO(sotetsuk): fix to calculate at once
    importance_list = []
    p_sample_efficiency_list = []
    q_sample_efficiency_list = []
    pq_sample_efficiency_list = []
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))  # seq_len * batch_size x logits_size
        scores_t = generator(out_t)  # seq_len * batch_size x voc_size

        proposed_weights = torch.FloatTensor(proposed_weights)
        log_q_weights = torch.FloatTensor(rewards) / tau

        loss_t, importance_t, p_sample_efficiency_t, q_sample_efficiency_t, pq_sample_efficiency_t = crit(scores_t, targ_t.view(-1), proposed_weights, log_q_weights, alpha, rewards)  # scholar (1-d)

        pred_t = scores_t.max(1)[1]  # seq_len * batch_size x 1

        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(Constants.PAD).data).sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        importance_list += importance_t
        p_sample_efficiency_list += p_sample_efficiency_t
        q_sample_efficiency_list += q_sample_efficiency_t
        pq_sample_efficiency_list += pq_sample_efficiency_t
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, num_correct, importance_list, p_sample_efficiency_list, q_sample_efficiency_list, pq_sample_efficiency_list
