from __future__ import division

import subprocess
import sys
import numpy as np
import codecs
import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import nltk

parser = argparse.ArgumentParser(description='train.py')

## Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-valid_src', required=True,
                    help='Path to test src')
parser.add_argument('-valid_tgt', required=True,
                    help='path to test tgt')
parser.add_argument('-test_src', required=True,
                    help='Path to test src')
parser.add_argument('-test_tgt', required=True,
                    help='path to test tgt')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

## Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

## Optimization options

parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")

#learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")

#pretrained word vectors

parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpu', default=-1, type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")

# RAML( and RAML alpha)
parser.add_argument('-raml', action="store_true",
                    help="""Use RAML training""")
parser.add_argument('-raml_alpha', action="store_true",
                    help="""Use RAML(alpha) training""")
parser.add_argument('-tau', type=float, default=0.85,
                    help="Parameter for entropy regularization")
parser.add_argument('-alpha', type=float, default=0.5,
                    help="Parameter for RAML(alpha)")
parser.add_argument('-alpha_increase', type=float, default=1.1,
                    help="Parameter for RAML(alpha)")
parser.add_argument('-alpha_max', type=float, default=0.90,
                    help="Parameter for RAML(alpha)")
parser.add_argument('-alpha_increase_epoch', type=int, default=10,
                    help="Parameter for RAML(alpha)")
parser.add_argument('-edit_cand_max', type=int, default=5,
                    help="Max edit-distance for data augmentation (inclusive).")


opt = parser.parse_args()

print(opt)
sys.stdout.flush()

if torch.cuda.is_available() and not opt.gpu >= 0:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.gpu >= 0:
    cuda.set_device(opt.gpu)

opt.cuda = opt.gpu >= 0

if opt.raml_alpha:
    alpha = [opt.alpha]  # TODO: fix

scores = []
max_epoch = [-1]
max_valid = [0.]
max_test = [0.]

def eval(model, criterion, data):
    total_loss = 0
    total_words = 0
    total_num_correct = 0

    model.eval()
    for i in range(len(data)):
        batch = data[i][:2] # exclude original indices
        outputs = model(batch)
        targets = batch[1][1:]  # exclude <s> from targets
        loss, _, num_correct = onmt.memoryEfficientLoss(
                outputs, targets, model.generator, criterion, opt.max_generator_batches, eval=True)
        total_loss += loss
        total_num_correct += num_correct
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return total_loss / total_words, total_num_correct / total_words


def trainModel(model, trainData, validData, dataset, optim):
    print(model)
    sys.stdout.flush()
    model.train()

    # define criterion of each GPU
    criterion = onmt.NMTCriterion(dataset['dicts']['tgt'].size(), opt.cuda)
    if opt.raml_alpha:
        train_criterion = onmt.AlphaCriterion(dataset['dicts']['tgt'].size(), opt.cuda)
    else:
        train_criterion = criterion

    start_time = time.time()
    def trainEpoch(epoch):

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
        start = time.time()
        importance_list = []
        p_sample_efficiency_list = []
        q_sample_efficiency_list = []
        pq_sample_efficiency_list = []
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx]

            model.zero_grad()
            outputs = model(batch[:2])  # exclude original indices
            targets = batch[1][1:]  # exclude <s> from targets
            if opt.raml_alpha:
                rewards = batch[-2]
                proposed_weights = batch[-1]
                loss, gradOutput, num_correct, _importance_list, _p_sample_efficiency_list, _q_sample_efficiency_list, _pq_sample_efficiency_list = onmt.alpha_loss(
                    outputs, targets, model.generator, train_criterion, opt.max_generator_batches,
                    rewards, proposed_weights, opt.tau, alpha[0])
            else:
                loss, gradOutput, num_correct = onmt.memoryEfficientLoss(
                    outputs, targets, model.generator, train_criterion, opt.max_generator_batches)

            outputs.backward(gradOutput)

            # update the parameters
            optim.step()

            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss += loss
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += sum(batch[0][1])
            total_loss += loss
            total_num_correct += num_correct
            total_words += num_words
            if opt.raml_alpha:
                importance_list += _importance_list
                p_sample_efficiency_list += _p_sample_efficiency_list
                q_sample_efficiency_list += _q_sample_efficiency_list
                pq_sample_efficiency_list += _pq_sample_efficiency_list
            if i % opt.log_interval == -1 % opt.log_interval:
                print("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                      (epoch, i+1, len(trainData),
                      report_num_correct / report_tgt_words * 100,
                      math.exp(report_loss / report_tgt_words),
                      report_src_words/(time.time()-start),
                      report_tgt_words/(time.time()-start),
                      time.time()-start_time))
                sys.stdout.flush()

                report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                start = time.time()

        return total_loss / total_words, total_num_correct / total_words, importance_list, p_sample_efficiency_list, q_sample_efficiency_list, pq_sample_efficiency_list

    epoch = 0
    while True:
        epoch += 1
        print('')

        #  (1) train for one epoch on the training set
        if opt.raml_alpha:
            train_loss, train_acc, train_importance_list, p_sample_efficiency_list, q_sample_efficiency_list, pq_sample_efficiency_list = trainEpoch(epoch)
            train_importance_list = np.array(train_importance_list)
            p_sample_efficiency_list = np.array(p_sample_efficiency_list)
            q_sample_efficiency_list = np.array(q_sample_efficiency_list)
            pq_sample_efficiency_list = np.array(pq_sample_efficiency_list)
            # print('Train importance mean: %g' % train_importance_list.mean())
            # print('Train importance std: %g' % train_importance_list.std())
            # print('P Sample efficiency mean: %g' % p_sample_efficiency_list.mean())
            # print('P Sample efficiency std: %g' % p_sample_efficiency_list.std())
            # print('Q Sample efficiency mean: %g' % q_sample_efficiency_list.mean())
            # print('Q Sample efficiency std: %g' % q_sample_efficiency_list.std())
            # print('PQ-mix Sample efficiency mean: %g' % pq_sample_efficiency_list.mean())
            # print('PQ-mix Sample efficiency std: %g' % pq_sample_efficiency_list.std())
        else:
            train_loss, train_acc, importance_list, _, _, _ = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)
        print('Train accuracy: %g' % (train_acc*100))

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, criterion, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        print('Validation accuracy: %g' % (valid_acc*100))

        bleu = translate(model, opt.valid_src, opt.valid_tgt, dataset['dicts']['src'], dataset['dicts']['tgt'], beam_size=1)
        valid_bleu = float(bleu.split(' ')[2][:-1])
        print("Valid BLEU score: {}".format(valid_bleu))
        test_bleu = 0.0
        if valid_bleu > max_valid[0]:
            bleu = translate(model, opt.test_src, opt.test_tgt, dataset['dicts']['src'], dataset['dicts']['tgt'], beam_size=1)
            test_bleu = float(bleu.split(' ')[2][:-1])
            print("Test BLEU score: {}".format(test_bleu))

            max_epoch[0] =  epoch
            max_valid[0] = valid_bleu
            max_test[0] = test_bleu
            print("Max vlidation bleu score updated!")
            best_msg = "[BEST] Max Valid BLEU, Test BLEU: {}, {} @epoch {}".format(max_valid[0], max_test[0], max_epoch[0]) + ' model: %s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, 100*valid_acc, valid_ppl, epoch)
            print(best_msg)
        scores.append((epoch, valid_bleu, test_bleu))

        #  (3) update the learning rate
        decayed = optim.dev_decay(valid_bleu)
        if opt.raml_alpha and decayed:
            alpha[0] += opt.alpha_increase
            if alpha[0] > opt.alpha_max:
                alpha[0] = opt.alpha_max
            print("Update alpha to {} @epoch {}".format(alpha[0], epoch))

        model_state_dict = model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }
        print('%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, 100*valid_acc, valid_ppl, epoch))
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, 100*valid_acc, valid_ppl, epoch))

        if epoch > opt.epochs and epoch >= max_epoch[0] + 5:
            return

def translate(model, src, tgt, src_dict, tgt_dict, beam_size=10):
    opt.beam_size = beam_size
    opt.n_best = 1
    opt.replace_unk = True

    def addone(f):
        for line in f:
            yield line
        yield None

    translator = onmt.Translator(opt, model, src_dict, tgt_dict)

    srcBatch, tgtBatch = [], []

    tgtF = codecs.open(tgt, 'r', 'utf-8')
    pred_list = []
    out_name = 'tmp/' + opt.save_model.split('/')[-1] + '.tmp'
    out = codecs.open(out_name, 'w', 'utf-8')
    for line in addone(codecs.open(src, 'r', 'utf-8')):

        if line is not None:
            srcTokens = line.split()
            srcBatch += [srcTokens]
            tgtTokens = tgtF.readline().split()
            tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        predBatch, _, _ = translator.translate(srcBatch, tgtBatch)
        pred_list += predBatch

        srcBatch, tgtBatch = [], []

    for pred in pred_list:
        out.write(' '.join(pred[0]) + '\n')

    tgtF.close()
    out.close()

    bleu_results = subprocess.Popen('perl -X multi-bleu.perl ' + tgt + ' < ' + out_name, stdout=subprocess.PIPE,
                     shell=True).stdout.readline()

    model.train()
    model.decoder.attn.applyMask(None)

    return str(bleu_results)


def main():

    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    dict_checkpoint = opt.train_from if opt.train_from else opt.train_from_state_dict
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.cuda)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.cuda,
                             volatile=True)

    if opt.raml_alpha:
        print("Use RAML(alpha) ...")
        print("tau: {}".format(opt.tau))
        print("alpha: {}".format(opt.alpha))
        sampler = onmt.HammingDistanceSampler(temperature=opt.tau, max_len=55, voc_min=4, voc_max=dataset['dicts']['tgt'].size()-4)
        trainData = onmt.ISDataset(trainData, sampler)

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    encoder = onmt.Models.Encoder(opt, dicts['src'])
    decoder = onmt.Models.Decoder(opt, dicts['tgt'])

    generator = nn.Sequential(
        nn.Linear(opt.rnn_size, dicts['tgt'].size()),
        nn.LogSoftmax())

    model = onmt.Models.NMTModel(encoder, decoder)

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = checkpoint['generator']
        model_state_dict = {k: v for k, v in chk_model if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from checkpoint at %s' % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.cuda:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.load_pretrained_vectors(opt)
        decoder.load_pretrained_vectors(opt)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        optim.lr = opt.learning_rate
        optim.start_decay_at = opt.start_decay_at
        optim.lr_decay = opt.learning_rate_decay
        optim.start_decay = False
        print(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset, optim)
    print("\nBest: Valid BLEU: {}, Test BLEU: {} @epoch {}\n".format(max_valid[0], max_test[0], max_epoch[0]))
    print("Epoch, Valid BLEU, Test BLEU")
    print("-" * 30)
    for score in scores:
        epoch, valid_bleu, test_bleu = score
        print("{}: {}, {}".format(epoch, valid_bleu, test_bleu))

if __name__ == "__main__":
    main()
