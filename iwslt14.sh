#!/usr/bin/env bash

# Preprocessing
if [ "$1" = "--preprocess" ]; then
  python preprocess.py \
    -train_src iwslt14/train.de-en.de \
    -train_tgt iwslt14/train.de-en.en \
    -valid_src iwslt14/valid.de-en.de \
    -valid_tgt iwslt14/valid.de-en.en \
    -save_data intermediates/iwslt14 \
    -src_vocab_size 32009 \
    -tgt_vocab_size 22822 \
    -seq_length 50 \
    -report_every 1000 \
    -lower
fi

if [ "$1" = "--train-ml" ]; then
  gpuno=-1
  if [ -n "$2" ]; then
        gpuno=$2
        echo "Train on GPU"${gpuno}
  fi
  sha=`git rev-parse HEAD`

  python train.py \
    -data intermediates/iwslt14.train.pt \
    -valid_src iwslt14/valid.de-en.de \
    -valid_tgt iwslt14/valid.de-en.en \
    -test_src iwslt14/test.de-en.de \
    -test_tgt iwslt14/test.de-en.en \
    -max_sent_length 180 \
    -save_model models/iwslt14_ml_${sha:0:7} \
    -layers 1 \
    -rnn_size 256 \
    -word_vec_size 256 \
    -input_feed 1 \
    -brnn \
    -brnn_merge concat \
    -batch_size 128 \
    -max_generator_batches 32 \
    -epochs 10 \
    -start_epoch 1 \
    -param_init 0.1 \
    -optim sgd \
    -max_grad_norm 5 \
    -dropout 0.3 \
    -curriculum \
    -extra_shuffle \
    -learning_rate 1.0 \
    -learning_rate_decay 0.5 \
    -log_interval 25 \
    -gpu $gpuno > log/gpu_${gpuno}_iwslt14_ml_${sha:0:7}.log
fi

if [ "$1" = "--train-alpha" ]; then
  sha=`git rev-parse HEAD`
  tau=$2
  alpha_init=$3
  alpha_max=$4
  gpuno=-1
  if [ -n "$5" ]; then
        gpuno=$5
        echo "Train on GPU"${gpuno}
  fi

  python train.py \
    -data intermediates/iwslt14.train.pt \
    -valid_src iwslt14/valid.de-en.de \
    -valid_tgt iwslt14/valid.de-en.en \
    -test_src iwslt14/test.de-en.de \
    -test_tgt iwslt14/test.de-en.en \
    -max_sent_length 180 \
    -save_model models/iwslt14_alpha_${sha:0:7}_alpha_${alpha_init}_${alpha_max}_tau_${tau} \
    -layers 1 \
    -rnn_size 256 \
    -word_vec_size 256 \
    -input_feed 1 \
    -brnn \
    -brnn_merge concat \
    -batch_size 128 \
    -max_generator_batches 32 \
    -epochs 10 \
    -start_epoch 1 \
    -param_init 0.1 \
    -optim sgd \
    -max_grad_norm 5 \
    -dropout 0.3 \
    -curriculum \
    -extra_shuffle \
    -learning_rate 1.0 \
    -learning_rate_decay 0.5 \
    -log_interval 25 \
    -gpu $gpuno    \
    -raml_alpha \
    -tau $tau \
    -alpha $alpha_init \
    -alpha_increase 0.1 \
    -alpha_max $alpha_max \
    > log/gpu_${gpuno}_iwslt14_${sha:0:7}_alpha_${alpha_init}_${alpha_max}_tau_${tau}.log
fi

if [ "$1" = "--translate" ]; then
  gpuno=$2
  bs=$3
  model=$4

  python translate.py \
    -model models/$model \
    -src iwslt14/test.de-en.de \
    -tgt iwslt14/test.de-en.en \
    -output results/iwslt14_${gpuno}_results_${model}_bs_${bs}.txt \
    -beam_size $bs \
    -batch_size 32 \
    -max_sent_length 180 \
    -replace_unk \
    -n_best 1 \
    -gpu $gpuno

    perl -X multi-bleu.perl iwslt14/test.de-en.en < results/iwslt14_${gpuno}_results_${model}_bs_${bs}.txt > results/iwslt14_${gpuno}_results_${model}_bs_${bs}.bleu
fi

