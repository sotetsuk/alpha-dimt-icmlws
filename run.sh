#!/bin/sh

wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
./download_iwslt14.sh
./iwslt14.sh --preprocess

./iwslt14.sh --train-ml 0
./iwslt14.sh --train-alpha 3.0 0.0 0.0 0
./iwslt14.sh --train-alpha 3.0 0.3 0.3 0
./iwslt14.sh --train-alpha 3.0 0.5 0.5 0
./iwslt14.sh --train-alpha 3.0 0.7 0.7 0

./show_results.sh
