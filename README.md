# Neural Sequence Model Training via α-divergence Minimization

The reference implementation for this paper:

```
@inproceedings{alphadimt,
 title={Neural Sequence Model Training via $\alpha$-divergence Minimization,
 author={Koyamada, Sotetsu and Kikuchi, Yuta and Kanemura, Atsunori and Maeda, Shin-ichi and Ishii, Shin},
 booktitle={Proceedings of the 2017 ICML workshop on Learning to Generate Natural Language (LGNL)},
 year={2017}
}
```

Note that
- This implementation is a fork from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- The code is provided only for replication purposes, further development is not planned.

## Dependency

- numpy
- scipy
- [PyTorch](http://pytorch.org/)

## Usage

```
$ ./run.sh
```

## Pretrained models/log

### log

```sh
$ ls log
gpu_0_iwslt14_04be179_alpha_0.0_0.0_tau_3.0.log
gpu_1_iwslt14_04be179_alpha_0.3_0.3_tau_3.0.log
gpu_2_iwslt14_04be179_alpha_0.5_0.5_tau_3.0.log
gpu_3_iwslt14_04be179_alpha_0.7_0.7_tau_3.0.log
gpu_4_iwslt14_ml_04be179.log
$ ./show_results.sh
gpu_0_iwslt14_04be179_alpha_0.0_0.0_tau_3.0.log
[BEST] Max Valid BLEU, Test BLEU: 29.65, 27.5 @epoch 24 model: models/iwslt14_alpha_04be179_alpha_0.0_0.0_tau_3.0_acc_60.15_ppl_9.55_e24.pt
gpu_1_iwslt14_04be179_alpha_0.3_0.3_tau_3.0.log
[BEST] Max Valid BLEU, Test BLEU: 29.9, 27.73 @epoch 26 model: models/iwslt14_alpha_04be179_alpha_0.3_0.3_tau_3.0_acc_60.14_ppl_9.74_e26.pt
gpu_2_iwslt14_04be179_alpha_0.5_0.5_tau_3.0.log
[BEST] Max Valid BLEU, Test BLEU: 29.91, 28.02 @epoch 21 model: models/iwslt14_alpha_04be179_alpha_0.5_0.5_tau_3.0_acc_60.19_ppl_9.78_e21.pt
gpu_3_iwslt14_04be179_alpha_0.7_0.7_tau_3.0.log
[BEST] Max Valid BLEU, Test BLEU: 29.72, 27.81 @epoch 25 model: models/iwslt14_alpha_04be179_alpha_0.7_0.7_tau_3.0_acc_59.72_ppl_10.43_e25.pt
gpu_4_iwslt14_ml_04be179.log
[BEST] Max Valid BLEU, Test BLEU: 29.83, 27.96 @epoch 18 model: models/iwslt14_ml_04be179_acc_60.63_ppl_9.23_e18.pt
$ ls results | grep bs_10.bleu
iwslt14_0_results_iwslt14_ml_04be179_acc_60.63_ppl_9.23_e18.pt_bs_10.bleu
iwslt14_2_results_iwslt14_alpha_04be179_alpha_0.0_0.0_tau_3.0_acc_60.15_ppl_9.55_e24.pt_bs_10.bleu
iwslt14_2_results_iwslt14_alpha_04be179_alpha_0.3_0.3_tau_3.0_acc_60.14_ppl_9.74_e26.pt_bs_10.bleu
iwslt14_2_results_iwslt14_alpha_04be179_alpha_0.5_0.5_tau_3.0_acc_60.19_ppl_9.78_e21.pt_bs_10.bleu
iwslt14_2_results_iwslt14_alpha_04be179_alpha_0.7_0.7_tau_3.0_acc_59.72_ppl_10.43_e25.pt_bs_10.bleu
$ ls results | grep bs_10.bleu | awk '{print "cat results/"$1}' | sh
BLEU = 28.26, 63.8/36.9/23.1/14.8 (BP=0.944, ratio=0.946, hyp_len=124030, ref_len=131141)
BLEU = 28.35, 63.4/36.6/22.9/14.7 (BP=0.954, ratio=0.955, hyp_len=125289, ref_len=131141)
BLEU = 28.29, 64.1/37.0/23.1/14.8 (BP=0.943, ratio=0.945, hyp_len=123921, ref_len=131141)
BLEU = 28.49, 64.1/37.2/23.3/14.9 (BP=0.944, ratio=0.946, hyp_len=124037, ref_len=131141)
BLEU = 28.25, 63.7/36.7/23.0/14.7 (BP=0.948, ratio=0.949, hyp_len=124513, ref_len=131141)
```

### models

```sh
$ ./download_iwslt14_pretrained_models.sh
$ ./iwslt14.sh --translate 0 10 iwslt14_alpha_04be179_alpha_0.5_0.5_tau_3.0_acc_60.19_ppl_9.78_e21.pt
$ cat results/iwslt14_0_results_iwslt14_alpha_04be179_alpha_0.5_0.5_tau_3.0_acc_60.19_ppl_9.78_e21.pt_bs_10.bleu
```
