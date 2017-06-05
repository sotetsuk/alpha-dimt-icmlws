#!/bin/sh

cd models
wget https://storage.googleapis.com/alpha-dimt-icmlws-models/iwslt14_ml_04be179_acc_60.63_ppl_9.23_e18.pt
wget https://storage.googleapis.com/alpha-dimt-icmlws-models/iwslt14_alpha_04be179_alpha_0.0_0.0_tau_3.0_acc_60.15_ppl_9.55_e24.pt
wget https://storage.googleapis.com/alpha-dimt-icmlws-models/iwslt14_alpha_04be179_alpha_0.3_0.3_tau_3.0_acc_60.14_ppl_9.74_e26.pt
wget https://storage.googleapis.com/alpha-dimt-icmlws-models/iwslt14_alpha_04be179_alpha_0.5_0.5_tau_3.0_acc_60.19_ppl_9.78_e21.pt
wget https://storage.googleapis.com/alpha-dimt-icmlws-models/iwslt14_alpha_04be179_alpha_0.7_0.7_tau_3.0_acc_59.72_ppl_10.43_e25.pt
cd ..
