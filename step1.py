import numpy as np
from nmc import meta_correction
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--stct_epoch', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--sample_times', type=int, default=50)
parser.add_argument('--sampling_rate', type=float, default=0.5)
parser.add_argument('--nmc_epoch', type=int, default=400)
parser.add_argument('--nmc_lr', type=float, default=0.005)
parser.add_argument('--nmc_bs', type=int, default=5000)
param = parser.parse_args()


epoch = param.stct_epoch
if epoch == 0:
    feat = np.load('./init_feat.npy')
else:
    feat = np.load('./improved_feat.npy')

noisy_labels = np.load('./cifar10_noisy_labels_sym_0.9.npy')
clean_labels = np.load('./clean_labels.npy')
# We notice that if we conduct label correction from the original noisy labels every epoch, we can have a slightly
# better result.
corrected_noisy_labels = meta_correction(param=param, feat=feat, noisy_labels=noisy_labels,
                                         clean_labels=clean_labels, current_epoch=epoch)
# np.save('./noisy_labels_{}.npy'.format(epoch), corrected_noisy_labels)
