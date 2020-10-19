import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-pre_ep', '--pretrain_epoch')
parser.add_argument('-rank_ep', '--ranking_epoch')
parser.add_argument('--gpu')
args = parser.parse_args()

os.system("CUDA_VISIBLE_DEVICES=%s python2 train_ug_pretrain.py --max_epoch_pre %s" % (args.gpu, args.pretrain_epoch))
os.system("CUDA_VISIBLE_DEVICES=%s python2 train_ug_ranking_pretrain.py --max_epoch_pre %s --max_epoch %s" % (args.gpu, args.pretrain_epoch, args.ranking_epoch))


