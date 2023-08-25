import numpy as np
import os
import pdb
import argparse
import shutil


def main(args):

	DataID_list = ['_'.join(ID.split('_')[0:3]) for ID in os.listdir(args.root_dir + '/' + args.experiment_dir + '/' + args.dataset_dir + '/' + args.exp_num) if not ID.startswith('.')]
	DataID_list = list(set(DataID_list))
	DataID_list.sort()

	outdir = args.root_dir + '/' + args.out_dir + '/' + args.experiment_dir + '/' + args.dataset_dir + '/' + args.exp_num

	if os.path.exists(outdir):
		shutil.rmtree(outdir)
	os.makedirs(outdir)
	pdb.set_trace()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Graph_embedding calculate results')
	parser.add_argument('-r','--root_dir', type=str, default='.', help='path for root')
	parser.add_argument('-e','--experiment_dir', type=str, default='exps_embedding128', help='experiment dir')
	parser.add_argument('-s','--dataset_dir', type=str, default='testing_results', help='test data set dir')
	parser.add_argument('-exp','--exp_num', type=str, default='exp_0', help='exp num')
	parser.add_argument('-o','--out_dir', type=str, default='calculated_results', help='output dir')

	args = parser.parse_args()
	print(args)
	main(args)