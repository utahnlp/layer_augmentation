import sys
sys.path.insert(0, '../')

import torch
from torch import nn

def activate(one, d):
	return relu(one - d)

def relu(x):
	#return nn.LeakyReLU()(x)
	return nn.ReLU()(x)

# select a > theta
def geq_theta(theta, a):
	m = (a >= theta).float()
	return a * m

def untag_nul_on_selector2(selector2):
	selector2[:, 0, :] = 0.0
	selector2[:, 1:, 0] = 0.0
	return selector2

def get_h_selector(opt, shared, res_name, with_nul=False):
	mask = torch.zeros(shared.batch_l, shared.sent_l2)	# mask that marks the columns

	# if res_name does not present, mark all words
	if res_name is None:
		mask[:, 1:] = 1.0
		if with_nul:
			mask[:, 0] = 1.0
	else:
		for ex, pair in enumerate(shared.res_map[res_name]):
			h_contents = pair[1]
	
			if len(h_contents) != 0:
				mask[ex][h_contents] = 1.0
			if with_nul:
				mask[ex][0] = 1.0
		
	if opt.gpuid != -1:
		mask = mask.cuda()
	return mask

def get_p_selector(opt, shared, res_name, with_nul=False):
	mask = torch.zeros(shared.batch_l, shared.sent_l1)

	# if res_name does not present, mark all words
	if res_name is None:
		mask[:, 1:] = 1.0
		if with_nul:
			mask[:, 0] = 1.0
	else:
		for ex, pair in enumerate(shared.res_map[res_name]):
			p_contents = pair[0]
	
			if len(p_contents) != 0:
				mask[ex][p_contents] = 1.0
			if with_nul:
				mask[ex][0] = 1.0
		
	if opt.gpuid != -1:
		mask = mask.cuda()
	return mask

def get_h_nul_selector(opt, shared):
	mask = torch.zeros(shared.batch_l, shared.sent_l2)
	mask[:, 0] = 1.0
	if opt.gpuid != -1:
		mask = mask.cuda()
	return mask

def get_p_nul_selector(opt, shared):
	mask = torch.zeros(shared.batch_l, shared.sent_l1)
	mask[:, 0] = 1.0
	if opt.gpuid != -1:
		mask = mask.cuda()
	return mask


def get_rel_selector2(opt, shared, res_name):
	mask = torch.zeros(shared.batch_l, shared.sent_l2, shared.sent_l1)

	for ex in range(shared.batch_l):
		rel_res = shared.res_map[res_name][ex]
		rel_keys = rel_res[res_name]

		for k in rel_keys:
			src_idx, tgt_idx = rel_res[k]

			for i in tgt_idx:
				mask[ex][i][src_idx] = 1.0
	if opt.gpuid != -1:
		mask = mask.cuda()

	return mask