import sys
sys.path.insert(0, './constraint')

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import time
from holder import *
from util import *
from n1 import *
from n2 import *

class WithinLayer(torch.nn.Module):
	def __init__(self, opt, shared):
		super(WithinLayer, self).__init__()
		self.opt = opt
		self.shared = shared
		#self.num_att_labels = opt.num_att_labels
		self.within_constr = self.get_within_constr(opt.within_constr)

		self.constr_on_att1 = False
		self.constr_on_att2 = False
		for t in self.opt.constr_on.split(','):
			if t == '1':
				self.constr_on_att1 = True
			elif t == '2':
				self.constr_on_att2 = True
			else:
				pass

		self.zero = Variable(torch.zeros(1), requires_grad=False)
		rho_w = torch.ones(1) * opt.rho_w
		if opt.gpuid != -1:
			rho_w = rho_w.cuda()
			self.zero = self.zero.cuda()
		self.rho_w = nn.Parameter(rho_w, requires_grad=False)

		if len(self.within_constr) != 0:
			print('within-layer constraint enabled')

	# DEPRECATED
	def grow_rho(self, x):
		rs = None
		if self.opt.grow_rho == 'log':
			# the log_10(epoch)
			rs = torch.log(torch.ones(1) * float(x)) / torch.log(torch.ones(1) * 10.0)
		elif self.opt.grow_rho == '1log':
			# the log_10(epoch) + 1
			rs = torch.log(torch.ones(1) * float(x)) / torch.log(torch.ones(1) * 10.0) + 1.0
		elif self.opt.grow_rho == 'inv':
			# 1 - 1/epoch
			rs = torch.ones(1) - torch.ones(1) / (torch.ones(1) * float(x))
		if self.opt.gpuid != -1:
			rs = rs.cuda()
		return rs

	# the function that grabs constraints
	def get_within_constr(self, names):
		layers = []
		if names == '':
			return layers
	
		for n in names.split(','):
			if n == 'n1':
				layers.append(N1(self.opt, self.shared))
			elif n == 'n2':
				layers.append(N2(self.opt, self.shared))
			else:
				print('unrecognized constraint layer name: {0}'.format(n))
				assert(False)
	
		return layers

	def forward(self, score1, score2, att1, att2):
		batch_l = self.shared.batch_l
		sent_l1 = self.shared.sent_l1
		sent_l2 = self.shared.sent_l2

		# logic pass
		batch_l = self.shared.batch_l
		datt1_ls = []
		datt2_ls = []
		for layer in self.within_constr:
			if self.constr_on_att1:
				datt1_ls.append(layer(att1.transpose(1,2)).transpose(1,2).contiguous().view(1, batch_l, sent_l1, sent_l2))
			if self.constr_on_att2:
				datt2_ls.append(layer(att2).view(1, batch_l, sent_l2, sent_l1))

		datt1 = self.zero
		datt2 = self.zero
		if len(datt1_ls) != 0:
			datt1 = torch.cat(datt1_ls, 0).sum(0)
		if len(datt2_ls) != 0:
			datt2 = torch.cat(datt2_ls, 0).sum(0)
			# stats
			self.shared.w_hit_cnt = (datt2.data.sum(-1).sum(-1) > 0.0).sum()

		rho_w = self.rho_w

		constrained_score1 = score1 + rho_w * datt1
		constrained_score2 = score2 + rho_w * datt2

		# stats
		self.shared.rho_w = rho_w

		return [constrained_score1, constrained_score2]




