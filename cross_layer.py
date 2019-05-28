import sys
sys.path.insert(0, './constraint')

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from n3 import *


class CrossLayer(torch.nn.Module):
	def __init__(self, opt, shared):
		super(CrossLayer, self).__init__()
		assert(opt.num_att_label == 1)

		self.opt = opt
		self.shared = shared
		self.cross_constr = nn.ModuleList(self.get_cross_constr(opt.cross_constr))

		self.constr_on_att1 = False
		self.constr_on_att2 = False
		self.constr_customized = False
		for t in self.opt.constr_on.split(','):
			if t == '1':
				self.constr_on_att1 = True
			elif t == '2':
				self.constr_on_att2 = True
			elif t == '3':
				self.constr_customized = True
			else:
				assert(False)

		# 
		self.one = Variable(torch.ones(1), requires_grad=False)
		# alternatively, use nn.ParameterList 
		rho_c = torch.Tensor([float(c) for c in opt.rho_c.split(',')])
		if opt.gpuid != -1:
			rho_c = rho_c.cuda()
		self.rho_c = nn.Parameter(rho_c, requires_grad=False)

		if len(self.cross_constr) != 0:
			print('cross constraint enabled')

	# the function that grabs constraints
	def get_cross_constr(self, names):
		layers = []
		if names == '':
			return layers
	
		for n in names.split(','):
			if n == 'n3':
				layers.append(N3(self.opt, self.shared))
			else:
				print('unrecognized constraint layer name: {0}'.format(n))
				assert(False)
	
		return layers


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


	def forward(self, att1, att2, y_score):
		# logic pass
		#dy_ls = []
		#for layer, rho in zip(self.cross_constr, self.rho_c):
		#	if self.constr_on_att1:
		#		dy_ls.append(rho * layer(att1.transpose(1,2)).unsqueeze(0))
		#	if self.constr_on_att2:
		#		dy_ls.append(rho * layer(att2).unsqueeze(0))
#
		#rhoed_dy = self.d_joiner(dy_ls).sum(0)

		# composite constraint scores
		#constrained_score = y_score + rhoed_dy

		y_score = y_score.unsqueeze(0)	# (1, batch_l, num_label)
		constr_y_score = []
		for layer, rho in zip(self.cross_constr, self.rho_c):
			if self.constr_customized:
				dy = rho * layer(att1.transpose(1,2), att2).unsqueeze(0)
				constr_y_score.append(y_score + dy)
			else:
				if self.constr_on_att1:
					dy = rho * layer(att1.transpose(1,2)).unsqueeze(0)
					constr_y_score.append(y_score + dy)
				if self.constr_on_att2:
					dy = rho * layer(att2).unsqueeze(0)
					constr_y_score.append(y_score + dy)

		constr_y_score = torch.cat(constr_y_score, 0).sum(0)

		# stats
		#self.shared.x_hit_cnt = (rhoed_dy.data.sum(1) > 0.0).sum()
		#self.shared.dy = rhoed_dy
		#self.shared.rho_c = self.rho_c[0]	# 

		return constr_y_score


	def get_param_dict(self, root):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		#param_dict['rho_c'] = torch2np(self.rho_c.data, is_cuda)
		#if self.opt.logic_att == 1:
		#	param_dict.update(self.logic_att.get_param_dict('logic_att'))

		return param_dict


