import sys
sys.path.insert(0, './encoder')
sys.path.insert(0, './attention')
sys.path.insert(0, './classifier')

import torch
from torch import nn
from torch import cuda
from holder import *
from proj_encoder import *
from rnn_encoder import *
from local_attention import *
from local_classifier import *
from torch.autograd import Variable
import numpy as np
from optimizer import *
from embeddings import *
from char_embeddings import *
import time

class Pipeline(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Pipeline, self).__init__()

		self.shared = shared
		self.opt = opt

		self.embeddings = Embeddings(opt, shared)
		if opt.use_char_enc == 1:
			self.char_embeddings = CharEmbeddings(opt, shared)

		if opt.encoder == 'proj':
			self.encoder = ProjEncoder(opt, shared)
		elif opt.encoder == 'rnn':
			self.encoder = RNNEncoder(opt, shared)
		else:
			raise Exception('unrecognized enocder: {0}'.format(opt.encoder))

		if opt.attention == 'local':
			self.attention = LocalAttention(opt, shared)
		else:
			raise Exception('unrecognized attention: {0}'.format(opt.attention))

		if opt.classifier == 'local':
			self.classifier = LocalClassifier(opt, shared)
		else:
			raise Exception('unrecognized classifier: {0}'.format(opt.classifier))


	def init_weight(self):
		missed_names = []
		if self.opt.param_init_type == 'xavier_uniform':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_uniform_(p)
						#p.data.mul_(self.opt.param_init)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'xavier_normal':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_normal_(p)
						#p.data.mul_(self.opt.param_init)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'no':
			for n, p in self.named_parameters():
				missed_names.append(n)
		else:
			assert(False)

		if len(missed_names) != 0:
			print('uninitialized fields: {0}'.format(missed_names))


	def forward(self, token1, token2, char1, char2):
		shared = self.shared

		if self.opt.use_char_enc == 1:
			char1 = self.char_embeddings(char1)	# (batch_l, context_l, token_l, char_emb_size)
			char2 = self.char_embeddings(char2)	# (batch_l, response_l, token_l, char_emb_size)
		else:
			char1, char2 = None, None

		token1 = self.embeddings(token1)	# (batch_l, context_l, word_vec_size)
		token2 = self.embeddings(token2)	# (batch_l, response_l, word_vec_size)


		input_emb1, input_emb2, input_enc1, input_enc2 = self.encoder(token1, token2)
		att1, att2 = self.attention(input_enc1, input_enc2)
		out = self.classifier(input_emb1, input_emb2, att1, att2)

		# if there is any fwd pass hooks, execute them
		if hasattr(self.opt, 'forward_hooks') and self.opt.forward_hooks != '':
			run_forward_hooks(self.opt, self.shared, self)

		return out

	# call this explicitly
	def update_context(self, batch_ex_idx, batch_l, sent_l1, sent_l2, res_map=None):
		self.shared.batch_ex_idx = batch_ex_idx
		self.shared.batch_l = batch_l
		self.shared.sent_l1 = sent_l1
		self.shared.sent_l2 = sent_l2
		self.shared.res_map = res_map


	def begin_pass(self):
		self.encoder.begin_pass()
		self.attention.begin_pass()
		self.classifier.begin_pass()

	def end_pass(self):
		self.encoder.end_pass()
		self.attention.end_pass()
		self.classifier.end_pass()


	def get_param_dict(self):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		skipped_fields = []
		for n, p in self.named_parameters():
			# save all parameters that do not have skip_save flag
			# 	unlearnable parameters will also be saved
			if not hasattr(p, 'skip_save') or p.skip_save == 0:
				param_dict[n] =  torch2np(p.data, is_cuda)
			else:
				skipped_fields.append(n)
		#print('skipped fields:', skipped_fields)
		return param_dict

	def set_param_dict(self, param_dict):
		skipped_fields = []
		rec_fields = []
		for n, p in self.named_parameters():
			if n in param_dict:
				rec_fields.append(n)
				# load everything we have
				print('setting {0}'.format(n))
				p.data.copy_(torch.from_numpy(param_dict[n][:]))
			else:
				skipped_fields.append(n)
		print('skipped fileds: {0}'.format(skipped_fields))


