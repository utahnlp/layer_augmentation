import sys
import h5py
import torch
from torch import nn
from torch import cuda
from holder import *
from util import *
from torch.autograd import Variable


class CharEmbeddings(torch.nn.Module):
	def __init__(self, opt, shared):
		super(CharEmbeddings, self).__init__()
		self.opt = opt
		self.shared = shared

		#print('loading char dict from {0}'.format(opt.char_dict))
		#if opt.dict != '':
		#	self.vocab = load_dict(opt.dict)
		
		self.embeddings = nn.Embedding(opt.num_char, opt.char_emb_size)
		self.embeddings.weight.data[0,:] = torch.zeros(1, opt.char_emb_size).float()
		self.embeddings.weight.data[1:] = rand_tensor((opt.num_char-1, opt.char_emb_size), -0.1, 0.1).float()
		self.embeddings.weight.skip_init = 1
		self.embeddings.weight.requires_grad = True


	def forward(self, char_idx):
		batch_l, seq_l, token_l = char_idx.shape
		char_emb_size = self.opt.char_emb_size

		# flatten idx tensor
		# 	pytorch 0.3.1 does not support high-order indices (dim >2)
		char_idx = char_idx.contiguous().view(-1)
		char_emb = self.embeddings(char_idx)	# (batch_l *  seq_l * token_l, char_emb_size)
		char_emb = char_emb.view(batch_l, seq_l, token_l, char_emb_size)
		return char_emb

	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass

