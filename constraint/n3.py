import sys
sys.path.insert(0, '../')

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from constr_utils import *


# If there exists a content word in the hypothesis not aligned with any content word, then the output should not be ENTAIL
class N3(torch.nn.Module):
	def __init__(self, opt, shared):
		super(N3, self).__init__()
		self.opt = opt
		self.shared = shared

		self.y_mask = Variable(torch.Tensor([1.0, 0.0, 0.0]).view(1,3), requires_grad=False)
		self.one = Variable(torch.ones(1), requires_grad=False)
		self.zero = Variable(torch.zeros(1), requires_grad=False)

		if opt.gpuid >= 0:
			self.one = self.one.cuda()
			self.zero = self.zero.cuda()
			self.y_mask = self.y_mask.cuda()

	# V_j a_ij <-> a_i
	# a connector layer
	# batch_aij is of shape (batch_l, sent_l2, sent_l1)
	def aij_to_ai(self, batch_aij):
		return torch.min(self.one, batch_aij.sum(-1))	# (batch_l, sent_l2)

	# ! /\_i a_i -> ! ent
	# ai is of shape (batch_l, sent_l2)
	# ai_mask is of shape (batch_l, sent_l2)
	# 	for a_i that was not selected (e.g. noncontent word), the distance function should ignore it
	def exists_no_ai(self, ai, ai_mask):
		# (batch_l, 1)
		return torch.min(self.one, (ai_mask * (self.one - ai)).sum(-1, keepdim=True))


	# att1 of shape (batch_l, sent_l2, sent_l1)
	# att2 of shape (batch_l, sent_l2, sent_l1) (the same shape)
	def forward(self, att1, att2):
		assert(self.opt.num_att_label == 1)

		# hypo content selector of shape (batch_l, sent_l2)
		h_mask = Variable(get_h_selector(self.opt, self.shared, 'content_word', with_nul=False),
			requires_grad=False)
		# prem content selector of shape (batch_l, sent_l1)
		p_mask = Variable(get_p_selector(self.opt, self.shared, 'content_word', with_nul=False), 
			requires_grad=False)
		# (batch_l, sent_l2, sent_l1)
		content_mask = h_mask.unsqueeze(-1) * p_mask.unsqueeze(1)

		a1_i = self.aij_to_ai(att1 * content_mask)
		a2_i = self.aij_to_ai(att2 * content_mask)

		z1 = self.exists_no_ai(a1_i, h_mask)
		z2 = self.exists_no_ai(a2_i, h_mask)

		# (batch_l, 1)
		d = torch.max(self.zero, z1 + z2 - self.one)

		# (batch_l, num_labels)
		dy = -d * self.y_mask.expand(self.shared.batch_l,3)
		return dy
