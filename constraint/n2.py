import sys
sys.path.insert(0, '../')

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from constr_utils import *


# A content word in the hypothesis should align to a content word that is in relation set.
# Let R denote the set of content alignments appear in relation set (i.e. LHS)
# Let A denote the set of content desired alignments (i.e. RHS)
# R_i /\ A_i -> A_i'

# NOTE, if a content hypo word has no such related content premise word, then skip
class N2(torch.nn.Module):
	def __init__(self, opt, shared):
		super(N2, self).__init__()
		self.opt = opt
		self.shared = shared

		self.zero = Variable(torch.Tensor([0.0]), requires_grad=False)
		self.one = Variable(torch.Tensor([1.0]), requires_grad=False)
		if opt.gpuid >= 0:
			self.zero = self.zero.cuda()
			self.one = self.one.cuda()


	# run logic of N2
	#	assumes the input att is att2
	#	for logic on att1, simply call with att1.T, and transpose result after done
	def logic(self, att):
		# first get the content relation selectors
		h_content_selector = get_h_selector(self.opt, self.shared, 'content_word', with_nul=False).view(self.shared.batch_l, self.shared.sent_l2, 1)
		p_content_selector = get_p_selector(self.opt, self.shared, 'content_word', with_nul=False).view(self.shared.batch_l, 1, self.shared.sent_l1)
		# content word alignment selector
		C = h_content_selector * p_content_selector
		# related alignment selector
		R = get_rel_selector2(self.opt, self.shared, 'all_rel')
		# select content alignment
		R = Variable(C * R, requires_grad=False)
		if self.opt.gpuid != -1:
			R = R.cuda()

		d = torch.max(self.zero, att + R - self.one)

		return d

	def forward(self, att):
		assert(self.opt.num_att_label == 1)

		d = self.logic(att)

		return d