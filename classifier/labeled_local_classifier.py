import sys
sys.path.insert(0, '../')

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

class LabeledLocalClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(LabeledLocalClassifier, self).__init__()

		# temp stuff will be changed on the fly
		batch_l = 1
		sent_l1 = 2
		sent_l2 = 3

		labeled_cat_size = opt.hidden_size * 2 * opt.num_att_labels
		self.input_view1 = View(batch_l * sent_l1, labeled_cat_size)
		self.input_view2 = View(batch_l * sent_l2, labeled_cat_size)
		self.input_unview1 = View(batch_l, sent_l1, opt.hidden_size)
		self.input_unview2 = View(batch_l, sent_l2, opt.hidden_size)
		self.input_joiner = JoinTable(2)
		self.phi_joiner = JoinTable(1)
		self.sl = StructLayer(opt, shared)

		# bookkeeping
		self.opt = opt
		self.shared = shared
		self.dropout = opt.dropout
		self.hidden_size = opt.hidden_size
		self.num_att_labels = opt.num_att_labels

		# NOTE, this part is different from structatt-torch branch
		# the performance is on the same bar, but this requires fewer parameters
		self.g = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(labeled_cat_size, opt.hidden_size),
			nn.ReLU(),
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU())

		cat_size = opt.hidden_size * 2
		self.h = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(cat_size, opt.hidden_size),
			nn.ReLU(),
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU(),
			nn.Linear(opt.hidden_size, opt.num_labels))
		self.log_softmax = nn.LogSoftmax(1)


	def get_param_dict(self, root):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		for i in [1,4]:
			param_dict['{0}.g[{1}].weight'.format(root, i)] = torch2np(self.g[i].weight.data, is_cuda)
			if self.g[i].bias is not None:
				param_dict['{0}.g[{1}].bias'.format(root, i)] = torch2np(self.g[i].bias.data, is_cuda)

		for i in [1,4,6]:
			param_dict['{0}.h[{1}].weight'.format(root, i)] = torch2np(self.h[i].weight.data, is_cuda)
			if self.h[i].bias is not None:
				param_dict['{0}.h[{1}].bias'.format(root, i)] = torch2np(self.h[i].bias.data, is_cuda)

		return param_dict

	def set_param_dict(self, param_dict, root):
		for i in [1,4]:
			self.g[i].weight.data.copy_(torch.from_numpy(param_dict['{0}.g[{1}].weight'.format(root, i)][:]))
			if self.g[i].bias is not None:
				self.g[i].bias.data.copy_(torch.from_numpy(param_dict['{0}.g[{1}].bias'.format(root, i)][:]))

		for i in [1,4,6]:
			self.h[i].weight.data.copy_(torch.from_numpy(param_dict['{0}.h[{1}].weight'.format(root, i)][:]))
			if self.h[i].bias is not None:
				self.h[i].bias.data.copy_(torch.from_numpy(param_dict['{0}.h[{1}].bias'.format(root, i)][:]))



	def forward(self, sent1, sent2, att1, att2):
		self.update_context()

		labeled_input1 = []
		labeled_input2 = []
		for l in xrange(self.num_att_labels):
			att_l1 = att1[:, :, l*self.shared.sent_l2:(l+1)*self.shared.sent_l2]
			att_l2 = att2[:, :, l*self.shared.sent_l1:(l+1)*self.shared.sent_l1]

			attended2 = att_l1.bmm(sent2)
			attended1 = att_l2.bmm(sent1)

			labeled_input1.extend([sent1, attended2])
			labeled_input2.extend([sent2, attended1])

		cat1 = self.input_joiner(labeled_input1)
		cat2 = self.input_joiner(labeled_input2)

		phi1 = self.input_unview1(self.g(self.input_view1(cat1)))
		phi2 = self.input_unview2(self.g(self.input_view2(cat2)))

		flat_phi1 = phi1.sum(1)
		flat_phi2 = phi2.sum(1)

		phi = self.phi_joiner([flat_phi1, flat_phi2])
		y_scores = self.h(phi)

		# if struct layer is enabled
		if self.opt.constr_layers != '':
			y_scores = self.sl(y_scores, att1, att2)

		self.shared.out = self.log_softmax(y_scores)

		return self.shared.out

	def update_context(self):
		batch_l = self.shared.batch_l
		sent_l1 = self.shared.sent_l1
		sent_l2 = self.shared.sent_l2
		num_att_labels = self.num_att_labels
		labeled_cat_size = self.hidden_size * 2 * num_att_labels

		self.input_view1.dims = (batch_l * sent_l1, labeled_cat_size)
		self.input_view2.dims = (batch_l * sent_l2, labeled_cat_size)
		self.input_unview1.dims = (batch_l, sent_l1, self.hidden_size)
		self.input_unview2.dims = (batch_l, sent_l2, self.hidden_size)

	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	sys.path.insert(0, '../attention/')
	from torch.autograd import Variable
	from labeled_local_attention import *

	opt = Holder()
	opt.hidden_size = 3
	opt.dropout = 0.0
	opt.num_labels = 3
	opt.num_att_labels = 3
	shared = Holder()
	shared.batch_l = 2
	shared.sent_l1 = 5
	shared.sent_l2 = 8
	shared.input1 = Variable(torch.randn(shared.batch_l, shared.sent_l1, opt.hidden_size), True)
	shared.input2 = Variable(torch.randn(shared.batch_l, shared.sent_l2, opt.hidden_size), True)

	# build network
	attender = LabeledLocalAttention(opt, shared)
	classifier = LabeledLocalClassifier(opt, shared)

	# update batch info
	shared.batch_l = 2
	shared.sent_l1 = 5
	shared.sent_l2 = 8

	# run network
	shared.att1, shared.att2 = attender(shared.input1, shared.input2)
	shared.out = classifier(shared.input1, shared.input2, shared.att1, shared.att2)

	print(shared.att1)
	print(shared.att1.sum(2))
	print(shared.att2)
	print(shared.att2.sum(2))
	print(shared.out)
	print(classifier)
	#print(classifier.g[1].weight)
	#print(classifier.g[1].bias)
	#classifier.apply(classifier.weights)
#
	#for i, p in enumerate(classifier.parameters()):
	#	print(p.data)
	#	print(p.grad)
	#