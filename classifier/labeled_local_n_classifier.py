import sys
sys.path.insert(0, '../')

import torch
from torch import nn
from holder import *
from util import *

class LabeledLocalNClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(LabeledLocalNClassifier, self).__init__()

		# temp stuff will be changed on the fly
		batch_l = 1
		sent_l1 = 2
		sent_l2 = 3

		# bookkeeping
		self.opt = opt
		self.shared = shared
		self.dropout = opt.dropout
		self.hidden_size = opt.hidden_size
		self.num_att_labels = opt.num_att_labels

		labeled_cat_size = opt.hidden_size * 2
		self.input_view1 = View(batch_l * sent_l1, labeled_cat_size)
		self.input_view2 = View(batch_l * sent_l2, labeled_cat_size)
		self.input_unview1 = View(batch_l, sent_l1, opt.hidden_size)
		self.input_unview2 = View(batch_l, sent_l2, opt.hidden_size)
		self.input_joiner = JoinTable(2)
		self.phi_joiner = JoinTable(1)
		self.sl = StructLayer(opt, shared)

		for l in xrange(self.num_att_labels):
			g_l = nn.Sequential(
				nn.Dropout(opt.dropout),
				nn.Linear(labeled_cat_size, opt.hidden_size),
				nn.ReLU(),
				nn.Dropout(opt.dropout),
				nn.Linear(opt.hidden_size, opt.hidden_size),
				nn.ReLU())
			setattr(self, 'g_l_{0}'.format(l), g_l)

		cat_size = opt.hidden_size * 2 * self.num_att_labels
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
		for l in xrange(self.num_att_labels):
			g_l = getattr(self, 'g_l_{0}'.format(l))
			for i in [1,4]:
				param_dict['{0}.g_l_{1}[{2}].weight'.format(root, l, i)] = torch2np(g_l[i].weight.data, is_cuda)
				if g_l[i].bias is not None:
					param_dict['{0}.g_l_{1}[{2}].bias'.format(root, l, i)] = torch2np(g_l[i].bias.data, is_cuda)

		for i in [1,4,6]:
			param_dict['{0}.h[{1}].weight'.format(root, i)] = torch2np(self.h[i].weight.data, is_cuda)
			if self.h[i].bias is not None:
				param_dict['{0}.h[{1}].bias'.format(root, i)] = torch2np(self.h[i].bias.data, is_cuda)
		return param_dict

	def set_param_dict(self, param_dict, root):
		for l in xrange(self.num_att_labels):
			g_l = getattr(self, 'g_l_{0}'.format(l))
			for i in [1,4]:
				g_l[i].weight.data.copy_(torch.from_numpy(param_dict['{0}.g_l_{1}[{2}].weight'.format(root, l, i)][:]))
				if g_l[i].bias is not None:
					g_l[i].bias.data.copy_(torch.from_numpy(param_dict['{0}.g_l_{1}[{2}].bias'.format(root, l, i)][:]))

		for i in [1,4,6]:
			self.h[i].weight.data.copy_(torch.from_numpy(param_dict['{0}.h[{1}].weight'.format(root, i)][:]))
			if self.h[i].bias is not None:
				self.h[i].bias.data.copy_(torch.from_numpy(param_dict['{0}.h[{1}].bias'.format(root, i)][:]))



	def forward(self, sent1, sent2, att1, att2):
		self.update_context()

		labeled_phi = []
		for l in xrange(self.num_att_labels):
			g_l = getattr(self, 'g_l_{0}'.format(l))

			att_l1 = att1[:, :, l*self.shared.sent_l2:(l+1)*self.shared.sent_l2]
			att_l2 = att2[:, :, l*self.shared.sent_l1:(l+1)*self.shared.sent_l1]

			attended2 = att_l1.bmm(sent2)
			attended1 = att_l2.bmm(sent1)

			input1 = self.input_joiner([sent1, attended2])
			input2 = self.input_joiner([sent2, attended1])

			phi1 = self.input_unview1(g_l(self.input_view1(input1)))
			phi2 = self.input_unview2(g_l(self.input_view2(input2)))

			labeled_phi.append(phi1.sum(1))
			labeled_phi.append(phi2.sum(1))

		phi = self.phi_joiner(labeled_phi)
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
		labeled_cat_size = self.hidden_size * 2

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
	attender = LabeledLocalNClassifier(opt, shared)
	classifier = LabeledLocalNClassifier(opt, shared)

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