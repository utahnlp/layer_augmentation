from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import argparse
import copy
import sys

lemmatizer = WordNetLemmatizer()
DEBUG = False

def load_sent(data_file):
	sents = []
	with open(data_file, 'r+') as f:
		for l in f:
			if l.strip():
				sents.append(l.lower().split(' '))
	return sents


def load_cache(out_file, lemmatize = False):
	cache = {}
	with open(out_file, 'r+') as f:
		for l in f:
			toks = l.split("\t")
			if lemmatize:
				cache[toks[0]] = [lemmatizer.lemmatize(t.lower()) for t in toks[1:]]
			else:
				cache[toks[0]] = [t.lower() for t in toks[1:]]

	return cache

def load_table(in_file):
	cache = {}
	with open(in_file, 'r+') as f:
		ex_id = 0
		for l in f:
			toks = l.rstrip().split(' ')
			cache[ex_id] = toks
			ex_id += 1
	return cache


def merge_subgraphs(subgraphs):
	subgraphs_ = copy.deepcopy(subgraphs)
	merge_happened = True
	indep_flags = [True for _ in subgraphs]
	# merge
	while merge_happened:
		merge_happened = False

		for i in range(len(subgraphs_)):
			if len(subgraphs_[i][0]) == 0:
					continue

			for j in range(i+1, len(subgraphs_)):
				if len(subgraphs_[j][0]) == 0:
					continue

				s0, t0 = subgraphs_[i]
				s1, t1 = subgraphs_[j]

				# if has intersection, merge
				if len(set(s0).intersection(set(s1))) != 0 or len(set(t0).intersection(set(t1))) != 0:
					#print('merging {0} and {1}'.format(subgraphs_[i], subgraphs_[j]))
					subgraphs_[i] = (list(set(s0).union(set(s1))), list(set(t0).union(set(t1))))
					subgraphs_[j] = ([], [])
					merge_happened = True	# indicates there is a merge
					indep_flags[i] = False
					indep_flags[j] = False
					#print(subgraphs_)

	# after merging is done, add non-mergeable subgraphs
	subs = [sub for sub in subgraphs_ if len(sub[0]) > 0 and len(sub[1]) > 0]

	# get independent subgraphs
	indep_subs = [subgraphs[i] for i in range(len(subgraphs)) if indep_flags[i]]

	# get merged subgraphs
	merged_subs = [sub for i, sub in enumerate(subgraphs_) if len(sub[0]) > 0 and len(sub[1]) > 0 and not indep_flags[i]]

	return (subs, indep_subs, merged_subs)

def merge_this_with_that(left, rit):
	union = []
	union.extend(left)
	union.extend(rit)
	subgraphs, indep, merged = merge_subgraphs(union)

	left_indep = [l for l in left if l in indep]
	return (subgraphs, left_indep, merged)

def check_dict(res, l, r):
	r_in_l = l in res and r in res[l]
	l_in_r = r in res and l in res[r]
	return r_in_l or l_in_r

def overlap_check(left, rit):
	rs = [True for l in left for r in rit if len(set(l[0]).intersection(set(r[0]))) > 0 or len(set(l[1]).intersection(set(r[1]))) > 0]
	return len(rs) != 0

def process_syn(ex_id, src, targ, cache_dict, verbose=False):
	syn_subgraphs = {}	# subgraph keyed by token id of src sentence
	rel_subgraphs = {}

	syn = cache_dict['syn'] if 'syn' in cache_dict else {}
	ant = cache_dict['ant'] if 'ant' in cache_dict else {}
	rel = cache_dict['rel'] if 'rel' in cache_dict else {}
	isa = cache_dict['isa'] if 'isa' in cache_dict else {}

	for i, s in enumerate(src):
		for j, t in enumerate(targ):
			s_syn = syn[s] if s in syn else {}
			t_syn = syn[t] if t in syn else {}
			if s == t or check_dict(syn, s, t) or check_dict(isa, s, t):
				if i in syn_subgraphs:
					syn_subgraphs[i].append(j)
				else:
					syn_subgraphs[i] = [j]
			if check_dict(ant, s, t) or check_dict(rel, s, t):
				if i in rel_subgraphs:
					rel_subgraphs[i].append(j)
				else:
					rel_subgraphs[i] = [j]

	# change dict into list
	syn_subgraphs = [([k], v) for k, v in syn_subgraphs.items()]
	rel_subgraphs = [([k], v) for k, v in rel_subgraphs.items()]
	if verbose:
		print('syn_subgraphs: {0}'.format(syn_subgraphs))
		print('rel_subgraphs: {0}'.format(rel_subgraphs))

	# isa and syn will be classified as synilar
	# ant and rel will be taken as just related
	# the core syn subgraphs are the independent ones (i.e. non-mergeable ones)
	subs, indep, _ = merge_this_with_that(syn_subgraphs, rel_subgraphs)

	return indep

def process_excl_syn(ex_id, src, targ, cache_dict, verbose=False):
	syn_subgraphs = {}	# subgraph keyed by token id of src sentence
	rel_subgraphs = {}

	syn = cache_dict['syn'] if 'syn' in cache_dict else {}
	ant = cache_dict['ant'] if 'ant' in cache_dict else {}
	rel = cache_dict['rel'] if 'rel' in cache_dict else {}
	isa = cache_dict['isa'] if 'isa' in cache_dict else {}

	for i, s in enumerate(src):
		for j, t in enumerate(targ):
			s_syn = syn[s] if s in syn else {}
			t_syn = syn[t] if t in syn else {}
			if (s == t or check_dict(syn, s, t) or check_dict(isa, s, t)) and not check_dict(ant, s, t):
				if i in syn_subgraphs:
					syn_subgraphs[i].append(j)
				else:
					syn_subgraphs[i] = [j]

	# change dict into list
	syn_subgraphs = [([k], v) for k, v in syn_subgraphs.items()]
	if verbose:
		print('syn_subgraphs: {0}'.format(syn_subgraphs))

	subs, _, _ = merge_subgraphs(syn_subgraphs)

	return subs

def process_ant(ex_id, src, targ, cache_dict, verbose=False):
	ant_subgraphs = {}	# subgraph keyed by token id of src sentence
	rel_subgraphs = {}

	syn = cache_dict['syn'] if 'syn' in cache_dict else {}
	ant = cache_dict['ant'] if 'ant' in cache_dict else {}
	rel = cache_dict['rel'] if 'rel' in cache_dict else {}
	isa = cache_dict['isa'] if 'isa' in cache_dict else {}

	for i, s in enumerate(src):
		for j, t in enumerate(targ):
			s_syn = syn[s] if s in syn else {}
			t_syn = syn[t] if t in syn else {}
			if check_dict(ant, s, t):
				if i in ant_subgraphs:
					ant_subgraphs[i].append(j)
				else:
					ant_subgraphs[i] = [j]

			if s == t or check_dict(syn, s, t) or check_dict(isa, s, t) or check_dict(rel, s, t):
				if i in rel_subgraphs:
					rel_subgraphs[i].append(j)
				else:
					rel_subgraphs[i] = [j]

	# change dict into list
	ant_subgraphs = [([k], v) for k, v in ant_subgraphs.items()]
	rel_subgraphs = [([k], v) for k, v in rel_subgraphs.items()]
	if verbose:
		print('ant_subgraphs: {0}'.format(ant_subgraphs))
		print('rel_subgraphs: {0}'.format(rel_subgraphs))

	# ant will be classified as antonym
	# syn, isa and rel will be taken as just related
	# the core syn subgraphs are the independent ones (i.e. non-mergeable ones)
	subs, indep, _ = merge_this_with_that(ant_subgraphs, rel_subgraphs)

	return indep

def process_excl_ant(ex_id, src, targ, cache_dict, verbose=False):
	ant_subgraphs = {}	# subgraph keyed by token id of src sentence
	rel_subgraphs = {}

	syn = cache_dict['syn'] if 'syn' in cache_dict else {}
	ant = cache_dict['ant'] if 'ant' in cache_dict else {}
	rel = cache_dict['rel'] if 'rel' in cache_dict else {}
	isa = cache_dict['isa'] if 'isa' in cache_dict else {}

	for i, s in enumerate(src):
		for j, t in enumerate(targ):
			s_syn = syn[s] if s in syn else {}
			t_syn = syn[t] if t in syn else {}
			if check_dict(ant, s, t) and s != t and not check_dict(syn, s, t) and not check_dict(isa, s, t):
				if i in ant_subgraphs:
					ant_subgraphs[i].append(j)
				else:
					ant_subgraphs[i] = [j]

	# change dict into list
	ant_subgraphs = [([k], v) for k, v in ant_subgraphs.items()]
	if verbose:
		print('ant_subgraphs: {0}'.format(ant_subgraphs))

	subs, _, _ =  merge_subgraphs(ant_subgraphs)
	return subs

def process_all_ant(ex_id, src, targ, cache_dict, verbose=False):
	ant_subgraphs = {}	# subgraph keyed by token id of src sentence
	rel_subgraphs = {}

	syn = cache_dict['syn'] if 'syn' in cache_dict else {}
	ant = cache_dict['ant'] if 'ant' in cache_dict else {}
	rel = cache_dict['rel'] if 'rel' in cache_dict else {}
	isa = cache_dict['isa'] if 'isa' in cache_dict else {}

	for i, s in enumerate(src):
		for j, t in enumerate(targ):
			s_syn = syn[s] if s in syn else {}
			t_syn = syn[t] if t in syn else {}
			if check_dict(ant, s, t):
				if i in ant_subgraphs:
					ant_subgraphs[i].append(j)
				else:
					ant_subgraphs[i] = [j]

	# change dict into list
	ant_subgraphs = [([k], v) for k, v in ant_subgraphs.items()]
	if verbose:
		print('ant_subgraphs: {0}'.format(ant_subgraphs))

	subs, _, _ =  merge_subgraphs(ant_subgraphs)
	return subs


def process_rel(ex_id, src, targ, cache_dict, verbose=False):
	syn_subgraphs = {}	# subgraph keyed by token id of src sentence
	ant_subgraphs = {}
	rel_subgraphs = {}

	syn = cache_dict['syn'] if 'syn' in cache_dict else {}
	ant = cache_dict['ant'] if 'ant' in cache_dict else {}
	rel = cache_dict['rel'] if 'rel' in cache_dict else {}
	isa = cache_dict['isa'] if 'isa' in cache_dict else {}

	for i, s in enumerate(src):
		for j, t in enumerate(targ):
			s_syn = syn[s] if s in syn else {}
			t_syn = syn[t] if t in syn else {}
			if s == t or check_dict(syn, s, t) or check_dict(isa, s, t):
				if i in syn_subgraphs:
					syn_subgraphs[i].append(j)
				else:
					syn_subgraphs[i] = [j]
			if check_dict(ant, s, t):
				if i in ant_subgraphs:
					ant_subgraphs[i].append(j)
				else:
					ant_subgraphs[i] = [j]
			if check_dict(rel, s, t):
				if i in rel_subgraphs:
					rel_subgraphs[i].append(j)
				else:
					rel_subgraphs[i] = [j]

	# change dict into list
	syn_subgraphs = [([k], v) for k, v in syn_subgraphs.items()]
	ant_subgraphs = [([k], v) for k, v in ant_subgraphs.items()]
	rel_subgraphs = [([k], v) for k, v in rel_subgraphs.items()]

	if verbose:
		print('syn_subgraphs: {0}'.format(syn_subgraphs))
		print('ant_subgraphs: {0}'.format(ant_subgraphs))
		print('rel_subgraphs: {0}'.format(rel_subgraphs))

	# the core rel subgraphs are those independent rel subgraphs and merged syn and ant subgraphs
	so_subs, syn_indep, so_merged = merge_this_with_that(syn_subgraphs, ant_subgraphs)
	_, ant_indep, _ = merge_this_with_that(ant_subgraphs, syn_subgraphs)
	rel_subs, rel_indep, rel_merged = merge_this_with_that(rel_subgraphs, so_subs)

	rel_merged.extend(so_merged)
	rel_merged.extend(rel_indep)

	# filter by exclusion
	#	ONLY kill those identical subgraphs
	rel_merged = [sub for sub in rel_merged if sub not in syn_indep and sub not in ant_indep]

	rel_merged.extend(so_merged)
	rel_merged.extend(rel_indep)
	rel_merged, _, _ = merge_subgraphs(rel_merged)

	return rel_merged

def process_all_rel(ex_id, src, targ, cache_dict, verbose=False):
	rel_subgraphs = {}

	syn = cache_dict['syn'] if 'syn' in cache_dict else {}
	ant = cache_dict['ant'] if 'ant' in cache_dict else {}
	rel = cache_dict['rel'] if 'rel' in cache_dict else {}
	isa = cache_dict['isa'] if 'isa' in cache_dict else {}

	for i, s in enumerate(src):
		for j, t in enumerate(targ):
			s_syn = syn[s] if s in syn else {}
			t_syn = syn[t] if t in syn else {}
			if s == t or check_dict(syn, s, t) or check_dict(isa, s, t) or check_dict(ant, s, t) or check_dict(rel, s, t):
				if i in rel_subgraphs:
					rel_subgraphs[i].append(j)
				else:
					rel_subgraphs[i] = [j]

	# change dict into list
	rel_subgraphs = [([k], v) for k, v in rel_subgraphs.items()]
	if verbose:
		print('rel_subgraphs: {0}'.format(rel_subgraphs))

	ensemble, indep, _ = merge_subgraphs(rel_subgraphs)

	return ensemble
	

def wn_synonyms(lemma):
	synonyms = wn.synsets(lemma)
	return list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))

def process_conj(ex_id, src, targ, cache_dict, verbose=False):
	rel_subgraphs = {}
	src_pos = cache_dict['src_pos'][ex_id]
	targ_pos = cache_dict['targ_pos'][ex_id]

	src_sub = []
	targ_sub = []
	for i, s in enumerate(src_pos):
		if s == 'PUNCT':
			src_sub.append(i)

	for j, t in enumerate(targ_pos):
		if t == 'PUNCT':
			targ_sub.append(j)

	sub = [(src_sub, targ_sub)]

	if verbose:
		print('pos_subgraphs: {0}'.format(sub))

	return sub

def process_det_num(ex_id, src, targ, cache_dict, verbose=False):
	rel_subgraphs = {}
	src_pos = cache_dict['src_pos'][ex_id]
	targ_pos = cache_dict['targ_pos'][ex_id]

	src_sub = []
	targ_sub = []
	for i, s in enumerate(src_pos):
		if s == 'DET' or s == 'NUM':
			src_sub.append(i)

	for j, t in enumerate(targ_pos):
		if t == 'DET' or t == 'NUM':
			targ_sub.append(j)

	sub = [(src_sub, targ_sub)]

	if verbose:
		print('pos_subgraphs: {0}'.format(sub))

	return sub


def process_content_word(ex_id, src, targ, cache_dict, verbose=False):
	rel_subgraphs = {}
	src_pos = cache_dict['src_pos'][ex_id]
	targ_pos = cache_dict['targ_pos'][ex_id]
	content_pos = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'NUM']

	src_sub = []
	targ_sub = []
	for i, s in enumerate(src_pos):
		if s in content_pos:
			src_sub.append(i)

	for j, t in enumerate(targ_pos):
		if t in content_pos:
			targ_sub.append(j)

	sub = [(src_sub, targ_sub)]

	if verbose:
		print('pos_subgraphs: {0}'.format(sub))

	return sub

def process(opt, verbose=False):
	rel = opt.output_rel
	src = load_sent(opt.src)
	targ = load_sent(opt.targ)
	cache = {}

	lemmatize = False
	if opt.syn != '':
		cache['syn'] = load_cache(opt.syn, lemmatize) 
	if opt.ant != '':
		cache['ant'] = load_cache(opt.ant, lemmatize)
	if opt.rel != '':
		cache['rel'] = load_cache(opt.rel, lemmatize)
	if opt.isa != '':
		cache['isa'] = load_cache(opt.isa, lemmatize)

	if opt.src_pos != '':
		cache['src_pos'] = load_table(opt.src_pos)
	if opt.targ_pos != '':
		cache['targ_pos'] = load_table(opt.targ_pos)


	subgraphs = []
	typ = 'map'
	cnt = 0
	for s, t in zip(src, targ):
		print('processing {0} for {1}'.format(rel, cnt))
		subs = []
		if rel == 'syn':
			subs = process_syn(cnt, s, t, cache, verbose)
		elif rel == 'ant':
			subs = process_ant(cnt, s, t, cache, verbose)
		elif rel == 'rel':
			subs = process_rel(cnt, s, t, cache, verbose)
		elif rel == 'all_rel':
			subs = process_all_rel(cnt, s, t, cache, verbose)
		elif rel == 'excl_syn':
			subs = process_excl_syn(cnt, s, t, cache, verbose)
		elif rel == 'excl_ant':	# remove pairs in syn
			subs = process_excl_ant(cnt, s, t, cache, verbose)
		elif rel == 'all_ant':	# remove pairs in syn
			subs = process_all_ant(cnt, s, t, cache, verbose)
		elif rel == 'conj':
			typ = 'list'
			subs = process_conj(cnt, s, t, cache, verbose)
		elif rel == 'det_num':
			typ = 'list'
			subs = process_det_num(cnt, s, t, cache, verbose)
		elif rel == 'content_word':
			typ = 'list'
			subs = process_content_word(cnt, s, t, cache, verbose)
		else:
			print('unrecognized rel {0}'.format(rel))
			assert(False)
		subgraphs.append(subs)
		cnt += 1

	print('converting subgraphs into json string')

	rs = to_json_map(subgraphs, rel) if typ == 'map' else to_json_list(subgraphs, rel)
	output_path = opt.output + '.' + rel + '.json'
	with open(output_path, 'w+') as f:
		f.write(rs)


def to_json_map(subs, name):
	rs = '{ \"type\":\"map\", \"' + name + '\":{\n'
	for ex, sub in enumerate(subs):
		rs += '\"' + str(ex) + '\":{\"' + name + '\":'
		# get keys
		keys = [i for i,_ in enumerate(sub)]
		rs += '[' + ','.join('\"{0}\"'.format(k) for k in keys) + ']'
		if len(keys) > 0:
			rs += ','
			for j, p in enumerate(sub):
				k = j
				rs += '\"{0}\":[{1},{2}]'.format(k, str(p[0]).replace(' ', ''), str(p[1]).replace(' ', ''))
				if j != len(sub)-1:
					rs += ','
		rs += '}'
		if ex != len(subs) - 1:
			rs += ','
		rs += '\n'
	rs += '}}'
	return rs

def to_json_list(subs, name):
	rs = '{ \"type\": \"list", \"' + name + '\":{\n'
	for ex, sub in enumerate(subs):
		p = sub[0][0]
		h = sub[0][1]
		rs += '\"' + str(ex) + '\":{\"p\":' + str(p).replace(' ', '') + ',\"h\":' + str(h).replace(' ', '') +'}'
		if ex != len(subs) - 1:
			rs += ','
		rs += '\n'
	rs += '}}'
	return rs

def main(args):
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--dir', help="Path to the data dir.", default = "./data/snli_1.0/")
	parser.add_argument('--src', help="Path to sent1 lemma data.", default = "dev.sent1_lemma.txt")
	parser.add_argument('--targ', help="Path to sent2 lemma data.", default = "dev.sent2_lemma.txt")
	parser.add_argument('--src_pos', help='Path to src pos data.', default='dev.sent1_pos.txt')
	parser.add_argument('--targ_pos', help='Path to targ pos data.', default='dev.sent2_pos.txt')
	parser.add_argument('--syn', help="Path to synonym cache", default = "conceptnet.syn.txt")
	parser.add_argument('--ant', help="Path to antonym cache", default = "conceptnet.distinct.txt")
	parser.add_argument('--rel', help="Path to related cache", default = "conceptnet.related.txt")
	parser.add_argument('--isa', help="Path to isa cache", default = "conceptnet.isa.txt")
	parser.add_argument('--output_rel', help='The type of relation to output', default='all_rel')
	parser.add_argument('--output', help="Prefix of the output file", default = "dev")
	opt = parser.parse_args(args)

	opt.src = opt.dir + opt.src
	opt.targ = opt.dir + opt.targ
	opt.src_pos = opt.dir + opt.src_pos
	opt.targ_pos = opt.dir + opt.targ_pos
	opt.output = opt.dir + opt.output
	opt.syn = opt.dir + opt.syn
	opt.ant = opt.dir + opt.ant
	opt.rel = opt.dir + opt.rel
	opt.isa = opt.dir + opt.isa


	process(opt, verbose=False)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

	#subgraphs = [([0], [1,2]), ([1], [5]), ([3], [3,4]), ([4], [1,4]), ([5], [7])]
	#merged, indep, _ = merge_subgraphs(subgraphs)
	##print(merged)
	##print(indep)
#
#
	#src = [0,1,2,3,4,5,9,10] 
	#targ = [0,1,2,3,4,5,6,7,8,11]
	#syn = {0: [0], 1: [1, 5]}
	#ant = {9: [8]}
	#rel = {5: [7], 7: [2], 10: [11]}
	#cache_dict = {'syn': syn, 'ant': ant, 'rel': rel}
	##print('syn: {}'.format(process_syn(src, targ, cache_dict, verbose=True)))
	##print('ant: {}'.format(process_ant(src, targ, cache_dict, verbose=True)))
	#print('rel: {}'.format(process_rel(src, targ, cache_dict, verbose=True)))
	##print('all rel: {}'.format(process_all_rel(src, targ, cache_dict, verbose=True)))#
