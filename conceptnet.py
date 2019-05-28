# modified from https://github.com/fitosegrera/python-conceptnet/blob/master/ConceptNet.py
# modifier: Tao Li
import ujson as json
from urllib.request import urlopen
import argparse
import sys
#import langdetect
from multiprocessing import Pool
import enchant

en_d = enchant.Dict('en_US')

def open_url(url):
	return urlopen(url)

class ConceptNet:

	def __init__(self):
		#self.url = "http://api.conceptnet.io/"
		self.url = 'http://localhost:8084/'

	def synonym_to(self, concept, verbose=False):
		url_to_search = self.url + "search?rel=/r/" + "Synonym" +"&end=/c/en/" + concept
		data = open_url(url_to_search)
		json_data = json.loads(data.read())
		if verbose:
			print(url_to_search)
			for i in json_data["edges"]:
				print("----------------")
				print(i["end"])
				print(i["start"])
				print("weight:", i["weight"])

		rs = [i["start"]["label"] for i in json_data["edges"]]
		rs = [i for i in rs if en_d.check(i)]	# only take english words
		##rs = [i for i in rs if langdetect.detect(i) == 'en']	# only take english words
		rs = [i.encode("ascii","ignore") for i in rs]	# translate into ascii whatsoever
		rs = [i for i in rs if len(i) > 0]	# filter out unicode-to-ascii failure (empty str)
		data.close()
		return list(set(rs))

	def synonym_from(self, concept, verbose=False):
		url_to_search = self.url + "search?rel=/r/" + "Synonym" +"&start=/c/en/" + concept
		data = open_url(url_to_search)
		json_data = json.loads(data.read())
		if verbose:
			print(url_to_search)
			for i in json_data["edges"]:
				print("----------------")
				print(i["end"])
				print(i["start"])
				print("weight:", i["weight"])

		rs = [i["end"]["label"] for i in json_data["edges"]]
		rs = [i for i in rs if en_d.check(i)]	# only take english words
		##rs = [i for i in rs if langdetect.detect(i) == 'en']	# only take english words
		rs = [i.encode("ascii","ignore") for i in rs]	# translate into ascii whatsoever
		rs = [i for i in rs if len(i) > 0]	# filter out unicode-to-ascii failure (empty str)
		data.close()
		return list(set(rs))

	def synonym(self, concept, verbose=False):
		combined = self.synonym_from(concept, verbose) + self.synonym_to(concept, verbose)
		return list(set(combined))

	def antonym_to(self, concept, verbose=False):
		url_to_search = self.url + "search?rel=/r/" + "Antonym" +"&end=/c/en/" + concept
		data = open_url(url_to_search)
		json_data = json.loads(data.read())
		if verbose:
			print(url_to_search)
			for i in json_data["edges"]:
				print("----------------")
				print(i["end"])
				print(i["start"])
				print("weight:", i["weight"])

		rs = [i["start"]["label"] for i in json_data["edges"]]
		##rs = [i for i in rs if langdetect.detect(i) == 'en']	# only take english words
		rs = [i.encode("ascii","ignore") for i in rs]	# translate into ascii whatsoever
		rs = [i for i in rs if len(i) > 0]	# filter out unicode-to-ascii failure (empty str)
		data.close()
		return list(set(rs))

	def antonym_from(self, concept, verbose=False):
		url_to_search = self.url + "search?rel=/r/" + "Antonym" +"&start=/c/en/" + concept
		data = open_url(url_to_search)
		json_data = json.loads(data.read())
		if verbose:
			print(url_to_search)
			for i in json_data["edges"]:
				print("----------------")
				print(i["end"])
				print(i["start"])
				print("weight:", i["weight"])

		rs = [i["end"]["label"] for i in json_data["edges"]]
		##rs = [i for i in rs if langdetect.detect(i) == 'en']	# only take english words
		rs = [i.encode("ascii","ignore") for i in rs]	# translate into ascii whatsoever
		rs = [i for i in rs if len(i) > 0]	# filter out unicode-to-ascii failure (empty str)
		data.close()
		return list(set(rs))

	def antonym(self, concept, verbose=False):
		combined = self.antonym_from(concept, verbose) + self.antonym_to(concept, verbose)
		return list(set(combined))

	def distinct_from(self, concept, verbose=False):
		url_to_search = self.url + "search?rel=/r/" + "DistinctFrom" +"&start=/c/en/" + concept
		data = open_url(url_to_search)
		json_data = json.loads(data.read())
		if verbose:
			print(url_to_search)
			for i in json_data["edges"]:
				print("----------------")
				print(i["end"])
				print(i["start"])
				print("weight:", i["weight"])

		rs = [i["end"]["label"] for i in json_data["edges"]]
		##rs = [i for i in rs if langdetect.detect(i) == 'en']	# only take english words
		rs = [i.encode("ascii","ignore") for i in rs]	# translate into ascii whatsoever
		rs = [i for i in rs if len(i) > 0]	# filter out unicode-to-ascii failure (empty str)
		data.close()
		return list(set(rs))

	def distinct_to(self, concept, verbose=False):
		url_to_search = self.url + "search?rel=/r/" + "DistinctFrom" +"&end=/c/en/" + concept
		data = open_url(url_to_search)
		json_data = json.loads(data.read())
		if verbose:
			print(url_to_search)
			for i in json_data["edges"]:
				print("----------------")
				print(i["end"])
				print(i["start"])
				print("weight:", i["weight"])

		rs = [i["start"]["label"] for i in json_data["edges"]]
		##rs = [i for i in rs if langdetect.detect(i) == 'en']	# only take english words
		rs = [i.encode("ascii","ignore") for i in rs]	# translate into ascii whatsoever
		rs = [i for i in rs if len(i) > 0]	# filter out unicode-to-ascii failure (empty str)
		data.close()
		return list(set(rs))

	def distinct(self, concept, verbose=False):
		combined = self.distinct_from(concept, verbose) + self.distinct_to(concept, verbose)
		return list(set(combined))

	def form_of(self, concept, verbose=False):
		url_to_search = self.url + "search?rel=/r/" + "FormOf" +"&start=/c/en/" + concept
		data = open_url(url_to_search)
		json_data = json.loads(data.read())
		if verbose:
			print(url_to_search)
			for i in json_data["edges"]:
				print("----------------")
				print(i["end"])
				print(i["start"])
				print("weight:", i["weight"])

		rs = [i["end"]["label"] for i in json_data["edges"]]
		##rs = [i for i in rs if langdetect.detect(i) == 'en']	# only take english words
		rs = [i.encode("ascii","ignore") for i in rs]	# translate into ascii whatsoever
		rs = [i for i in rs if len(i) > 0]	# filter out unicode-to-ascii failure (empty str)
		data.close()
		return list(set(rs))

	def related_from(self, concept, verbose=False):
		url_to_search = self.url + "search?rel=/r/" + "RelatedTo" +"&start=/c/en/" + concept
		data = open_url(url_to_search)
		json_data = json.loads(data.read())
		if verbose:
			print(url_to_search)
			for i in json_data["edges"]:
				print("----------------")
				print(i["end"])
				print(i["start"])
				print("weight:", i["weight"])

		rs = [i["end"]["label"] for i in json_data["edges"]]
		##rs = [i for i in rs if langdetect.detect(i) == 'en']	# only take english words
		rs = [i.encode("ascii","ignore") for i in rs]	# translate into ascii whatsoever
		rs = [i for i in rs if len(i) > 0]	# filter out unicode-to-ascii failure (empty str)
		data.close()
		return list(set(rs))

	def related_to(self, concept, verbose=False):
		url_to_search = self.url + "search?rel=/r/" + "RelatedTo" +"&end=/c/en/" + concept
		data = open_url(url_to_search)
		json_data = json.loads(data.read())
		if verbose:
			print(url_to_search)
			for i in json_data["edges"]:
				print("----------------")
				print(i["end"])
				print(i["start"])
				print("weight:", i["weight"])

		rs = [i["start"]["label"] for i in json_data["edges"]]
		##rs = [i for i in rs if langdetect.detect(i) == 'en']	# only take english words
		rs = [i.encode("ascii","ignore") for i in rs]	# translate into ascii whatsoever
		rs = [i for i in rs if len(i) > 0]	# filter out unicode-to-ascii failure (empty str)
		data.close()
		return list(set(rs))

	def related(self, concept, verbose=False):
		combined = self.related_from(concept, verbose) + self.related_to(concept, verbose)
		return list(set(combined))

	def isa_from(self, concept, verbose=False):
		url_to_search = self.url + "search?rel=/r/" + "IsA" +"&start=/c/en/" + concept
		data = open_url(url_to_search)
		json_data = json.loads(data.read())
		if verbose:
			print(url_to_search)
			for i in json_data["edges"]:
				prin("----------------")
				prin(i["end"])
				prin(i["start"])
				prin("weight:", i["weight"])

		rs = [i["end"]["label"] for i in json_data["edges"]]
		##rs = [i for i in rs if langdetect.detect(i) == 'en']	# only take english words
		rs = [i.encode("ascii","ignore") for i in rs]	# translate into ascii whatsoever
		rs = [i for i in rs if len(i) > 0]	# filter out unicode-to-ascii failure (empty str)
		data.close()
		return list(set(rs))

	def isa_to(self, concept, verbose=False):
		url_to_search = self.url + "search?rel=/r/" + "IsA" +"&end=/c/en/" + concept
		data = open_url(url_to_search)
		json_data = json.loads(data.read())
		if verbose:
			print(url_to_search)
			for i in json_data["edges"]:
				print("----------------")
				print(i["end"])
				print(i["start"])
				print("weight:", i["weight"])

		rs = [i["start"]["label"] for i in json_data["edges"]]
		##rs = [i for i in rs if langdetect.detect(i) == 'en']	# only take english words
		rs = [i.encode("ascii","ignore") for i in rs]	# translate into ascii whatsoever
		rs = [i for i in rs if len(i) > 0]	# filter out unicode-to-ascii failure (empty str)
		data.close()
		return list(set(rs))

	def isa(self, concept, verbose=False):
		combined = self.isa_from(concept, verbose) + self.isa_to(concept, verbose)
		return list(set(combined))

def load_lemma(sent1_lemma, sent2_lemma):
	lemma_map = {}
	for _, (sent1, sent2) in enumerate(zip(open(sent1_lemma,'r'), open(sent2_lemma,'r'))):
		toks1 = sent1.rstrip().split(' ')
		toks2 = sent2.rstrip().split(' ')
		for t in toks1:
			lemma_map[t] = None
		for t in toks2:
			lemma_map[t] = None

	lemma = [l for l,_ in lemma_map.items()]
	return lemma


def load_output(out_file):
	prev_lemma = []
	with open(out_file, 'r+') as f:
		for l in f:
			toks = l.split("\t")
			key = toks[0].strip()
			prev_lemma.append(key)
	return prev_lemma


def worker(p):
	l = p[0]
	rel = p[1]
	cn = ConceptNet()

	if rel == 'syn':
		rs = cn.synonym(l)
	elif rel == 'ant':
		rs = cn.antonym(l)
	elif rel == 'distinct':
		rs = cn.distinct(l)
	elif rel == 'form_of':
		rs = cn.form_of(l)
	elif rel == 'related':
		rs = cn.related(l)
	elif rel == 'isa':
		rs = cn.isa(l)
	else:
		print('unrecognized rel: {0}'.format(rel))
		assert(False)
	return rs


def process(opt):
	out_file = opt.output
	rel = opt.rel
	n_workers = opt.worker

	lemma = load_lemma(opt.sent1_lemma, opt.sent2_lemma)
	print('{0} lemma loaded'.format(len(lemma)))

	#
	prev_lemma = []
	if opt.continu == 1:
		prev_lemma = load_output(opt.output)
		print('{0} cached lemma loaded'.format(len(prev_lemma)))

	# touch file
	if opt.continu == 0:
		with open(out_file, 'w+') as f:
			pass

	# get lemma that have not yet cached
	lemma_to_cache = [l for l in lemma if l not in prev_lemma]
	print('{0} lemma to cache'.format(len(lemma_to_cache)))

	# worker process
	batch_size = n_workers
	batches = []
	for i in range(0, len(lemma_to_cache), batch_size):
		batches.append(lemma_to_cache[i:i+batch_size])

	cnt = 0
	for batch in batches:
		params = [(l, rel) for l in batch]
		print('processing {0}'.format(batch))

		pool = Pool(batch_size)
		results = pool.map(worker, params)

		pool.close()
		pool.join()

		with open(out_file, 'a+') as f:
			for i, rs in enumerate(results):
				if len(rs) == 0:
					f.write("{0}\n".format(batch[i]))
					continue

				print("{0} found for {1}: {2}".format(rel, batch[i], rs))
				rs_list = [batch[i]]
				rs_list.extend(list(set(rs)))
				rs_list = [t if isinstance(t, str) else t.decode('utf-8') for t in rs_list]
				f.write("{0}\n".format("\t".join(rs_list)))
				cnt += 1
			f.flush()


def main(arguments):
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--sent1_lemma', help="Path to tokenized sent1 lemma.", default = "./data/snli_1.0/dev.sent1_lemma.txt")
	parser.add_argument('--sent2_lemma', help="Path to tokenized sent2 lemma.", default = "./data/snli_1.0/dev.sent2_lemma.txt")
	parser.add_argument('--rel', help="Type of relations to output, e.g. ant", default = "ant")
	parser.add_argument('--output', help="Path to an output folder", default = "./data/snli_1.0/conceptnet.ant.txt")
	parser.add_argument('--continu', help="Whether to continue previous processing", type=int, default = 0)
	parser.add_argument('--worker', help="Number of workers run in parallel", type=int, default = 16)
	opt = parser.parse_args(arguments)
	process(opt)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
	#cn = ConceptNet()
	#cn.distinct_from("blue", True)