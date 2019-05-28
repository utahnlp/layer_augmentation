import numpy as np
import h5py
import re
import sys
import operator
import argparse
import unicodedata


def load_word_dict(fname):
    word_idx = {}
    with open(fname, 'r+') as f:
      for l in f:
        if l.strip() != '':
          w, idx, cnt = l.split(' ')
          word_idx[w] = (int(idx), int(cnt.strip()))
    return word_idx


def count_char_freq(word_idx):
    char_freq = {}
    for w, (i, cnt) in word_idx.items():
      for c in w:
        #c = c.lower()
        if c not in char_freq:
          char_freq[c] = cnt
        else:
          char_freq[c] = char_freq[c] + cnt
    return char_freq


def register_char(word_idx, freq):
    char_freq = count_char_freq(word_idx)
    char_pool = [c for c, cnt in char_freq.items() if cnt >= freq]
    char_map = {'<blank>': 0}
    char_map['<unk>'] = 1
    _word_idx = _word_idx = sorted([(t, i) for t, (i, _) in word_idx.items()], key=lambda x: x[1])
    for t, _ in _word_idx:
      for c in t:
        #c = c.lower()
        if c in char_pool and c not in char_map:
          char_map[c] = len(char_map)
          #print(c, char_map[c])
    return char_map


def fix_length(idx, length):
    if len(idx) > length:
      return idx[:length]
    return idx + [0] * (length - len(idx))


def get_char_idx(word_idx, token_l, freq):
    char_map = register_char(word_idx, freq)
    tokens = [w for w, _ in word_idx.items()]

    num_word = len(word_idx)
    num_char = len(char_map)

    char_idx = np.zeros((num_word, token_l), dtype=int)
    _word_idx = sorted([(t, i) for t, (i, _) in word_idx.items()], key=lambda x: x[1])
    print(_word_idx[:100])
    for t, i in _word_idx:
        if i == 0 :  # for the <blank> word (the WORD!), set all 0
          assert(t == '<blank>')
          char_idx[i] = np.zeros((token_l,), dtype=int)
          continue

        #t_lower = [c.lower() for c in t]
        c_idx = [char_map[c] if c in char_map else 1 for c in t] # if not in char_map due to low freq, set to <unk>
        c_idx = fix_length(c_idx, token_l)
        char_idx[i] = np.array(c_idx, dtype=int)

    return char_idx, char_map


def write_char_dict(path, char_dict):
    _ordered = sorted([(k, idx) for k, idx in char_dict.items()], key=lambda x: x[1])
    print('writing {0} chars to dict file.'.format(len(_ordered)))
    with open(path, 'w+') as f:
        for c, idx in _ordered:
            f.write('{0}\t{1}\n'.format(c, idx))



def write_char_idx(path, char_idx):
    # Write output
    f = h5py.File(path, "w")        
    f["char_idx"] = char_idx
    
    print('writing char indices for {0} tokens.'.format(len(char_idx)))
    f.close()



def main(arguments):
  parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--dict', help="*.dict file", type=str, default='data/sarc.allword.dict')
  parser.add_argument('--output', help="output hdf5 file", type=str, default='data/char')
  parser.add_argument('--token_l', help="The maximal word length", type=int, default=16)
  parser.add_argument('--freq', help="The frequence bar for char to appear", type=int, default=50)
  
  opt = parser.parse_args(arguments)

  word_idx = load_word_dict(opt.dict)
  char_idx, char_map = get_char_idx(word_idx, opt.token_l, opt.freq)
  write_char_idx(opt.output + '.idx.hdf5', char_idx)
  write_char_dict(opt.output + '.dict.txt', char_map)

    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))