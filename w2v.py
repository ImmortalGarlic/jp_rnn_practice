import glob, torch, sys, json, pickle
import MeCab as mcb
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


''' Owakati '''
def Tokenize(text):
  tagger = mcb.Tagger('-Owakati')
  wakati = tagger.parse(text)
  with open ('./{}_text.token'.format(author), 'a', encoding='utf-8') as f_0:
    f_0.write(wakati+'\n')

# list of punctuations used for splitting text
split_list = "\n　、。；：””’’ー＝｛｝「」（）【】『』《》〈〉｜！＊※＆〜｀"

''' Word Embedding '''
def word_embed(text):
  for punc in split_list:
    text.replace(punc, '; ')
  term_list = text.split('; ')
  # create unique term list
  word_uniq = list(set(term_list))
  pickle.dump(word_uniq, open('./{}_uni_words.pickle'.format(author), 'wb'))

  # create term dictionary: (term: idx)
  term_dict = {}
  for term in term_list:
    term_dict[term] = word_uniq.index(term)
  with open ('./{}_text_dict.json'.format(author), 'w', encoding='utf-8') as f_1:
    json.dump(term_dict, f_1)

  # we set embedding dimension to 10 here
  embeds = nn.Embedding(len(term_list), 10)
  lookup_tensor = torch.LongTensor([term_dict[x] for x in term_list])
  input_tensor = autograd.Variable(lookup_tensor)
  embedded = embeds(input_tensor)
  pickle.dump(embedded, open('./{}_w2v_tensor.pickle'.format(author), 'wb'))

if __name__ == '__main__':
  # choose the author
  authors = ['akutagawa', 'dazai', 'natsume']
  author = authors[0]
  text_name = glob.glob('./aosora/{}/*.txt'.format(author))

  if '--Tokenize' in sys.argv:
    for name in text_name:
      text = open(name, 'r', encoding='utf-8').read()
      Tokenize(text)
    print ('Owakati finished.')

  if '--word_embed' in sys.argv:
    try:
      text = open('./{}_text.token'.format(author), 'r', encoding='utf-8').read()
      word_embed(text)
    except FileNotFoundError as e:
      print ('Tokenize the text first! Use "--Tokenize"')

