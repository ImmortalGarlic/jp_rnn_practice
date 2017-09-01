import glob, torch, sys, json, pickle, random
import MeCab as mcb
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


''' Owakati '''
def Tokenize(text, name):
  tagger = mcb.Tagger('-Owakati')
  wakati = tagger.parse(text)
  #with open ('./{}/all.token'.format(author, ), 'a', encoding='utf-8') as f_0:
  #  f_0.write(wakati+'\n')
  with open ('./{}/tokens/{}_text.token'.format(author, name), 'a', encoding='utf-8') as f_0:
    f_0.write(wakati+'\n')

# list of punctuations used for splitting text
split_list = "\n　、。；：❝❞“””’’ー＝＃[]−｛｝「」（）【】『』《》〈〉｜！＊※＆〜｀ "

''' Word Embedding '''
def word_embed(text, name):
  for punc in split_list:
    text = text.replace(punc, 'splithere')
  term_list = text.split('splithere')
  #with open ('./akutagawa/uni_words', 'w', encoding='utf-8') as f:
  #  word_uniq = list(set(term_list))
  #  for word in word_uniq:
  #    f.write('{}\n'.format(word))

  # create unique term list
  word_uniq = open('./{}/uni_words'.format(author), 'r', encoding='utf-8').read().split('\n')

  # create term dictionary: (term: idx)
  term_dict = {}
  for term in term_list:
    term_dict[term] = word_uniq.index(term)
  with open ('./{}/{}_dict.json'.format(author, name),'w', encoding='utf-8') as f_1:
    json.dump(term_dict, f_1, ensure_ascii=False)

  ''' We set embedding dimension to 100 here '''
  # 1st parameter should be the length of dictionary
  embeds = nn.Embedding(len(word_uniq), 100)
  # 1 list of indices is 1 batch
  
  lookup_tensor = torch.LongTensor([term_dict[x] for x in term_list])
  input_tensor = autograd.Variable(lookup_tensor)
  embedded = embeds(input_tensor)
  pickle.dump(embedded, open('./{}/{}_tensor.pickle'.format(author, name), 'wb'))

''' Tutorial starting from here
    TODO: Why no word embedding '''

''' Prepare training data '''
def random_chunk(token):
  for punc in split_list:
    token = token.replace(punc, 'weirdo')
  term_list = token.split('weirdo')
  chunk_len = int(len(term_list) * 0.6)
  # pick part of the data as training data
  print ('Picking random chunk...')
  start_index = random.randint(0, len(term_list) - chunk_len)
  end_index = start_index + chunk_len + 1
  # return indices of chunk
  word_uniq = open('./{}/uni_words'.format(author), 'r', encoding='utf-8').read().split('\n')
  chunk_index = [word_uniq.index(x) for x in term_list[start_index : end_index]]
  chunk_tensor = torch.zeros(len(chunk_index)).long()
  for i in range(len(chunk_index)):
    chunk_tensor[i] = chunk_index[i]
  chunk_variable = autograd.Variable(chunk_tensor)
  pickle.dump(chunk_variable, open('./{}/train_chunk.var'.format(author), 'wb'))
  print ('Finished dumping Torch.Variable.')
  # return autograd.Variable(chunk_index)


if __name__ == '__main__':
  # choose the author
  authors = ['akutagawa', 'dazai', 'natsume']
  author = authors[0]
  text_name = glob.glob('./aosora/{}/*.txt'.format(author))
  
  if '--Tokenize' in sys.argv:
    for name in text_name:
      text = open(name, 'r', encoding='utf-8').read()
      name = name.replace('./aosora/{}/'.format(author), '').replace('.txt', '')
      Tokenize(text, name)
    print ('Owakati finished.')

  if '--word_embed' in sys.argv:
    tokens = glob.glob ('./{}/*.token'.format(author))
    if tokens == []:
      print ('Tokenize the text first! Use "--Tokenize"')
    else:
      for name in tokens:
        text = open(name, 'r', encoding='utf-8').read()
        name = name.replace("./{}/".format(author), "").replace(".token", "")
        word_embed(text, name)

  if '--random_chunk' in sys.argv:
    tokens = glob.glob ('./{}/*.token'.format(author))
    if tokens == []:
      print ('Tokenize the text first! Use "--Tokenize"')
    else:
      for name in tokens:
        token = open(name, 'r', encoding='utf-8').read()
        random_chunk(token)
