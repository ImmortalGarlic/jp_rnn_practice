import torch, time, math
import torch.autograd as autograd
import torch.nn as nn
import MeCab as mcb


class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, n_layers=1):
    super(RNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers

    self.encoder = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=True)
    self.decoder = nn.Linear(hidden_size, output_size)

  def forward(self, input, hidden):
    input = self.encoder(input.view(1, -1))
    output, hidden = self.gru(input.view(1, 1, -1), hidden)
    output = self.decoder(output.view(1, -1))
    return output, hidden

  def initHidden():
    return autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
''' end of class RNN '''

split_list = "\n　、。；：❝❞“””’’ー＝＃[]−｛｝「」（）【】『』《》〈〉｜！＊※＆〜｀ "

def eva(prime_str='時間のある時', predict_len=200, heat=0.8):
  tagger = mcb.Tagger('-Owakati')
  wakati = tagger.parse(prime_str)
  wakati = wakati.replace('\n', '').split(' ').remove('')
  word_uniq = open('./{}/uni_words'.format(author), 'r', encoding='utf-8').read().split('\n')
  wakati_index = [word_uniq.index(x) for x in wakati]
  wakati_tensor = torch.zeros(len(wakati_index)).long()
  for i in range(len(wakati_index)):
    wakati_tensor[i] = wakati_index[i]
  wakati_var = autograd.Variable(wakati_tensor)

  hidden = decoder.init_hidden()
  prime_input = wakati_var
  predicted = prime_str
  # Use priming string to "build up" hidden state
  for j in range(len(prime_str) - 1):
    _, hidden = decoder(prime_input[j], hidden)
  inp = prime_input[-1]
    
  for p in range(predict_len):
    output, hidden = decoder(inp, hidden)
        
    # Sample from the network as a multinomial distribution
    output_dist = output.data.view(-1).div(temperature).exp()
    top_i = torch.multinomial(output_dist, 1)[0]
        
    # Add predicted character to string and use as next input
    predicted_char = all_characters[top_i]
    predicted += predicted_char
    inp = char_tensor(predicted_char)
  return predicted

def train(input, target):
  hidden = decoder.initHidden()
  decoder.zero_grad()
  loss = 0
  for i in range(input.size()[0]):
      output, hidden = decoder(input[i], hidden)
      loss += criterion(output, target[i])
  loss.backward()
  decoder_optimizer.step()

  return output, loss.data[0] / input.size()[0]

if __main__ == '__name__':
  ''' training parameters '''
  n_epochs = 2000
  print_every = 100
  hidden_size = 1000
  n_layers = 1
  learning_rate = 0.0005

  decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
  decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()

  start = time.time()
  all_losses = []
  loss_avg = 0

  for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set())       
    loss_avg += loss

    if epoch % print_every == 0:
      print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
      print(evaluate('Wh', 100), '\n')

    if epoch % plot_every == 0:
      all_losses.append(loss_avg / plot_every)
      loss_avg = 0
