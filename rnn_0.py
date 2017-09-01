import torch, time, math
import torch.autograd as autograd
import torch.nn as nn


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

def eva(prime_str='時間のある時', predict_len=200, )

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

  decoder = RNN ()
