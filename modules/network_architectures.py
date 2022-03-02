# pytorch for neural network
import torch
import torch.nn as nn

class Linear_Classifier(nn.Module):

  def __init__(self, num_classes:int=2, inp_dim:int=3) -> None:
    super(Linear_Classifier, self).__init__()
    self.classifier = nn.Sequential(
      nn.Linear(inp_dim,16),
      nn.ReLU(inplace=True),
      nn.Linear(16,32),
      nn.ReLU(inplace=True),
      nn.Linear(32,16),
      nn.ReLU(inplace=True),
      nn.Linear(16, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.classifier(x)
    return x

class Conv_Classifier(nn.Module):

  def __init__(self, num_classes:int=2, inp_channels:int=3) -> None:
    super(Conv_Classifier, self).__init__()
    self.features = nn.Sequential(
      nn.Conv1d(inp_channels, 64, kernel_size=12, stride=4),   # np.floor((Lin + 2*padding - kernel_size)/stride + 1)
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=2, stride=4),
      nn.Conv1d(64, 192, kernel_size=4, stride=4),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=4, stride=4),
      nn.Conv1d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv1d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv1d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=4, stride=4),
    )
    self.avgpool = nn.AdaptiveAvgPool1d(6)
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 6, 512),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(512, 128),
      nn.ReLU(inplace=True),
      nn.Linear(128, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x




# 1D AlexNet network class definition
class AlexNet(nn.Module):

  def __init__(self, num_classes:int=2, inp_channels:int=3) -> None:
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(    # inp_len = 500             np.floor((Lin + 2*padding - kernel_size)/stride + 1)
      nn.Conv1d(inp_channels, 64, kernel_size=11, stride=4, padding=2),   # 500 -> 124
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2),    # 124 -> 61
      nn.Conv1d(64, 192, kernel_size=5, padding=2), # same
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2),  # 61 -> 30
      nn.Conv1d(192, 384, kernel_size=3, padding=1),  # same
      nn.ReLU(inplace=True),
      nn.Conv1d(384, 256, kernel_size=3, padding=1),  # same
      nn.ReLU(inplace=True),
      nn.Conv1d(256, 256, kernel_size=3, padding=1),  # same
      nn.ReLU(inplace=False),   # inplace operation would prevent backward hook
      nn.MaxPool1d(kernel_size=3, stride=2),    # 30 -> 14
    )
    self.avgpool = nn.AdaptiveAvgPool1d(6)
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 6, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x




class MV_LSTM(nn.Module):
  def __init__(self,n_features,seq_length,n_classes):
    super(MV_LSTM, self).__init__()
    self.n_features = n_features
    self.seq_len = seq_length
    self.n_hidden = 20 # number of hidden states
    self.n_layers = 1 # number of LSTM layers (stacked)

    self.l_lstm = nn.LSTM(input_size = n_features, 
                              hidden_size = self.n_hidden,
                              num_layers = self.n_layers, 
                              batch_first = True)
    # according to pytorch docs LSTM output is 
    # (batch_size,seq_len, num_directions * hidden_size)
    # when considering batch_first = True
    self.l_linear = nn.Linear(self.n_hidden*self.seq_len, n_classes)


  def init_hidden(self, batch_size):
    # even with batch_first = True this remains same as docs
    hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
    cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
    self.hidden = (hidden_state, cell_state)
  
  
  def forward(self, x):        
    batch_size, seq_len, _ = x.size()
    
    lstm_out, self.hidden = self.l_lstm(x,self.hidden)
    # lstm_out(with batch_first = True) is 
    # (batch_size,seq_len,num_directions * hidden_size)
    # for following linear layer we want to keep batch_size dimension and merge rest       
    # .contiguous() -> solves tensor compatibility error
    x = lstm_out.contiguous().view(batch_size,-1)
    return self.l_linear(x)  