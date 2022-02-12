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