import numpy as np

# sklearn
from sklearn.preprocessing import MinMaxScaler




def flatten_apply(func):
  def inner(inputs, *args, **kwargs):
    n_people, n_vids = inputs.shape[:2]
    inputs = inputs.reshape(-1, *inputs.shape[2:])
    inputs = np.array([func(input_, *args, **kwargs) for input_ in inputs])
    inputs = inputs.reshape(n_people, n_vids, *inputs.shape[1:])
    return inputs
  return inner

@flatten_apply
def interpolate(inp, new_len):    # channel wise interpolation
  inp_ch, inp_len = inp.shape
  for c in range(inp_ch):
    inp[c] = np.interp(np.linspace(0,len(inp[c])-1,new_len), range(len(inp[c])), inp[c])

@flatten_apply
def normalize(inp, feature_range=(-1,1)):    # channel wise normalization
  inp_ch, inp_len = inp.shape
  for c in range(inp_ch):
    scaler = MinMaxScaler(feature_range=feature_range).fit(inp[c].reshape(-1,1)) 
    inp[c] = scaler.transform(inp[c].reshape(-1,1))[:,0]


def sliding_window(inp, sub_window_size, stride_size):
  inp_ch, inp_len = inp.shape
    
  sub_windows = (
      np.expand_dims(np.arange(sub_window_size), 0) +
      # Create a rightmost vector as [0, V, 2V, ...].
      np.expand_dims(np.arange(inp_len - sub_window_size + 1, step=stride_size), 0).T
  )

  return inp[:,sub_windows]


@flatten_apply
def get_features(inputs, sub_window_size, stride_size):

  windows = sliding_window(inputs, sub_window_size=sub_window_size, stride_size=stride_size)
  mean = windows.mean(axis=2)
  std = windows.std(axis=2)

  features = np.vstack((mean, std))
  return features