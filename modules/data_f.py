import os
import re
import zipfile
import requests
import numpy as np
from tqdm.auto import tqdm
from functools import partial

# pytorch for neural network
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# sklearn
from sklearn.model_selection import train_test_split

tqdm = partial(tqdm, position=0, leave=False)


# download file from url
def download_file(url,save_path):
  print(f'downloading file from {url}')
  print(f'to {save_path}')
  response = requests.get(url, stream=True)
  d = response.headers['content-disposition']
  fname = re.findall("filename=(.+)", d)[0][1:-1]
  save_path += fname

  if not os.path.exists(save_path):
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 2048 #2 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    else:    
      print("file downloaded")
  else:
    print("file already exists")
  return fname  


# download file from url
def extract_zip(path_to_zip_file, directory_to_extract_to=None, overwrite=True):
  if not directory_to_extract_to:
    directory_to_extract_to = os.path.dirname(path_to_zip_file)
  print(f'extracting {path_to_zip_file} to {directory_to_extract_to}')
  if not os.path.exists(directory_to_extract_to) or overwrite:
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
      for member in tqdm(zip_ref.infolist(), desc='Extracting '):
        zip_ref.extract(member, directory_to_extract_to)
    print("folder extracted from zip")
  else:
    print("zip already extracted")


def get_weights(target):
  class_sample_index, class_sample_count = np.unique(target, return_counts=True)
  weight = 1. / class_sample_count
  samples_weight = np.array([weight[class_sample_index==t][0] for t in tqdm(target)])
  samples_weight = torch.from_numpy(samples_weight)
  return samples_weight

def create_trainValLoaders(train_inputs, train_labels, test_size=.25, batch_size=32):
  dataloaders = {}
  train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels, test_size=test_size, random_state=0)

  dataloaders['train'] = createLoader(train_inputs, train_labels, batch_size=batch_size, shuffle=True)
  dataloaders['val'] = createLoader(val_inputs, val_labels, batch_size=batch_size)
  return dataloaders

# function to create train and val loaders
def createLoader(inputs, labels, batch_size=32, shuffle=False, sample_equally=False):
  inputs, labels = map(lambda x: torch.from_numpy(x), (inputs, labels))
  
  my_dataset = TensorDataset(inputs, labels)
  if sample_equally:
    samples_weight = get_weights(labels.numpy())
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
  else: sampler = None
  my_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

  return my_loader