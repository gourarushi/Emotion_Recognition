import copy
import numpy as np
from tqdm.auto import tqdm
from functools import partial
import matplotlib.pyplot as plt

# pytorch for neural network
import torch
import torch.nn as nn

from sklearn.metrics import classification_report

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tqdm = partial(tqdm, position=0, leave=False)



def trainNet(net,criterion,optimizer,data_loaders,epochs, check_every=None,earlyStopping=False,verbose=3):

  if verbose>0: print("training network")
  train_loader = data_loaders['train']
  val_loader = data_loaders.get('val', [])
  sel = 'val' if val_loader else 'train'
  best_epoch = None

  if not check_every:
      check_every = epochs // 10 if epochs > 10 else 1

  avg_Losses = {'train':[], 'val':[]}

  for epoch in tqdm(range(epochs), disable=verbose<2):  # loop over the dataset multiple times

    train_loss = []
    val_loss = []
    avg_Loss = {}

    net.train()
    for i, (inputBatch,labelBatch) in enumerate(tqdm(train_loader, desc='train', disable=verbose<3)):

        inputBatch = inputBatch.to(device).float()
        labelBatch = labelBatch.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # for LSTM
        if hasattr(net,'init_hidden'):
          net.hidden = net.init_hidden(inputBatch.shape[0]) # detaching it from its history on the last instance.

        # forward
        outputBatch = net(inputBatch)
        loss = criterion(outputBatch, labelBatch)
        train_loss.append(loss.item())

        # backward + optimize
        loss.backward()
        optimizer.step()
    avg_Loss['train'] = np.mean(train_loss)

    if val_loader:
      net.eval()
      for i, (inputBatch,labelBatch) in enumerate(tqdm(val_loader, desc='val', disable=verbose<3)):
        with torch.no_grad():

          inputBatch = inputBatch.to(device).float()
          labelBatch = labelBatch.to(device)

          # for LSTM
          if hasattr(net,'init_hidden'):
            net.hidden = net.init_hidden(inputBatch.shape[0]) # detaching it from its history on the last instance.

          # forward
          outputBatch = net(inputBatch)
          loss = criterion(outputBatch, labelBatch)
          val_loss.append(loss.item())

      avg_Loss['val'] = np.mean(val_loss)


    if epoch > 0:
      if avg_Loss[sel] < min(avg_Losses[sel]):
        best_params = copy.deepcopy(net.state_dict())
        best_epoch, best_loss = epoch, avg_Loss[sel]
    else:
      movAvg_old = avg_Loss[sel]

    avg_Losses['train'].append(avg_Loss['train'])
    if val_loader:
      avg_Losses['val'].append(avg_Loss['val'])

    # print statistics
    if epoch % check_every == check_every - 1:
      if verbose > 1:
        print('epoch: %d  | train loss: %.3f, val loss: %.3f' % (epoch + 1, avg_Loss['train'], avg_Loss.get('val',np.nan)), end="  | ")
        print('avg train loss: %.3f, avg val loss: %.3f' % (np.mean(avg_Losses['train'][epoch+1-check_every:epoch+1]), np.mean(avg_Losses['val'][epoch+1-check_every:epoch+1]) if val_loader else np.nan) )

      movAvg_new = np.mean(avg_Losses[sel][epoch+1-check_every:epoch+1])
      if earlyStopping:
        if movAvg_old < movAvg_new:
          break
        else:
          movAvg_old = movAvg_new

  if not best_epoch:
    best_params = copy.deepcopy(net.state_dict())


  if verbose > 0:
    print('Finished Training')
    plt.plot(avg_Losses['train'], label='train loss')
    if val_loader:
      plt.plot(avg_Losses['val'], label='val loss')
    if best_epoch:
      #plt.plot([best_loss]*epoch, linestyle='dashed')
      plt.plot(best_epoch, best_loss, 'o')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

  return best_params


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def get_predn(net, loader, predict_fn='max'):
  net.eval()
  outTrue = []
  outPred = []

  for i, (inputBatch,outTrueBatch) in enumerate(tqdm(loader, disable=True)):
    with torch.no_grad():

      inputBatch = inputBatch.to(device).float()
      outTrue.extend(outTrueBatch.cpu())

      # for LSTM
      if hasattr(net,'init_hidden'):
        net.hidden = net.init_hidden(inputBatch.shape[0]) # detaching it from its history on the last instance.

      # forward
      if predict_fn == 'max':
        outPredBatch = net(inputBatch).argmax(1)
      elif predict_fn == 'threshold':
        outPredBatch = net(inputBatch)
        outPredBatch = torch.where(outPredBatch>=0.5, 1, 0)

      outPred.extend(outPredBatch.cpu())

  return outTrue, outPred

def evaluate(net, loader, output_dict=False, predict_fn='max'):
  outTrue, outPred = get_predn(net, loader, predict_fn=predict_fn)
  return classification_report(outTrue, outPred, digits=4, output_dict=output_dict)