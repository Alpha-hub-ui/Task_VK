from tqdm import tqdm
from sklearn.metrics import mean_absolute_error as mae
import torch

def train_model(model, criterion, optimizer, epochs, train_dataloader, test_dataloader=None):
    mae_train_all, mae_val_all = [], []
    for ep in range(epochs):
        loss = None
        mae_train_all = []
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            predict = model(x_batch)

            loss = criterion(predict, y_batch.reshape([len(y_batch), 1]))
            loss.backward()
            optimizer.step()

            mae_train_all.append(mae(y_batch.detach().numpy(), predict.detach().numpy()))

        print('\n')
        print('Epoch: ' + str(ep))

        print('Loss: ' + str(loss))
        print('Mean MAE on the train set: ' + str(sum(mae_train_all) / len(mae_train_all)))

        if test_dataloader is None:
          continue

        mae_val_all = []
        with torch.no_grad():
            for x_batch, y_batch in test_dataloader:
                predict = model(x_batch)

                mae_val_all.append(mae(y_batch.detach().numpy(), predict.detach().numpy()))

            print('Mean MAE on the test set: '+ str(sum(mae_val_all) / len(mae_val_all)))

    if test_dataloader is None:
      return sum(mae_train_all) / len(mae_train_all)

    return sum(mae_train_all) / len(mae_train_all),  sum(mae_val_all) / len(mae_val_all)