import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_features, ch):
        super().__init__()

        self.input_features = input_features

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        self.lin1 = nn.Linear(input_features, ch)
        self.lin2 = nn.Linear(ch, 1)

        self.block = nn.Sequential(
                    self.lin1,
                    self.dropout,
                    self.lin2,
                )


    def forward(self, x):
        out = self.block(x)
        return out