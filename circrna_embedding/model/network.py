import torch
import torch.nn as nn


class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()

        self.fc11 = nn.Linear(742, 300)
        self.fc12 = nn.Linear(300, 300)
        self.fc13 = nn.Linear(300, 300)
        self.fc14 = nn.Linear(300, 71)

        self.bn_input = nn.BatchNorm1d(300, momentum=0.5)
        self.bn_input2 = nn.BatchNorm1d(500, momentum=0.5)

    def forward(self, x1):

        x1 = self.fc11(x1)
        embedding_vector = x1

        x1 = torch.relu(x1)

        x1 = self.fc12(x1)
        x1 = torch.sigmoid(x1)

        x1 = self.fc13(x1)
        x1 = torch.relu(x1)

        x1 = self.fc14(x1)

        output = x1

        return output, embedding_vector
