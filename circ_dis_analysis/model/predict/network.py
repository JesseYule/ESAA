import torch
import torch.nn as nn


class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()

        self.fc11 = nn.Linear(300, 300)
        self.fc12 = nn.Linear(300, 300)
        self.fc13 = nn.Linear(300, 300)
        self.fc14 = nn.Linear(300, 300)

        self.fc21 = nn.Linear(300, 300)
        self.fc22 = nn.Linear(300, 300)
        self.fc23 = nn.Linear(300, 300)
        self.fc24 = nn.Linear(300, 300)

        self.fc31 = nn.Linear(1200, 1800)
        self.fc32 = nn.Linear(1800, 1200)
        self.fc33 = nn.Linear(1200, 1)


        self.bn_input = nn.BatchNorm1d(300, momentum=0.5)
        self.bn_input2 = nn.BatchNorm1d(500, momentum=0.5)

    def forward(self, x1, x2):

        origin_x1 = x1
        origin_x2 = x2

        x1 = self.fc11(x1)

        x1 = torch.relu(x1)

        x1 = self.fc12(x1)
        x1 = torch.sigmoid(x1)

        x1 = self.fc13(x1)
        x1 = torch.relu(x1)

        x1 = self.fc14(x1)

        x2 = self.fc21(x2)
        x2 = torch.relu(x2)

        x2 = self.fc22(x2)
        x2 = torch.sigmoid(x2)

        x2 = self.fc23(x2)
        x2 = torch.relu(x2)

        x2 = self.fc24(x2)

        diff = x1 - x2
        multi = x1.mul(x2)

        # output = torch.cat((origin_x1, origin_x2), 1)
        # output = torch.cat((output, x1), 1)
        # output = torch.cat((output, x2), 1)

        output = torch.cat((x1, x2), 1)

        output = torch.cat((output, diff), 1)
        output = torch.cat((output, multi), 1)

        output = self.fc31(output)
        output = torch.relu(output)

        output = self.fc32(output)
        output = torch.relu(output)

        output = self.fc33(output)
        output = torch.sigmoid(output)
        output = torch.squeeze(output)

        return output
