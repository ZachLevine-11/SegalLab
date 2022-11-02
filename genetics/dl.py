import torch
from LabData.DataLoaders.PRSLoader import PRSLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from genetics_dataset import GeneticsDataset

data = GeneticsDataset()

class fc_snpnet(nn.Module):
    def __init__(self):
        super(fc_snpnet, self).__init__()
        self.conv1 = nn.Conv1d(31, 31, 2000)
        self.conv2 = nn.Conv1d(31, 31, 1000)
        self.conv4 = nn.Conv1d(31, 31, 1000)
        self.conv6 = nn.Conv1d(31, 31, 1000)
        self.conv7 = nn.Conv1d(31, 31, 1000)
        self.fc1 = nn.Linear(3596, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 516)
        self.fc4 = nn.Linear(516, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.reshape(x, [2, 31, 6053])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv4(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.flatten()
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        x = self.fc6(x)
        return x

net = fc_snpnet()

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.2)
for epoch in range(30):
    i = 0
    for tenK_index in list(data.binaries.sample.values)[0:4000]:
        try:
            snps, label = data.__getitem__(tenK_index)
            optimizer.zero_grad()
            outputs = net(snps)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            if i % 10 == 0: ##print every 50 people
                print("Epoch: ", epoch)
                print("Loss: ", loss)
                print("output: ", outputs, ", true height: ", label)
        except IndexError:
            pass
        i += 1