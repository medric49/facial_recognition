from torch import nn
from torch.nn import functional as F

WIDTH = 150
HEIGHT = 150
PEOPLE = ['Behekaken', 'Djenal', 'Nafi', 'Rossif']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_1 = 780
        hidden_2 = 256
        hidden_3 = 64
        hidden_4 = len(PEOPLE)
        self.fc1 = nn.Linear(WIDTH * HEIGHT, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc4 = nn.Linear(hidden_3, hidden_4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, WIDTH * HEIGHT)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
