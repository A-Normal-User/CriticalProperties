import torch
import torch.nn as nn

class ResNet_std(nn.Module):
    def __init__(self, hidden_size, activation='tanh'):
        super(ResNet_std, self).__init__()
        if activation == 'tanh':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
        elif activation == 'sigmoid':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
        elif activation == 'gelu':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
        elif activation == 'silu':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            )

    def forward(self, x):
        y = self.L1(x)
        return x + y
    
class ResNet_1_1(nn.Module):
    def __init__(self, hidden_size, activation='tanh'):
        super(ResNet_1_1, self).__init__()
        if activation == 'tanh':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.L2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            )
        elif activation == 'sigmoid':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
            self.L2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
        elif activation == 'gelu':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
            self.L2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
        elif activation == 'silu':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            )
            self.L2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            )
    def forward(self, x):
        y = self.L1(x)
        return self.L2(y) + y
    
class ResNet_2_2(nn.Module):
    def __init__(self, hidden_size, activation='tanh'):
        super(ResNet_2_2, self).__init__()
        if activation == 'tanh':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.L2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
        elif activation == 'sigmoid':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
            self.L2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
        elif activation == 'gelu':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
            self.L2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
        elif activation == 'silu':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            )
            self.L2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            )
    def forward(self, x):
        y = self.L1(x)
        return self.L2(y) + y

class ResNet_1_3(nn.Module):
    def __init__(self, hidden_size, activation='tanh'):
        super(ResNet_1_3, self).__init__()
        if activation == 'tanh':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.L2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
        elif activation == 'sigmoid':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
            self.L2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
        elif activation == 'gelu':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
            self.L2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
        elif activation == 'silu':
            self.L1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            )
            self.L2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            )
    def forward(self, x):
        y = self.L1(x)
        return self.L2(y) + y

class Crit_model(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_type, hidden_num, output_size, activation='tanh', dropout=0.1):
        super(Crit_model, self).__init__()
        self.start = nn.Linear(input_size, hidden_size)
        self.end = nn.Linear(hidden_size, output_size)
        self.mid = nn.Sequential()
        self.drop_ = nn.Dropout(p=dropout)
        if hidden_type == 'std':
            for i in range(hidden_num):
                self.mid.add_module(f"resnet_{i}", ResNet_std(hidden_size, activation=activation))
                if i == hidden_num - 2:
                    self.mid.add_module(f"dropout_{i}", self.drop_)
        elif hidden_type == '1_1':
            for i in range(hidden_num):
                self.mid.add_module(f"resnet_{i}", ResNet_1_1(hidden_size, activation=activation))
                if i == hidden_num - 2:
                    self.mid.add_module(f"dropout_{i}", self.drop_)
        elif hidden_type == '2_2':
            for i in range(hidden_num):
                self.mid.add_module(f"resnet_{i}", ResNet_2_2(hidden_size, activation=activation))
                if i == hidden_num - 2:
                    self.mid.add_module(f"dropout_{i}", self.drop_)
        elif hidden_type == '1_3':
            for i in range(hidden_num):
                self.mid.add_module(f"resnet_{i}", ResNet_1_3(hidden_size, activation=activation))
                if i == hidden_num - 2:
                    self.mid.add_module(f"dropout_{i}", self.drop_)
    def forward(self, x):
        return self.end(self.mid(self.start(x)))