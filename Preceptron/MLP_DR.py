import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_DR(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(MLP_DR, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # layer 1
        self.fc1 = nn.Linear(input_size, hidden_layers[0])

        # layer 2
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])

        # layer 3
        self.fc3 = nn.Linear(hidden_layers[1], output_size)

    def forward(self, x):

        linear_1 = self.fc1(x)
        act_1 = F.relu(linear_1)

        linear_2 = self.fc2(act_1)
        act_2 = F.relu(linear_2)

        linear_3 = self.fc3(act_2)
        act_3 = F.softmax(linear_3, dim=1)

        return act_3