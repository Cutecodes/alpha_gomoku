

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



class PolicyValueNet(nn.Module):
    
    def __init__(self, board_size):
        super(PolicyValueNet, self).__init__()

        self.board_size = board_size

        # 公共层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 输出策略层
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_size*board_size,
                                 board_size*board_size)
        # 当前评分层
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_size*board_size, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # 公共层
        
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 输出策略层
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_size*self.board_size)
        x_act = F.log_softmax(self.act_fc1(x_act),dim=1)
        # 当前评分层
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_size*self.board_size)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val
    def policy_value_fn(self,state):

        legal_positions = state.availables()
        current_state = np.ascontiguousarray(state.current_state().reshape(
                -1, 4, self.board_size, self.board_size))
        
        log_act_probs, value = self.forward(
            Variable(torch.from_numpy(current_state)).float())
        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        
        act_probs = zip(legal_positions, act_probs[legal_positions])
        
        value = value.data[0]
        
        return act_probs, value


