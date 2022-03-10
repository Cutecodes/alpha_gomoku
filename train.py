import os
import random
import numpy as np
import torch.optim as optim
from Board import Board
from Game import Game
from MCTS import MCTSPlayer
import torch
from torch.autograd import Variable
from policy_value_net import PolicyValueNet
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from collections import deque

class STATE(Dataset):
    #need implement __len__(),__getitem__():
    def __init__(self,buffersize=10000):
        self.data_buffer = deque(maxlen=buffersize)

    def __call__(self,state,act_prob,z):
        self.state = torch.tensor(state, dtype=torch.float32)
        self.act_prob = torch.tensor(act_prob, dtype=torch.float32)
        self.z = torch.tensor(z, dtype=torch.float32)
        self.board_size = len(state[0][0][0])
        self.data = []
        for i in range(len(state)):
            self.data.append((self.state[i],self.act_prob[i],self.z[i]))
        self.data = self.get_equi_data(self.data)
        self.data_buffer.extend(self.data)
        

    def get_equi_data(self, play_data):
        """
        扩容数据
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_size, self.board_size)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self,index):
        return self.data_buffer[index]

def train(epoch,net,data_set):
    optimizer = optim.Adam(net.parameters(),lr=0.00001,
                                    weight_decay=1e-4)
    #weight_decay l2 正则项
    loss_meter = []
    entropy_meter =[]
    f = open("loss",'a')
    for i in range(1):
        print("self playing")
        player = MCTSPlayer(policy_value_function=net.policy_value_fn,
                 c_puct=5, n_playout=400,is_selfplay=1)
        game = Game(board_size=8,player1=player,player2=player)
        state,act_prob,z = game.self_play()
        #print(z)
        data_set(state,act_prob,z)
        data_loader = DataLoader(dataset=data_set,batch_size=512,shuffle=True)
        for batch_idx,data in enumerate(data_loader):
            inputdata = data[0]
            act_prob = data[1]
            val = data[2]
            target_act_prob,target_val = net(inputdata)
            # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
            # Note: the L2 penalty is incorporated in optimizer
            value_loss = F.mse_loss(val.view(-1,1), target_val)
            policy_loss = -torch.mean(torch.sum(act_prob*target_act_prob, 1))
            loss = value_loss + policy_loss
            # backward and optimize
            #print(loss_meter)
            loss_meter.append(loss.data)
            #entropy_meter.append(entropy.data)
            optimizer.zero_grad() #梯度清零
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*1000 , len(data_loader.dataset),
                100*batch_idx*1000 / len(data_loader.dataset), loss.data))
        #f.write("%s,%s\n"%(str(sum(loss_meter)/len(loss_meter)),str(sum(entropy_meter)/len(entropy_meter))))
        f.write("%s\n"%(str(sum(loss_meter)/len(loss_meter))))
    f.close()
             

        
def main():
    # 如果模型文件存在则尝试加载模型参数
    net = PolicyValueNet(8)
    if os.path.exists('./model.pth'):
        try:
            net.load_state_dict(torch.load('./model.pth'))
        except Exception as e:
            print(e)
            print("Parameters Error")
    data_set = STATE()
    for epoch in range(1, 500):
        train(epoch,net,data_set)
        torch.save(net.state_dict(),'./model.pth')
        

if __name__ == '__main__':
    main()
