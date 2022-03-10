import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode():
    """
    蒙特卡洛树节点

    Q：奖励 
    P： 可能性，由神经网络计算
    N：访问次数
    U：置信度

    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {} 
        self._N = 0
        self._Q = 0
        self._U = 0
        self._P = prior_p

    def expand(self, action_priors):
        """
        Expand
        action_priors: 由策略价值函数生成的下子可能性
        """

        for action, prob in action_priors:
            if self._children.get(action) is None:
                self._children[action] = TreeNode(self, prob)
        #print(len(self._children))

    def select(self, c_puct):
        """
        Select
        选出UCB最大的节点
        UCB = Q + U
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        Update 

        leaf_value: 神经网络估值或结果值，从当前PLAYER看
        """
        
        self._N += 1
        # 更新 Q, 求平均值
        self._Q += 1.0*(leaf_value - self._Q) / self._N

    def update_recursive(self, leaf_value):
        """
        递归更新所有祖先节点
        """
        
        if self._parent:    
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        计算UCB
        """
        self._U = (c_puct * self._P *
                   np.sqrt(self._parent._N) / (1 + self._N))
        return self._Q + self._U

    def is_leaf(self):
        """
        是否叶节点？
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS():
    """蒙特卡洛树搜索"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        """
        policy_value_fn: 输入当前状态，给出下一步各个策略的概率值和当前局面的打分
        [-1,1]，代表双方嬴面
        
        c_puct: 控制探索和利用的平衡，值越大越倾向探索
        n_playout:每次模拟次数
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        模拟，从根开始一直到叶子，然后反向传播更新
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            
            action, node = node.select(self._c_puct)
            state.move(action)
        #直到叶节点
        
        action_probs, leaf_value = self._policy(state)
        

        # 是否决出胜负
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            
            if winner == -1:  
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        根据模拟结果，返回实际走子和其可能性，用于训练
        temp: (0, 1] 控制探索等级
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        
        act_visits = [(act, node._N)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """
        模拟前进一步，通常在走子后更新树或初始化
        """

        if last_move in self._root._children:
        #走子在模拟的走子里面
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
        #走子从未模拟过
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI玩家"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000,is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.AI = True

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3):
        sensible_moves = board.availables()
        
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)

            if self._is_selfplay:
                return move, move_probs
            else:
                return move
            
        else:
            print("WARNING: the board is full")
    def __call__(self,board):
        return self.get_action(board, temp=1e-3)

