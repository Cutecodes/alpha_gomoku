import pygame
import os
import torch
from Board import Board
import argparse
from Game import Game
import numpy as np
from MCTS import MCTSPlayer
from policy_value_net import PolicyValueNet

def human_player(board):
    board_size = board.get_board_size()
    GRID_WIDTH = 36
    WIDTH = (board_size+2) * GRID_WIDTH
    HEIGHT = (board_size+2) * GRID_WIDTH
    while True:
        for event in pygame.event.get():
            pygame.display.flip()
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                
                pos = event.pos
                if pos[0] < GRID_WIDTH or pos[1] < GRID_WIDTH or pos[0] > WIDTH - GRID_WIDTH or pos[
                    1] > HEIGHT - GRID_WIDTH:
                    pass
                else:
                    grid = int((pos[0] - GRID_WIDTH) / GRID_WIDTH)*board_size+int((pos[1] - GRID_WIDTH) / GRID_WIDTH)
                    return grid
           
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="display", type=str, help="decide which mode too choose, we can choose:display, game,human_play")
    
    args = parser.parse_args()
    net = PolicyValueNet(8)
    if os.path.exists('./model.pth'):
        try:
            net.load_state_dict(torch.load('./model.pth'))
        except Exception as e:
            print(e)
            print("Parameters Error")
    player = MCTSPlayer(policy_value_function=net.policy_value_fn,
                 c_puct=5, n_playout=1000,is_selfplay=0)
    if args.mode == "display":
        game = Game(8,player,player)
    elif args.mode == "game":
        game = Game(8,human_player,player)
    elif args.mode == "human_play":
        game = Game(8,human_player,human_player)
    else:
        raise KeyError("we must select a mode between 'display' and 'game'.")
    winner = game.run()
    if winner !=-1:
        print("The winner is:")
        if winner==0:
            print("black")
        else:
            print("white")
    else:
        print("Tie")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

        

if __name__ == "__main__":
    main()

