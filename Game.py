from Board import Board
import pygame
import os
import numpy as np
import threading
class Game():
    def __init__(self, board_size,player1,player2):
        self.board_size = board_size
        self._board = Board(board_size)
        self._player1 = player1
        self._player2 = player2
        self.step = False

    def draw_background(self,screen,background, back_rect):
        GRID_WIDTH = 36
        WIDTH = (self.board_size+2) * GRID_WIDTH
        HEIGHT = (self.board_size+2) * GRID_WIDTH
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        screen.blit(background, back_rect)
        rect_lines = [
            ((GRID_WIDTH, GRID_WIDTH), (GRID_WIDTH, HEIGHT - GRID_WIDTH)),
            ((GRID_WIDTH, GRID_WIDTH), (WIDTH - GRID_WIDTH, GRID_WIDTH)),
            ((GRID_WIDTH, HEIGHT - GRID_WIDTH),
             (WIDTH - GRID_WIDTH, HEIGHT - GRID_WIDTH)),
            ((WIDTH - GRID_WIDTH, GRID_WIDTH),
             (WIDTH - GRID_WIDTH, HEIGHT - GRID_WIDTH)),
        ]
        for line in rect_lines:
            pygame.draw.line(screen, BLACK, line[0], line[1], 2)

        for i in range(self.board_size):
            pygame.draw.line(screen, BLACK,
                             (GRID_WIDTH * (2 + i), GRID_WIDTH),
                             (GRID_WIDTH * (2 + i), HEIGHT - GRID_WIDTH))
            pygame.draw.line(screen, BLACK,
                             (GRID_WIDTH, GRID_WIDTH * (2 + i)),
                             (HEIGHT - GRID_WIDTH, GRID_WIDTH * (2 + i)))

        circle_center = [
            (GRID_WIDTH * 4, GRID_WIDTH * 4),
            (WIDTH - GRID_WIDTH * 4, GRID_WIDTH * 4),
            (WIDTH - GRID_WIDTH * 4, HEIGHT - GRID_WIDTH * 4),
            (GRID_WIDTH * 4, HEIGHT - GRID_WIDTH * 4),
        ]
        for cc in circle_center:
            pygame.draw.circle(screen, BLACK, cc, 5)

    def draw_stone(self,screen):
        GRID_WIDTH = 36
        WIDTH = (self.board_size+2) * GRID_WIDTH
        HEIGHT = (self.board_size+2) * GRID_WIDTH
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        for i in range(self.board_size):
            for j in range(self.board_size):
                cur_player = self._board.current_state()[3][0][0]
                if(cur_player==0):
                    if self._board.current_state()[0][i,j] == 1:
                        pygame.draw.circle(screen, BLACK, (int((i+1.5)*GRID_WIDTH), int((j+1.5)*GRID_WIDTH)), 16)
                    if self._board.current_state()[1][i,j] == 1:
                        pygame.draw.circle(screen, WHITE, (int((i + 1.5) * GRID_WIDTH), int((j + 1.5) * GRID_WIDTH)), 16)
                else:
                    if self._board.current_state()[1][i,j] == 1:
                        pygame.draw.circle(screen, BLACK, (int((i+1.5)*GRID_WIDTH), int((j+1.5)*GRID_WIDTH)), 16)
                    if self._board.current_state()[0][i,j] == 1:
                        pygame.draw.circle(screen, WHITE, (int((i + 1.5) * GRID_WIDTH), int((j + 1.5) * GRID_WIDTH)), 16)
    
    def player1(self):
        print("Waiting player1....")
        move = self._player1(self._board)           
        self.step = self._board.move(move)
    def player2(self):
        print("Waiting player2....")
        move = self._player2(self._board)          
        self.step = self._board.move(move)
    def run(self):
        GRID_WIDTH = 36
        WIDTH = (self.board_size+2) * GRID_WIDTH
        HEIGHT = (self.board_size+2) * GRID_WIDTH
        FPS = 30

        # define colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)

        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("五子棋")
        clock = pygame.time.Clock()

        all_sprites = pygame.sprite.Group()

        base_folder = os.path.dirname(__file__)

        img_folder = os.path.join(base_folder, 'images')
        background_img = pygame.image.load(os.path.join(img_folder, 'back.png')).convert()

        background = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
        back_rect = background.get_rect()

        win,winner = self._board.game_end()
        self.draw_background(screen,background, back_rect)
        self.draw_stone(screen)
        pygame.display.flip()
        while not win:
            clock.tick(FPS)
            
            if hasattr(self._player1, 'AI'):
                ti = threading.Thread(target=self.player1)
                ti.start()
            else:
                self.player1()

            while True:
                #self.draw_background(screen,background, back_rect)
                #self.draw_background(screen,background, back_rect)
                #self.draw_stone(screen)
                #pygame.display.flip()
                if self.step:
                    self.step = False
                    break
                clock.tick(FPS)

                for event in pygame.event.get():
                    pygame.display.flip()
                    if event.type == pygame.QUIT:
                        pygame.quit()
            win,winner = self._board.game_end()
            #print(self._board.current_state())
            self.draw_background(screen,background, back_rect)
            self.draw_stone(screen)
            pygame.display.flip()
            if win:
                return winner
            
            if hasattr(self._player1, 'AI'):
                ti = threading.Thread(target=self.player2)
                ti.start()
            else:
                self.player2()

            while True:
                #move = self._player2(self._board)
                #self.draw_background(screen,background, back_rect)
                #self.draw_stone(screen)
                #pygame.display.flip()
                #T = self._board.move(move)
                #self.draw_background(screen,background, back_rect)
                #self.draw_stone(screen)
                #pygame.display.flip()
                if self.step:
                    self.step = False
                    break
                clock.tick(FPS)
                
                for event in pygame.event.get():
                    pygame.display.flip()
                    if event.type == pygame.QUIT:
                        pygame.quit()
                
            win,winner = self._board.game_end()
            self.draw_background(screen,background, back_rect)
            self.draw_stone(screen)
            pygame.display.flip()
            if win:
                return winner
    
    def self_play(self):
        state = []
        action_probs = []
        z = []
        win,winner = self._board.game_end()
        
        while not win:

            
            state.append(self._board.current_state())
            while True:
                move,act_p = self._player1(self._board)
                
                T=self._board.move(move)
                if T:
                    break
            action_probs.append(act_p)
            
            print(self._board.current_state())
            win,winner = self._board.game_end()
            if win:
                z = np.zeros(len(state))
                if winner==-1:
                    return state,action_probs,z
                else:
                    for i in range(len(state)):
                        if state[i][3][0][0]==winner:  
                            z[i]=1
                        else:
                            z[i]=-1
                    return state,action_probs,z

            state.append(self._board.current_state())
            while True:

                move,act_p = self._player2(self._board)
                
                T = self._board.move(move)
                if T:
                    break
            action_probs.append(act_p)
            
            print(self._board.current_state())
            win,winner = self._board.game_end()
            if win:
                z = np.zeros(len(state))
                if winner==0:
                    return state,action_probs,z
                else:
                    for i in range(len(state)):
                        if state[i][3][0][0]==winner:
                            z[i]=1
                        else:
                            z[i]=-1
                    return state,action_probs,z
    


