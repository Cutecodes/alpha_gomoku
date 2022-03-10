import numpy as np 

class Board(object):
    """status for the game"""

    def __init__(self, size = 8):
        self.width = size
        self.height = size
        self.board = np.zeros((2,self.width,self.height),dtype=np.int)
        # 0 is the black ,1 is the white
        self.current_player = 0
        self.nums = 0
        self.last_move = None

    def move(self, move):
        
        _x = move//self.width
        _y = move%self.width
        if self.board[self.current_player][_x][_y]==0:
            self.last_move = move
            self.board[self.current_player][_x][_y] = 1.0
            self.current_player = (self.current_player + 1) % 2
            self.nums = self.nums+1
            return True
        else:
            return False

    def current_state(self):
        """
        return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height),dtype=np.float32)
        # current board
        square_state[0] = self.board[self.current_player]
        square_state[1] = self.board[(self.current_player+1)%2]
        
        # last_move
        if self.last_move is not None: 
            _x = self.last_move//self.width
            _y = self.last_move%self.width
            square_state[2][_x][_y] = 1
        square_state[3]=self.current_player
        return square_state

    def get_current_player(self):
        return self.current_player

    def get_board_size(self):
        return self.width

    def has_a_winner(self):
        width = self.width
        height = self.height

        for x in range(width):
            for y in range(height):
                for cur in range(2):
                    if self.board[cur][x][y]==1:
                        n=0
                        for i in range(5):
                            if y+i<height and self.board[cur][x][y+i]==self.board[cur][x][y]:
                                n=n+1
                        if n==5:
                            return True,cur

                        n=0
                        for i in range(5):
                            if x+i<width and self.board[cur][x+i][y]==self.board[cur][x][y]:
                                n=n+1
                        if n==5:
                            return True,cur

                        n=0
                        for i in range(5):
                            if x+i<width and y+i<height and self.board[cur][x+i][y+i]==self.board[cur][x][y]:
                                n=n+1
                        if n==5:
                            return True,cur

                        n=0
                        for i in range(5):
                            if x+i<width and y-i>=0 and self.board[cur][x+i][y-i]==self.board[cur][x][y]:
                               n=n+1
                        if n==5:
                            return True,cur

        if self.nums==width*height:
            return True,-1
        
        return False,-1
    
    def availables(self):
        '''
        返回可以下棋的位置
        '''
        ret = []
        for x in range(self.width):
            for y in range(self.height):
                if(self.board[0][x][y]==0 and self.board[1][x][y]==0):
                    ret.append(x*self.width+y)
        return ret
        

        

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        
        return False, -1
