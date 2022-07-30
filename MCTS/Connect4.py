import numpy as np

class Connect4Environment:
    EMPTY = '◌'
    MAX_ROWS = 6
    MAX_COLS = 7

    def __init__(self):
        self.state_space = (self.MAX_ROWS*self.MAX_COLS)
        self.action_space = (self.MAX_COLS)
        self.reset()
        
        
    def reset(self):
        self.board = []  # drawing the board
        for i in range(self.MAX_ROWS):
            row = []
            for j in range(self.MAX_COLS):
                row.append(self.EMPTY)
            self.board.append(row)
        # players
        self.player = {
            "p1": "◍",
            "p2": "●"

        }
        self.p1_to_move = True  # flag to select which player will play
        self.move_log = []      # moves log to save recent moves to enable undo button
        self.p_won = False
        self.curr_pos = [0, 0]

    def getValidMoves(self):
        valid_move = []
        for pos in range(self.MAX_COLS):
            if self.board[0][pos] == self.EMPTY:
                valid_move.append(pos)
        return valid_move

    def get_valid_actions(self):
        valid = np.zeros(self.MAX_COLS)
        valid_moves = self.getValidMoves()
        for i in range(self.MAX_COLS):
            if i in valid_moves:
                valid[i] = 1
        return valid

    def grid_to_state(self,grid):
        move_dict = {'◍':1,'●':-1}  # Default is player == 1, X will be 1, O will be -1
        state = np.zeros(self.MAX_COLS*self.MAX_ROWS)
        for i in range(self.MAX_ROWS):
            for j in range(self.MAX_COLS):
                if grid[i][j] in [ "◍", "●"]:
                    state[i*self.MAX_COLS+j] = move_dict[grid[i][j]]

        if not self.p1_to_move: # X will be -1, O will be 1
            state *= -1

        return state
    
    def game_state(self):
        observation = self.grid_to_state(self.board)
        r = self.winning()
        if r:
            return observation,r,True,None
        
        if np.sum(self.get_valid_actions())==0:
            return observation,0,True,None
        
        return observation,0,False,None

    def take_action(self,action):
        self.makeMove(action)
        return self.game_state()

    def makeMove(self, col: int):
        pos_x_board = 0
        while self.board[pos_x_board][col] == self.EMPTY and pos_x_board != (self.MAX_ROWS - 1):
            pos_x_board = pos_x_board + 1
        if self.board[pos_x_board][col] != self.EMPTY:
            pos_x_board = pos_x_board - 1

        if self.p1_to_move:
            self.board[pos_x_board][col] = self.player["p1"]
        else:
            self.board[pos_x_board][col] = self.player["p2"]
        self.move_log.append(col)  # log the move so we can undo it later
        self.curr_pos = [pos_x_board, col]
        self.p1_to_move = not self.p1_to_move

    def undoMove(self):
        pos_x_board = 0
        if len(self.move_log) != 0:  # make sure that there is a move to undo
            col = self.move_log.pop()
            while self.board[pos_x_board][col] == self.EMPTY:
                pos_x_board = pos_x_board + 1

            self.board[pos_x_board][col] = self.EMPTY
            self.p1_to_move = not self.p1_to_move

    def winning(self):
        piece = self.board[self.curr_pos[0]][self.curr_pos[1]]
        piece_dict = {'◍':1,'●':-1}
        # horizontally
        for col in range(self.MAX_COLS - 3):
            for row in range(self.MAX_ROWS):
                if self.board[row][col] == piece and self.board[row][col + 1] == piece and self.board[row][
                    col + 2] == piece and self.board[row][
                    col + 3] == piece and piece != self.EMPTY:
                    return piece_dict[piece]

        # vertically
        for col in range(self.MAX_COLS):
            for row in range(self.MAX_ROWS - 3):
                if self.board[row][col] == piece and self.board[row + 1][col] == piece and self.board[row + 2][
                    col] == piece and self.board[row + 3][
                    col] == piece and piece != self.EMPTY:
                    return piece_dict[piece]

        # positive slope
        for col in range(self.MAX_COLS - 3):
            for row in range(self.MAX_ROWS - 3):
                if self.board[row][col] == piece and self.board[row + 1][col + 1] == piece and self.board[row + 2][
                    col + 2] == piece and self.board[row + 3][
                    col + 3] == piece and piece != self.EMPTY:
                    return piece_dict[piece]

        # negative slope
        for col in range(self.MAX_COLS - 3):
            for row in range(3, self.MAX_ROWS):
                if self.board[row][col] == piece and self.board[row - 1][col + 1] == piece and self.board[row - 2][
                    col + 2] == piece and self.board[row - 3][
                    col + 3] == piece and piece != self.EMPTY:
                    return piece_dict[piece]
        return 0

    def set_state(self,state,to_play):
        self.reset()
        num_X = np.sum(state[state==1])
        num_O = np.sum(state[state==-1])
        if to_play==1:
            self.p1_to_move = True
        else:
            self.p1_to_move = False
            state = -1*np.copy(state)
        for i in range(self.MAX_ROWS):
            for j in range(self.MAX_COLS):
                if state[i*self.MAX_COLS+j] == 1:
                    self.board[i][j] = '◍'
                elif state[i*self.MAX_COLS+j] == -1:
                    self.board[i][j] = '●'

        self.total_moves = np.sum(np.abs(state))  # to count the moves

    def print(self):
        print("  0     1    2     3    4    5     6 ")
        print("--------------------------------------")
        for row in range(self.MAX_ROWS):
            print(self.board[row])
