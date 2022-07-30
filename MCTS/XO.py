import numpy as np

class Environment:
    def __init__(self):
        self.grid = {
            '0': '0', '1': '1', '2': '2',
            '3': '3', '4': '4', '5': '5',
            '6': '6', '7': '7', '8': '8'
        }
        self.default_grid = {
            '0': '0', '1': '1', '2': '2',
            '3': '3', '4': '4', '5': '5',
            '6': '6', '7': '7', '8': '8'
        }
        
        self.state_space = (9)
        self.action_space = (9)
        
        self.player = 1  # to initialise first player
        self.total_moves = 0  # to count the moves
        
    def grid_to_state(self,grid):
        move_dict = {'X':1,'O':-1}  # Default is player == 1, X will be 1, O will be -1
        state = np.zeros(9)
        for i in range(9):
            if self.grid[str(i)] != self.default_grid[str(i)]:
                state[i] = move_dict[grid[str(i)]]
                
        if self.player==2: # X will be -1, O will be 1
            state *= -1
            
        return state
            
    def game_state(self):
        end_check = self.check()
        if end_check == 1:
            observation = self.grid.copy()
            reward = 1
            done = True
            info = ""
            return self.grid_to_state(observation), reward, done, info
        if end_check == 2:
            observation = self.grid.copy()
            reward = -1
            done = True
            info = ""
            return self.grid_to_state(observation), reward, done, info
        if self.total_moves == 9:
            # print("Draw!")
            observation = self.grid.copy()
            reward = 0
            done = True
            info = ""
            return self.grid_to_state(observation), reward, done, info

        observation = self.grid.copy()
        reward = 0
        done = False
        info = ""
        return self.grid_to_state(observation), reward, done, info
    
    def get_valid_actions(self):
        valid = np.ones(9)
        for i in range(9):
            if self.grid[str(i)] != self.default_grid[str(i)]:
                valid[i] = 0
        return valid

    def take_action(self, action):

        if self.player == 1:  # choose player
            p1_input = str(action)
            if p1_input.upper() in self.grid and self.grid[p1_input.upper()] != 'X' and self.grid[p1_input.upper()] != 'O':
                self.grid[p1_input.upper()] = 'X'
                self.player = 2
                self.total_moves += 1
                return self.game_state()
            # on wrong input
            else:
                print("Invalid move by p1")
                observation = None
                reward = None
                done = None
                info = "Invalid Move"
                return observation, reward, done, info

        if self.player == 2:  # choose player
            p2_input = str(action)
            if p2_input.upper() in self.grid and self.grid[p2_input.upper()] != 'X' and self.grid[p2_input.upper()] != 'O':
                self.grid[p2_input.upper()] = 'O'
                self.player = 1
                self.total_moves += 1

                return self.game_state()

            # on wrong input
            else:
                print("Invalid move by p2")
                observation = None
                reward = None
                done = None
                info = "Invalid Move"
                return observation, reward, done, info

    def reset(self):
        self.grid = self.default_grid.copy()
        self.total_moves = 0  # to count the moves
        
    def set_state(self,state,to_play):
        self.reset()
        num_X = np.sum(state[state==1])
        num_O = np.sum(state[state==-1])
        if to_play==1:
            self.player = 1
        else:
            self.player = 2
            state = -1*np.copy(state)
        for i in range(9):
            if state[i] == 1:
                self.grid[str(i)] = 'X'
            elif state[i] == -1:
                self.grid[str(i)] = 'O'
        
        self.total_moves = np.sum(np.abs(state))  # to count the moves

    def print(self):
        print(self.grid['0'] + '|' + self.grid['1'] + '|' + self.grid['2'])
        print('-+-+-')
        print(self.grid['3'] + '|' + self.grid['4'] + '|' + self.grid['5'])
        print('-+-+-')
        print(self.grid['6'] + '|' + self.grid['7'] + '|' + self.grid['8'])

    def check(self):
        # checking the moves of player one
        # for horizontal(start)
        if self.grid['0'] == 'X' and self.grid['1'] == 'X' and self.grid['2'] == 'X':
            return 1
        if self.grid['3'] == 'X' and self.grid['4'] == 'X' and self.grid['5'] == 'X':
            return 1
        if self.grid['6'] == 'X' and self.grid['7'] == 'X' and self.grid['8'] == 'X':
            return 1
        # for horizontal(end)
        # for diagonal(start)
        if self.grid['0'] == 'X' and self.grid['4'] == 'X' and self.grid['8'] == 'X':
            return 1
        if self.grid['2'] == 'X' and self.grid['4'] == 'X' and self.grid['6'] == 'X':
            return 1
        # for diagonal(end)
        # for vertical(start)
        if self.grid['0'] == 'X' and self.grid['3'] == 'X' and self.grid['6'] == 'X':
            return 1
        if self.grid['1'] == 'X' and self.grid['4'] == 'X' and self.grid['7'] == 'X':
            return 1
        if self.grid['2'] == 'X' and self.grid['5'] == 'X' and self.grid['8'] == 'X':
            return 1
        # for vertical(end)

        # checking the moves of player two
        if self.grid['0'] == 'O' and self.grid['1'] == 'O' and self.grid['2'] == 'O':
            return 2  # used to end the game
        if self.grid['3'] == 'O' and self.grid['4'] == 'O' and self.grid['5'] == 'O':
            return 2
        if self.grid['6'] == 'O' and self.grid['7'] == 'O' and self.grid['8'] == 'O':
            return 2
        if self.grid['0'] == 'O' and self.grid['4'] == 'O' and self.grid['8'] == 'O':
            return 2
        if self.grid['2'] == 'O' and self.grid['4'] == 'O' and self.grid['6'] == 'O':
            return 2
        if self.grid['0'] == 'O' and self.grid['3'] == 'O' and self.grid['6'] == 'O':
            return 2
        if self.grid['1'] == 'O' and self.grid['4'] == 'O' and self.grid['7'] == 'O':
            return 2
        if self.grid['2'] == 'O' and self.grid['5'] == 'O' and self.grid['8'] == 'O':
            return 2
        return 0

'''
game1 = Environment()

game1.print()
while True:
    if game1.player == 1:
        action = input('player one ')
        _,r,d,_= game1.take_action(action)
        print(r)
        print(d)
        game1.print()
        if d: break
    if game1.player == 2:
        action = input('player two ')
        _, r, d, _ = game1.take_action(action)
        print(r)
        print(d)
        game1.print()
        if d: break

'''