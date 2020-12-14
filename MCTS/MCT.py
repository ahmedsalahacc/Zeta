'''*
/*Auto Generated*/
    * @author Ahmed Salah
    * @email AhmedSalahEldin125@gmail.com
    * @create date 2020-12-12 09:24:50
    * @modify date 2020-12-12 09:24:50
    * @desc Monte Carlo Tree Search
*'''
#from board import Board
import numpy as np
import math
import torch
from tqdm import tqdm
#*** Node ***#
class Node:
    def __init__(self,prior,to_play):
        self.state = None       # State that describes the board
        self.prior = prior      # Prior probability of selecting this node from its parent and among its siblings
        self.to_play = to_play  # Whose player turn is it (1 or -1)
        self.children = {}      # Dictionary that holds the lookup of legal states (children) positions
        self.visit_count = 0    # Record of the number of times that this node (state) was visited during the search (MCTS)
        self.values_sum = 0     # The total value of this state from the visits during search (MCTS)
        self.is_terminal = False
    def isExpanded(self):
        '''
        Check whether the state's children are added or not (check the expansion)
        --------------------------------------------------------------------------------
        -Return-
        expanded : boolean that describes whether the node has children (1) or not (0)
        '''
        return len(self.children)>0

    ########################### to Implement ###########################
    def getValue(self):
        '''
        -Return-
        value : the value of the node 
        '''
        if self.visit_count == 0: return 0
        return self.values_sum / self.visit_count

    def select_action(self):
        '''
        selects next action
        '''
        best_action = -1
        best_score = -np.inf
        best_child = None

        # Selection
        for action, child in self.children.items():
            score = self._get_ucb_score(self,child)
            if score > best_score:
                best_score, best_action, best_child = score, action, child
            
        return best_action
    ###################################################################

    def select_child(self,verbose=False):
        '''
        selects the child with the highest UCB score
        --------------------------------------------------------------------------------
        -Return-
        best_child  : best child selected with the highest UCB value
        best_action : best state with t
        '''
        best_action = -1
        best_score = -np.inf
        best_child = None

        # Selection
        for action, child in self.children.items():
            score = self._get_ucb_score(self,child)
            if verbose:
                print("Action: {}, UCB_Score: {:.4f}".format(action,score))
            if score > best_score and not child.is_terminal:
                best_score, best_action, best_child = score, action, child
            
        return best_child, best_action

    def expand(self,state,to_play,action_probs):
        '''
        expand node and keep track of the prior policy prob given by the neural network
        --------------------------------------------------------------------------------
        -Arguments-
            state        : state to expand
            to_play      : players turn
            action_probs : prior policy probabilities given by the neural network
        '''
        self.state = state
        self.play = to_play

        # Expansion
        for p,prob in  enumerate(action_probs):
            if prob != 0:
                self.children[p] = Node(prior = prob, to_play = -1*self.to_play)

    def _get_ucb_score(self,parent, child):
        '''
        Calculates the UCB score of the action that will cause the transision from parent to child
        --------------------------------------------------------------------------------
        -Arguments-
            parent       : Parent Node
            child        : Child Node to calculate score

        -Return-
        score            : UCB score of the action
        '''
        if parent.visit_count==0: return child.prior
        prior_score = child.prior*math.sqrt(parent.visit_count)/(child.visit_count+1)
        
        if child.visit_count >0:
            value_score = - 1*child.getValue()
        else:
            value_score = 0
        
        
        score = value_score + prior_score
        return score
    def update_terminal(self):
        if not self.is_terminal and self.isExpanded():
            self.is_terminal = True
            for action,child in self.children.items():
                if not child.is_terminal:
                    self.is_terminal = False
                    break

    def __str__(self):
        return f"*-- Prior: {self.prior} Value: {self.values_sum} Visits Count: {self.visit_count}"

class MonteCarlo:
    def __init__(self,**kwargs):
        self.num_simulations = kwargs.get('num_simulations',50)   # number of simulations
        self.board = kwargs.get('board',None)                     # board Object
        self.board.reset()

    def play_and_search(self,value_net,policy_net,state, to_play,verbose=False):
        '''
        Searches for the best strategy (heuristically) to play the game 
        --------------------------------------------------------------------------------
        -Arguments-
            value_net    : Value network
            policy_net   : Policy network
            state        : Starting state
            to_play      : Player's turn

        -Return-
        root             : root node updated with the best strategy of the given state
        '''
        root = Node(0, to_play)

        # Expansion
        self._expand(root,value_net,policy_net,state,to_play)
        if verbose:
            print(root.children)
        for i in tqdm(range(self.num_simulations),position=0,leave=True):
            node = root
            search_path = [node]

            # selection
            if verbose:
                print("---------------------  Started Selection ----------------------------")
            while node.isExpanded():
                node, action = node.select_child(verbose)
                search_path.append(node)
                if node is None:
                    return root
                if verbose:
                    print(node)
                    print(node.children)
                    print(action)

            # Expansion
            level_1_parent = search_path[-2]
            state = level_1_parent.state
            self.board.set_state(state,level_1_parent.to_play)
            self.board.take_action(action)
            next_state, reward, done, _ = self.board.game_state()
            if verbose:
                print(level_1_parent.to_play)
                print(self.board.grid)
                print("Done: {}, Reward: {}".format(done,reward))
            
            if not done:
                # Do Expansion
                reward = self._expand(node,value_net,policy_net,next_state,-1*level_1_parent.to_play)
            else: 
                node.is_terminal = True # Terminal node
                node.state = next_state
            if verbose:
                print("Reward/Value: {}".format(reward))
            
            # back Propagation
            self._backpropagate(search_path[::-1],reward,-1*level_1_parent.to_play)
            
        return root

    def _expand(self,node,value_net,policy_net,state,to_play):
        '''
        Expands given node according to the value network
        --------------------------------------------------------------------------------
        -Arguments-
            root         : Node to expand
            model        : Value neural network
            state        : Starting state
            to_play      : Player's turn

        -Returns-
        value            : value to update
        '''
        # expand root state
        state_tensor = torch.tensor(state).type(torch.FloatTensor).reshape(1,-1)
        action_probs = policy_net.forward(state_tensor).reshape(-1).detach().cpu().numpy()
        value = value_net.forward(state_tensor).detach().cpu().numpy()[0][0]
        valid_moves = self.board.get_valid_actions()
        ## mask the inaccurate probs in the action probs
        action_probs = action_probs*valid_moves
        action_probs /= np.sum(action_probs)
        node.expand(state, to_play, action_probs)
        return value

    def _backpropagate(self,search_path,winner,to_play):
        '''
        Recursive back propagation reaching the root of the tree
        --------------------------------------------------------------------------------
        -Arguments-
            search_path         : Node to expand
            winner              : the winner player (1, -1, or 0)
            to_play             : Player's turn
        '''
        current_node = search_path[0]
        if current_node.to_play == to_play:
            current_node.values_sum += winner
        else:
            current_node.values_sum -= winner
        current_node.visit_count+=1
        current_node.update_terminal()
        if len(search_path)>1:
            self._backpropagate(search_path[1:],winner,to_play)
