import numpy as np
import math
import torch
from tqdm import tqdm
import random

#*** Node ***#
class Node:
    def __init__(self,prior,to_play):
        self.state = None       # State that describes the board
        self.prior = prior      # Prior probability of selecting this node from its parent and among its siblings
        self.to_play = to_play  # Whose player turn is it (1 or -1)
        self.children = {}      # Dictionary that holds the lookup of legal states (children) positions
        self.visit_count = 0    # Record of the number of times that this node (state) was visited during the search (MCTS)
        self.values_sum = 0     # The total value of this state from the visits during search (MCTS)
        
    def isExpanded(self):
        '''
        Check whether the state's children are added or not (check the expansion)
        --------------------------------------------------------------------------------
        -Return-
        expanded : boolean that describes whether the node has children (1) or not (0)
        '''
        return len(self.children)>0
    
    def getValue(self):
        '''
        -Return-
        value : the value of the node 
        '''
        if self.visit_count == 0: return 0
        return self.values_sum / self.visit_count
   
    def select_action(self,temperature=1):
        '''
        selects the child with the highest N
        --------------------------------------------------------------------------------
        -Return-
        best_child  : best child selected with the highest UCB value
        best_action : best state with t
        '''
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action
    
    def select_action_val(self,verbose=False):
        '''
        selects the child with the highest value
        --------------------------------------------------------------------------------
        -Return-
        best_child  : best child selected with the highest UCB value
        best_action : best state with t
        '''
        action_probs = np.zeros(9)
        for action,child in self.children.items():
            action_probs[action] = child.getValue()
            
        return (np.argmax(action_probs))

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
            if score > best_score:
                best_score, best_action, best_child = score, action, child
            
        return best_child, best_action

    def expand(self,state,to_play,action_probs):
        '''
        expand node and keep track of the prior policy prob given by the neural network
        --------------------------------------------------------------------------------
        -Arguments-
            state        : state to expand
            to_play      : players turn (either 1 or -1)
            action_probs : prior policy probabilities given by the policy network
        '''
        self.state = state
        self.play = to_play

        for p,prob in  enumerate(action_probs):
            if prob > 1e-6:
                self.children[p] = Node(prior = prob, to_play = -1*self.to_play)

    def _get_ucb_score(self,parent, child):
        '''
        Calculates the UCB score of the action that will cause the transition from parent to child
        --------------------------------------------------------------------------------
        -Arguments-
            parent       : Parent Node
            child        : Child Node to calculate score

        -Return-
        score            : UCB score of the action
        '''
        if parent.visit_count==0: return child.prior
        
        prior_score = child.prior*math.sqrt(parent.visit_count)/(child.visit_count+1)
        
        value_score = - 1*child.getValue()
        
        score = value_score + prior_score
        return score

    def __str__(self):
        return f"*-- Prior: {self.prior} Value: {self.values_sum} Visits Count: {self.visit_count}"

class MonteCarlo:
    def __init__(self,board,num_simulations=50):
        self.num_simulations = num_simulations   # number of simulations
        self.board = board                       # board Object

    def play_and_search(self,value_net,policy_net,state,to_play,verbose=False):
        '''
        Searches for the best strategy (heuristically) to play the game 
        --------------------------------------------------------------------------------
        -Arguments-
            value_net         : Value network
            policy_net        : Policy network
            state             : Starting state
            to_play           : Player's turn (either 1 or -1)
            verbose           : boolean, if True logging data is printed

        -Return-
            root              : root node with the tree built from it
        '''
        self.board.reset()
        self.board.set_state(state,to_play)
        root = Node(0, to_play)

        # Expansion of the root node
        self._expand(root,value_net,policy_net,state,to_play)
        if verbose:
            print(root.children)
            
        for i in tqdm(range(self.num_simulations),position=0,leave=True,disable=not verbose):
            node = root
            search_path = [node]

            # Selection
            if verbose:
                print("---------------------  Started Selection ----------------------------")
            while node.isExpanded():
                node, action = node.select_child(verbose)
                search_path.append(node)

            # Expansion
            level_1_parent = search_path[-2]
            state = level_1_parent.state
            
            self.board.reset()
            self.board.set_state(state,level_1_parent.to_play)
            next_state, reward, done, _ = self.board.take_action(action)
            
            if verbose:
                print(level_1_parent.to_play)
                print(self.board.grid)
                print("Done: {}, Reward: {}".format(done,reward))
            
            if not done: # Expansion if not terminal node
                reward = self._expand(node,value_net,policy_net,next_state,-1*level_1_parent.to_play)
                
            if verbose:
                print("Reward/Value: {}".format(reward))
            
            # back Propagation
            self._backpropagate(search_path[::-1],reward,-1*level_1_parent.to_play)
            
        return root

    def _expand(self,node,value_net,policy_net,state,to_play):
        '''
        Expands given node according to the value network
        --------------------------------------------------------------------------------
        Args:
            root              : Node to expand
            value_net         : Value network
            policy_net        : Policy network
            state             : Starting state
            to_play           : Player's turn  (either 1 or -1)
         
        Returns:
            value (float)     : estimated value using the value network
        '''
        state_tensor = torch.tensor(state).type(torch.FloatTensor).reshape(1,-1)
        action_probs = policy_net.forward(state_tensor).reshape(-1).detach().cpu().numpy()
        value = value_net.forward(state_tensor).detach().cpu().numpy()[0][0]
        valid_moves = self.board.get_valid_actions()
        action_probs = action_probs*valid_moves # mask the invalid moves in the action probs
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
        current_node.visit_count += 1
        
        if len(search_path)>1:
            self._backpropagate(search_path[1:],winner,to_play)
        
