from MCTS import *
import torch
import torch.nn as nn

class Trainer:
    def __init__(self,env,num_sims,eps_gen_per_iter):
        self.game = env()
        self.MC = MonteCarlo(board=env(),num_simulations=num_sims) # Make sure the board in MC is different from self.game
        self.policy_net = nn.Sequential(nn.Linear(self.game.state_space,32),
                          nn.ReLU(),
                          nn.Linear(32,32),
                          nn.ReLU(),
                          nn.Linear(32,self.game.action_space),
                          nn.Softmax(dim=1))

        self.value_net = nn.Sequential(nn.Linear(self.game.state_space,32),
                          nn.ReLU(),
                          nn.Linear(32,32),
                          nn.ReLU(),
                          nn.Linear(32,1),
                         nn.Tanh())
        self.value_optim = torch.optim.Adam(self.value_net.parameters(),lr=1e-2)
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(),lr=1e-2)
        self.eps_gen_per_iter = eps_gen_per_iter
        self.Val_loss = nn.MSELoss()
        self.NL = nn.NLLLoss()
        self.batch_size = 64
        
    def generate_episode(self,verbose=False):
        '''
        Generate an episode of data (state,action,reward) for each time step of a full game
        The actions are determined using the policy network, value network, and MCTS
        
        Arguments:
            verbose(bool): prints logging info if True
            
        Returns:
            states_data(torch.FloatTensor): shape=(N,self.game.state_space)
            action_probs_data(torch.FloatTensor): shape=(N,self.game.action_space)
            rewards_data(torch.FloatTensor): shape=(N,1)
            
                Where N is the number of timesteps in the episode
            
        '''
        self.game.reset()
        current_state = self.game.game_state()[0]
        to_play = 1
        train_data = []
        
        while True:
            tree = self.MC.play_and_search(self.value_net,self.policy_net,current_state,to_play,verbose=False)
            
            action_probs = np.zeros(self.game.action_space)
            for action,child in tree.children.items():
                action_probs[action] += child.visit_count
            
            action_probs *= self.game.get_valid_actions()
            
            action_probs /= np.sum(action_probs)
            best_action = tree.select_action()
            
            if verbose: 
                print(current_state)
                print(action_probs)
                print(best_action)
            train_data.append((current_state,to_play,action_probs))
            
            self.game.reset()
            self.game.set_state(current_state,to_play)
            current_state,reward,done,_ = self.game.take_action(best_action)
            to_play *= -1
            
            if done:
                ret = []
                for hist_state, hist_current_player, hist_action_probs in train_data:
                    ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != to_play))))

                return ret
            
    def train(self,num_iters,n_epochs):
        policy_losses =[]
        value_losses = []
        for itr in range(num_iters):
            train_data = []
            
            # Generate training data and expand replay memory bank
            for eps in tqdm(range(self.eps_gen_per_iter),position=0,leave=True):
                train_data.extend(self.generate_episode())
            
            # Training loops
            for e in range(n_epochs):
                self.policy_net.train()
                self.value_net.train()
                
                for b in range(len(train_data)//self.batch_size):
                    random_indices = np.random.randint(len(train_data), size=self.batch_size)
                    states, policy_probs, values = list(zip(*[train_data[i] for i in random_indices]))
                    states = torch.from_numpy(np.array(states)).type(torch.FloatTensor)
                    policy_probs = torch.from_numpy(np.array(states)).type(torch.FloatTensor)
                    values = torch.from_numpy(np.array(states)).type(torch.FloatTensor)

                    # Forward through policy and values networds
                    policy_pred = self.policy_net.forward(states)
                    value_pred = self.value_net.forward(states)
                    
                    # Policy and Values Losses
                    policy_loss = torch.mean(-(policy_probs * torch.log(policy_pred)).sum(dim=1))
                    value_loss = torch.sum((values-value_pred.reshape(-1))**2)/values.shape[0]

                    # Update Policy Network
                    self.policy_optim.zero_grad()
                    policy_loss.backward()
                    self.policy_optim.step()
                    
                    # Update Value Network
                    self.value_optim.zero_grad()
                    value_loss.backward()
                    self.value_optim.step()
                    value_losses.append(value_loss)
                    policy_losses.append(policy_loss)
                    
                
                print("Policy loss: {:.2f}".format(policy_loss))
                print("Value loss: {:.2f}".format(value_loss))
                
        self.policy_net.eval()
        self.value_net.eval()
        return policy_losses, value_losses
            