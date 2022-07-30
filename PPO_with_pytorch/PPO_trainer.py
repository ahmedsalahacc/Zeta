import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import gym
gym.logger.set_level(40) # To mute unneeded warnings
from nets import *
import cv2
from glob import glob

class Params:
    def __init__(self,env_name,solved_at=np.inf,export_video=None,num_episodes=2000,export_video_every=None):
        self.env_name = env_name                  # Name of the environment to learn
        self.solved_at = solved_at                # The rewards to stop training at when reached
        self.export_video = export_video          # Path to export video to (exported every log_episode_interval episodes)
        self.export_video_every = export_video_every # If not None, will record video every this num episodes instead of log_episode_interval
        self.num_episodes = num_episodes          # Max number of episodes to play
        self.max_steps_per_episode = 1500         # Max number of steps per episode
        self.log_episode_interval = 10            # Print some info every (log_episode_interval) episodes
        self.update_nets_period = 2048            # Update the policy and value nets using PPO every this number of steps/frames
        self.ppo_clipping = 0.2                   # PPO clipping parameter
        self.policy_lr = 0.002                    # Learning rate for the policy net
        self.value_lr = 0.002                     # Learning rate for the value net
        self.use_cuda = False                     # use_cuda=True => will use GPU for training, only recommended for CNNs
        self.ppo_training_num_epochs = 4          # Number of training epochs in the PPO train
        self.entropy_beta = 0.01                  # Factor multiplied by the entropy in the ppo_loss
        self.gamma = 0.99                         # Discount factor using in MDPs
        self.hidden_layer_size = 64               # Size of the hidden layer used in the policy and value nets
        self.action_var = 0.25                    # Variance of the action (only used for continuous action space environments)
        self.font_size = 0.8
        self.font_color = (20,20,255)
        self.font_margin = 50


class PPO_trainer:
    def __init__(self,params):
        self.params = params

        self.device = torch.device("cpu")
        if params.use_cuda:
            self.device = torch.device("cuda")

        self.env = gym.make(params.env_name)
        self.env.spec.max_episode_steps = params.max_steps_per_episode+1
        if params.export_video is not None:
            video_callable = lambda x: x%params.log_episode_interval==0
            if params.export_video_every is not None:
                video_callable = lambda x: x%params.export_video_every==0
            self.env = gym.wrappers.Monitor(self.env,params.export_video,video_callable=video_callable,force=True)

        
        
        self.state_space = np.array(self.env.observation_space.shape)

        if 'n' in self.env.action_space.__dict__: # Discrete Action Space
            self.action_space = self.env.action_space.n
            self.continuous = False
        else:                            # Continuous Action Space
            self.action_space = self.env.action_space.shape[0]
            self.action_var = torch.full((self.action_space,), self.params.action_var)
            self.cov_mat = torch.diag(self.action_var)
            self.continuous = True

        self.policy_net = Policy_Net(self.state_space,self.action_space,self.continuous,params.hidden_layer_size).to(self.device)
        self.value_net = Value_Net(self.state_space,params.hidden_layer_size).to(self.device)

        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(),lr=params.policy_lr)
        self.value_optim = torch.optim.Adam(self.value_net.parameters(),lr=params.value_lr)
        self.mse_loss = nn.MSELoss()

        self.memory = {
            'states':[],
            'rewards':[],
            'actions':[],
            'log_probs':[],
            'masks':[]
        }

        self.episode_rewards = []

    def process_state(self,state):
        if len(self.state_space)==1:
            state = torch.tensor(state).type(torch.FloatTensor).reshape(1,-1).to(self.device)
            return state
        else:
            state = torch.tensor(state.copy()).type(torch.FloatTensor)
            state = state.reshape(1,*self.state_space[[2,0,1]]).to(self.device)
            return state

    def ppo_train(self):
        returns = []
        disc_reward = 0
        for mask,reward in zip(reversed(self.memory['masks']),reversed(self.memory['rewards'])):
            disc_reward = reward + self.params.gamma*disc_reward*mask
            returns.append(disc_reward)

        returns  = torch.tensor(returns[::-1]).type(torch.FloatTensor).to(self.device)
        returns = (returns-torch.mean(returns))/(torch.std(returns)+1e-8)
        states = torch.cat(self.memory['states']).detach().type(torch.FloatTensor).to(self.device)
        actions = torch.tensor(self.memory['actions']).detach().type(torch.FloatTensor).to(self.device)
        if not self.continuous:
            actions = actions.type(torch.LongTensor).to(self.device)
            actions_hot = torch.nn.functional.one_hot(actions).to(self.device)

        log_probs = torch.tensor(self.memory['log_probs']).detach().type(torch.FloatTensor).to(self.device)

        for ppo_epoch in range(self.params.ppo_training_num_epochs):
            policy_pred = self.policy_net.forward(states)
            vals = self.value_net.forward(states).reshape(-1)

            if self.continuous:
                action_dist = MultivariateNormal(policy_pred, self.cov_mat)
                entropy = action_dist.entropy()
                new_action_logprobs = action_dist.log_prob(actions)
            else:
                action_dist = Categorical(torch.exp(policy_pred))
                entropy = action_dist.entropy()
                new_action_logprobs = torch.sum(actions_hot*policy_pred,dim=1)

            # Calculating PPO Loss
            ratios = torch.exp(new_action_logprobs-log_probs)
            advantages = returns - vals.detach()
            p1 = ratios*advantages
            p2 = torch.clamp(ratios,min=1-self.params.ppo_clipping,max=1+self.params.ppo_clipping)*advantages
            ppo_loss = -torch.min(p1,p2) - self.params.entropy_beta*entropy + 0.5*self.mse_loss(vals,returns)

            # Zero-ing out gradients of policy and value nets
            self.policy_optim.zero_grad()
            self.value_optim.zero_grad()

            # Backprop PPO loss into the policy and value nets
            torch.mean(ppo_loss).backward()

            # Update the weights of the policy and value nets
            self.policy_optim.step()
            self.value_optim.step()

        # Clear the memory after training is complete
        self.memory = {
            'states':[],
            'rewards':[],
            'actions':[],
            'log_probs':[],
            'masks':[]
        }


    def play_and_train(self,record_every=0):
        total_reward = 0
        avg_length = 0
        for e in range(1,self.params.num_episodes+1):
            episode_reward = 0
            current_state = self.process_state(self.env.reset())
            for t in range(self.params.max_steps_per_episode):
                policy_pred = self.policy_net.forward(current_state)

                if self.continuous:
                    dist = MultivariateNormal(policy_pred, self.cov_mat)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    action = action.cpu().detach().numpy().reshape(-1)
                else:
                    probs = torch.exp(policy_pred)
                    action = np.random.choice(self.action_space,p=probs.cpu().detach().numpy()[0])
                    log_prob = policy_pred[0][action]

                next_state,reward,done,_ = self.env.step(action)
                total_reward += reward
                episode_reward += reward

                # Saving episode step for later training
                self.memory['states'].append(current_state)
                self.memory['actions'].append(action)
                self.memory['log_probs'].append(log_prob)
                self.memory['rewards'].append(reward)
                self.memory['masks'].append(not done)

                # Set current state to the new state after taking an actions
                current_state = self.process_state(next_state)

                if len(self.memory['states'])%self.params.update_nets_period == 0:
                    self.ppo_train()

                if done:
                    break
            else:
                if type(self.env) is gym.wrappers.Monitor:
                    self.env.stats_recorder.done = True

            self.episode_rewards.append(episode_reward)

            avg_length += t
            if e%self.params.log_episode_interval == 0:
                avg_length = int(avg_length/self.params.log_episode_interval)
                total_reward = int((total_reward/self.params.log_episode_interval))

                print('Episode {} \t avg length: {} \t reward: {}'.format(e, avg_length, total_reward))
                if total_reward > self.params.solved_at:
                    print("Solved Game !!!!!!")
                    break
                total_reward = 0
                avg_length = 0

    def run_one_ep(self,render=False):
        current_state = self.process_state(self.env.reset())
        done = False
        episode_reward = 0
        while not done:
            if render:
                self.env.render()

            policy_pred = self.policy_net.forward(current_state)

            if self.continuous:
                dist = MultivariateNormal(policy_pred, self.cov_mat)
                action = dist.sample()
                action = action.cpu().detach().numpy().reshape(-1)
            else:
                probs = torch.exp(policy_pred)
                action = np.random.choice(self.action_space,p=probs.cpu().detach().numpy()[0])

            next_state,reward,done,_ = self.env.step(action)
            episode_reward += reward
            current_state = self.process_state(next_state)

        return episode_reward

    def evaluate_agent(self,num_episodes=10,render=False):
        ep_rewards = [self.run_one_ep(render) for e in range(num_episodes)]
        return ep_rewards

    def combined_episode_videos(self,to_save_at):
        video_names = np.asarray(glob(self.params.export_video+'/*.mp4'))
        if video_names.shape[0]==0: return
        cap = cv2.VideoCapture(video_names[0])
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(to_save_at+'/combined_{}.mp4'.format(self.params.env_name),cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width,frame_height))
        for idx,video in enumerate(video_names):
            episode_idx = idx*self.params.log_episode_interval
            cap = cv2.VideoCapture(video)
            counter = 0
            while(True):
                font_size = self.params.font_size*(frame_width/600)
                if font_size<0.5:
                    font_size = 0.5
                margin = int(self.params.font_margin/600*frame_width)
                # Capture frames in the video
                ret, frame = cap.read()

                if not ret:
                    break
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame,'Episode: {}'.format(episode_idx+1),(margin, margin),font, font_size,self.params.font_color,2, cv2.LINE_4)
                cv2.putText(frame,'Reward: {:.2f}'.format(self.episode_rewards[episode_idx]), (margin, frame_height-margin), font, font_size,self.params.font_color, 2,cv2.LINE_4)
                out.write(frame)
                counter += 1

            cap.release()
        out.release()

    def save_nets(self,pth_name):
        torch.save(self.policy_net.state_dict(),pth_name.format("policy_net"))
        torch.save(self.value_net.state_dict(),pth_name.format("value_net"))
