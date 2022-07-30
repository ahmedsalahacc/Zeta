# Our Libraries
from nets import *
from NN import *

# Necessary Libraries
import numpy as np    # Linear algebra library
import gym            # Contains environments used to train

# Libraries only used for visualization, logging, and plotting (can be removed with their code without affecting the learning agent in any way)
from tqdm import tqdm # Used to create loading bars while training
import cv2 # Used only to save videos of the agent learning and is not used to do any computer vision
from glob import glob # Used to read file names and open folders
gym.logger.set_level(40) # To mute unneeded warnings


class Params:
    def __init__(self,env_name,solved_at=np.inf,export_video=None,num_episodes=2000,export_video_every=None):
        self.env_name = env_name                  # Name of the environment to learn
        self.solved_at = solved_at                # The rewards to stop training at when reached
        self.export_video = export_video          # Path to export video to (exported every log_episode_interval episodes)
        self.export_video_every = export_video_every # If not None, will record video every this num episodes instead of log_episode_interval
        self.num_episodes = num_episodes          # Max number of episodes to play
        self.max_steps_per_episode = 1500         # Max number of steps per episode
        self.log_episode_interval = 10            # Print some info every (log_episode_interval) episodes
        self.update_nets_period = 512           # Update the policy and value nets using PPO every this number of steps/frames
        self.ppo_clipping = 0.2                   # PPO clipping parameter
        self.policy_lr = 0.02                    # Learning rate for the policy net
        self.value_lr = 0.02                     # Learning rate for the value net
        self.use_cuda = False                     # use_cuda=True => will use GPU for training, only recommended for CNNs
        self.ppo_training_num_epochs = 4          # Number of training epochs in the PPO train
        self.entropy_beta = 0.01                  # Factor multiplied by the entropy in the ppo_loss
        self.gamma = 0.99                         # Discount factor using in MDPs
        self.hidden_layer_size = 32               # Size of the hidden layer used in the policy and value nets
        self.action_var = 0.25                    # Variance of the action (only used for continuous action space environments)
        self.font_size = 0.8
        self.font_color = (20,20,255)
        self.font_margin = 50


class PPO_trainer:
    def __init__(self,params):
        self.params = params
        
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
            self.action_var = np.full((self.action_space,), self.params.action_var)
            self.cov_mat = np.diag(self.action_var)
            self.continuous = True

        self.policy_net = Policy_Net(self.state_space,self.action_space,params.policy_lr,self.continuous,params.hidden_layer_size)
        self.value_net = Value_Net(self.state_space,params.value_lr,params.hidden_layer_size)
        
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
            return np.array(state,dtype=np.float).reshape(-1,self.state_space[0])
        else:
            state = np.array(state,dtype=np.float).reshape(1,*self.state_space[[2,0,1]])
            return state

    def ppo_train(self):
        returns = []
        disc_reward = 0
        for mask,reward in zip(reversed(self.memory['masks']),reversed(self.memory['rewards'])):
            disc_reward = reward + self.params.gamma*disc_reward*mask
            returns.append(disc_reward)

        returns  = np.array(returns[::-1],dtype=np.float)
        returns = (returns-np.mean(returns))/(np.std(returns)+1e-7)
        states = np.array(self.memory['states']).reshape(-1,self.state_space[0])
        actions = np.array(self.memory['actions'])
        actions_hot = np.ones((actions.shape[0],self.action_space))
        if not self.continuous:
            actions = np.array(actions,dtype=np.int)
            actions_hot = np.eye(self.action_space)[actions.reshape(-1)]

        log_probs = np.array(self.memory['log_probs']).reshape(-1,1) # Shape = (batch_size,1)
        for ppo_epoch in range(self.params.ppo_training_num_epochs):
            # Zero-ing out gradients of policy and value nets
            self.policy_net.zero_grad()
            self.value_net.zero_grad()
            
            policy_pred = self.policy_net.forward(states) # Shape = (batch_size,self.action_space)
            vals = self.value_net.forward(states) # Shape = (batch_size,1)

            if self.continuous:
                new_action_logprobs = calc_log_prob_multi_normal(policy_pred,self.cov_mat,actions).reshape(-1,1)
                back_entropy = 0
            else:
                new_action_logprobs = np.sum(actions_hot*np.log(policy_pred),axis=1).reshape(-1,1) # Shape = (batch_size,1)
                back_entropy = (np.log(policy_pred)+1)*self.params.entropy_beta # Shape = (batch_size,action_space)
                
            # Calculating PPO Loss
            ratios = np.exp(new_action_logprobs-log_probs) # Shape = (batch_size,1)
            advantages = returns.reshape(-1,1) - vals # Shape = (batch_size,1)
            
            dratios_dp = (ratios*actions_hot)/(policy_pred+1e-7) # Shape = (batch_size,action_space)
            p1 = ratios*advantages # Shape = (batch_size,1)
            dp1_dp = dratios_dp*advantages # Shape = (batch_size,action_space)
            p2 = np.clip(ratios,1-self.params.ppo_clipping,1+self.params.ppo_clipping)*advantages # Shape = (batch_size,1)
            dp2_dp = dratios_dp*advantages*np.logical_and(ratios>1-self.params.ppo_clipping, ratios<1+self.params.ppo_clipping) # Shape = (batch_size,action_space)
            is_p1 = np.array(p1 <= p2,dtype=np.int) # Shape = (batch_size,1)
            dppo_dp = back_entropy-dp1_dp*is_p1-dp2_dp*(1-is_p1)
            
            # Remove outliers from advatanges and dppo_dp to prevent overflow gradients
            advantages[np.abs(advantages)>1e2] = 0
            dppo_dp[np.abs(dppo_dp)>1e2] = 0

            # Backprop PPO loss into the policy and value nets
            self.value_net.backward(-1e4*advantages/advantages.shape[0])
            self.policy_net.backward(1e4*dppo_dp/(dppo_dp.shape[0]))
            
            # Optimize the policy and value networks
            #print(np.linalg.norm(self.policy_net.layers[0].dW))
            self.policy_net.optimize()
            self.value_net.optimize()

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
                    action = np.random.multivariate_normal(policy_pred[0], self.cov_mat,1)[0]
                    log_prob = calc_log_prob_multi_normal(policy_pred[0],self.cov_mat,action[0])
                else:
                    action = np.random.choice(self.action_space,p=policy_pred[0]/np.sum(policy_pred[0]))
                    log_prob = np.log(policy_pred[0][action])

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
                action = np.random.multivariate_normal(policy_pred[0], self.cov_mat,1)[0]
            else:
                action = np.random.choice(self.action_space,p=policy_pred[0]/np.sum(policy_pred[0]))

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
        
        
def _batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.shape[-1]
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = len(bL.shape) - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (list(range(outer_batch_dims)) +
                    list(range(outer_batch_dims, new_batch_dims, 2)) +
                    list(range(outer_batch_dims + 1, new_batch_dims, 2)) +
                    [new_batch_dims])
    bx = bx.transpose(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.shape[0], n)  # shape = c x b x n
    flat_x_swap = flat_x.transpose(1, 2, 0)  # shape = b x n x c
    M_swap = np.sum(np.power(np.linalg.solve(flat_L,flat_x_swap)[0],2),axis=-2)
    M = M_swap.T  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.transpose(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)

def calc_log_prob_multi_normal(mean, covariance_matrix, action):
    diff = action - mean
    M = _batch_mahalanobis(np.linalg.cholesky(covariance_matrix), diff)
    half_log_det = np.log(np.diag(np.linalg.cholesky(covariance_matrix))).sum()
    return -0.5 * (covariance_matrix.shape[0] * np.log(2 * np.pi) + M) - half_log_det
