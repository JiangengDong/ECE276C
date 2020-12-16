""" Learn a policy using TD3 for the reach task"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gym
from juggle_env import JuggleEnv

from typing import Dict, Tuple
from tqdm import tqdm
import glob
import copy
import imageio
import time
import cv2


def weighSync(target_model: torch.nn.Module, source_model: torch.nn.Module, tau: float = 0.001) -> None:
    tau_2 = 1 - tau
    for (source_param, target_param) in zip(source_model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * source_param.data + tau_2 * target_param.data)


class Replay:
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """
        self.states = np.zeros(shape=(buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(shape=(buffer_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros(shape=(buffer_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros(shape=(buffer_size,), dtype=np.float32)
        self.dones = np.zeros(shape=(buffer_size,), dtype=np.float32)

        # circular queue
        self.size = 0
        self.buffer_size = buffer_size
        self.next_sample_index = 0

    def buffer_add(self, exp: Dict) -> None:
        """
        A function to add a dictionary to the buffer
        param: exp : A dictionary consisting of state, action, reward , next state and done flag
        """
        if exp["states"].ndim > 1:
            raise RuntimeError("Please feed one entry at a time")

        index = self.next_sample_index
        self.states[index] = exp["states"]
        self.actions[index] = exp["actions"]
        self.rewards[index] = exp["rewards"]
        self.next_states[index] = exp["next_states"]
        self.dones[index] = exp["dones"]

        self.next_sample_index = (index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def buffer_sample(self, n: int) -> Dict:
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        indices = np.random.choice(self.size, n)
        return {"states": self.states[indices],
                "actions": self.actions[indices],
                "rewards": self.rewards[indices], # unnormalized reward
                "next_states": self.next_states[indices],
                "dones": self.dones[indices]}


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        super(Actor, self).__init__()

        hidden_dim_1 = 256
        hidden_dim_2 = 256

        self.fc1 = nn.Linear(state_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        # self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.fc4 = nn.Linear(hidden_dim_2, action_dim)

        self.fc1.weight.data.uniform_(-1 / np.sqrt(state_dim), 1 / np.sqrt(state_dim))
        self.fc2.weight.data.uniform_(-1 / np.sqrt(hidden_dim_1), 1 / np.sqrt(hidden_dim_1))
        # self.fc3.weight.data.uniform_(-1 / np.sqrt(hidden_dim_2), 1 / np.sqrt(hidden_dim_2))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass
        param: state: The state of the environment
        """
        x = state
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        x = self.fc4(x)
        x = torch.tanh(x) # TODO: remove this layer

        return x

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        super(Critic, self).__init__()

        hidden_dim = 256
        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
            # nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the critic
        """
        x = torch.cat([state, action], dim=-1)
        return self.net.forward(x)

def concatenate_state(state: Dict):
    return np.concatenate(
        [state["robot0_joint_pos"], 
        state["robot0_joint_vel"], 
        state["robot0_eef_pos"], 
        state["robot0_eef_quat"], 
        state["pingpong_pos"]]
    )

class TD3:
    def __init__(
            self,
            train_env: gym.Env,
            test_env: gym.Env, 
            action_dim: int,
            state_dim: int,
            critic_lr: float = 3e-4,
            actor_lr: float = 3e-4,
            gamma: float = 0.99,
            batch_size: int = 100,
    ):
        """
        param: env: An gym environment
        param: action_dim: Size of action space
        param: state_dim: Size of state space
        param: critic_lr: Learning rate of the critic
        param: actor_lr: Learning rate of the actor
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = train_env
        self.evaluate_env = test_env

        # Create a actor and actor_target
        self.actor = Actor(state_dim, action_dim).cuda()
        self.actor_target = Actor(state_dim=state_dim, action_dim=action_dim).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.requires_grad_(False)

        # Create a critic and critic_target object
        self.critic1 = Critic(state_dim, action_dim).cuda()
        self.critic1_target = Critic(state_dim=state_dim, action_dim=action_dim).cuda()
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_target.requires_grad_(False)
        self.critic2 = Critic(state_dim, action_dim).cuda()
        self.critic2_target = Critic(state_dim=state_dim, action_dim=action_dim).cuda()
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_target.requires_grad_(False)

        # Define the optimizer for the actor
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # Define the optimizer for the critic
        self.optimizer_critic = torch.optim.Adam(list(self.critic1.parameters())+list(self.critic2.parameters()), lr=critic_lr)

        # define a replay buffer
        self.ReplayBuffer = Replay(buffer_size=100000, state_dim=state_dim, action_dim=action_dim)

    def update_target_networks(self) -> None:
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor)
        weighSync(self.critic1_target, self.critic1)
        weighSync(self.critic2_target, self.critic2)

    def update_critic(self) -> torch.Tensor:
        """
        A function to update the function just once
        """
        # sample from replay buffer and unpack
        batch = self.ReplayBuffer.buffer_sample(self.batch_size)
        states = torch.from_numpy(batch["states"]).cuda()
        actions = torch.from_numpy(batch["actions"]).cuda()
        next_states = torch.from_numpy(batch["next_states"]).cuda()
        rewards = torch.from_numpy(batch["rewards"]).cuda()
        dones = torch.from_numpy(batch["dones"]).cuda()

        # train critic
        next_actions = self.actor_target.forward(next_states)
        epsilon = torch.clamp(torch.randn_like(next_actions) * 0.1, -0.2, 0.2)
        next_actions = torch.clamp(next_actions + epsilon, -1.0, 1.0)
        Q1_next = self.critic1_target.forward(next_states, next_actions).squeeze()
        Q2_next = self.critic2_target.forward(next_states, next_actions).squeeze()
        Q_next = torch.min(Q1_next, Q2_next)
        Q_backup = rewards + self.gamma * (1 - dones) * Q_next
        Q1_predict = self.critic1.forward(states, actions).squeeze()
        Q2_predict = self.critic2.forward(states, actions).squeeze()
        critic_loss = F.mse_loss(Q1_predict, Q_backup) + F.mse_loss(Q2_predict, Q_backup)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        return critic_loss.detach().cpu()

    def update_actor(self) -> torch.Tensor:
        # sample from replay buffer and unpack
        batch = self.ReplayBuffer.buffer_sample(self.batch_size)
        states = torch.from_numpy(batch["states"]).cuda()
        actions = torch.from_numpy(batch["actions"]).cuda()
        next_states = torch.from_numpy(batch["next_states"]).cuda()
        rewards = torch.from_numpy(batch["rewards"]).cuda()
        dones = torch.from_numpy(batch["dones"]).cuda()

        # train actor
        actions = self.actor.forward(states)
        actor_loss = - self.critic1.forward(states, actions).squeeze().mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        return actor_loss.detach().cpu()

    def sample_action(self, state: np.ndarray, stochastic=True) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state.astype(np.float32)).cuda()
            action_tensor = self.actor.forward(state_tensor)
            action_mean = action_tensor.cpu().numpy()
        if stochastic:
            noise = np.random.randn(*action_mean.shape) * 0.33
            action = np.clip(action_mean + noise, -1, 1)
        else:
            action = action_mean
        return action.astype(np.float32)

    def train(self, num_steps: int, log_dir: str, model_dir: str) -> nn.Module:
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """

        # bring to local scope to increase speed
        env = self.env
        replay_buffer = self.ReplayBuffer

        # init buffer randomly
        state = concatenate_state(env.reset())
        for _ in range(1000):
            action = np.random.randn(7)
            next_state, reward, done, info = env.step(action)
            next_state = concatenate_state(next_state)
            replay_buffer.buffer_add({
                "states": state,
                "actions": action,
                "next_states": next_state,
                "rewards": reward,
                "dones": done
            })

            state = concatenate_state(env.reset()) if done else next_state

        # add logger and remove previous curve
        for filename in glob.glob(os.path.join(log_dir, "events.*")):
            os.remove(filename)
        writer = SummaryWriter(log_dir)

        state = concatenate_state(env.reset())
        for iter in tqdm(range(num_steps)):
            action = self.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = concatenate_state(next_state)
            replay_buffer.buffer_add({
                "states": state,
                "actions": action,
                "next_states": next_state,
                "rewards": reward,
                "dones": done
            })
            state = concatenate_state(env.reset()) if done else next_state

            critic_loss = self.update_critic()
            writer.add_scalar("critic_loss", critic_loss, iter)

            if iter % 1 == 0:
                actor_loss = self.update_actor()
                self.update_target_networks()
                writer.add_scalar("actor_loss", actor_loss, iter)

            if iter % 500 == 0:
                writer.add_scalar("return", self.evaluate(), iter)

            if iter % 5000 == 0:
                torch.save(self.actor_target, os.path.join(model_dir, "actor-%d.pt" % iter))

        return self.actor_target

    def evaluate(self) -> float:
        env = self.evaluate_env
        T = 1000
        returns = np.zeros((5, ))
        rewards = np.zeros((T,))
        for k in range(5):
            t = 0
            state = concatenate_state(env.reset())
            for t in range(T):
                action = self.sample_action(state, False)
                next_state, reward, done, _ = env.step(action)
                next_state = concatenate_state(next_state)
                state = next_state
                rewards[t] = reward
                if done:
                    break

            gamma = self.gamma
            for i in reversed(range(t)):
                rewards[i] += gamma * rewards[i + 1]
            returns[k] = rewards[0]

        return np.mean(returns)


def generate_movie(env: JuggleEnv, policy: nn.Module, video_path: str):
    env.reset()
    env.render()

    encode = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, encode, 24, (1920, 1080), True)

    with torch.no_grad():
        state = torch.from_numpy(concatenate_state(env.reset()).astype(np.float32)).cuda()
        for _ in range(1000):
            action_tensor = policy.forward(state)
            action = action_tensor.cpu().numpy()
            state, reward, done, info = env.step(action)
            if done:
                state = torch.from_numpy(concatenate_state(env.reset()).astype(np.float32)).cuda()
            else:
                state = torch.from_numpy(concatenate_state(state).astype(np.float32)).cuda()
            frame = env.render("rgb_array")
            out.write(frame)
    
    out.release()


if __name__ == "__main__":
    # Define the environment
    train_env = JuggleEnv()
    test_env = JuggleEnv()
    task = "train"

    root_dir = "data/TD3-0-shaping-rand"
    log_dir = os.path.join(root_dir, "log")
    model_dir = os.path.join(root_dir, "model")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if task == "train":
        TD3_object = TD3(
            train_env=train_env,
            test_env= test_env,
            state_dim=24,
            action_dim=7,
            critic_lr=1e-3,
            actor_lr=1e-3,
            gamma=0.99,
            batch_size=100,
        )
        # Train the policy
        policy = TD3_object.train(500000, log_dir, model_dir)
        torch.save(policy, os.path.join(model_dir, "actor-final.pt"))
    else: 
        policy = torch.load(os.path.join(model_dir, "actor-final.pt")).cuda()

    generate_movie(test_env, policy, os.path.join(root_dir, "output.mp4"))