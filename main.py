import os
import sys
import pandas as pd
import numpy as np
import gym
import argparse
from tqdm import tqdm
from stable_baselines3 import PPO
from behavior_cloning import Trainer, BehaviorCloningModel
from data_selection import select_cluster, select_data

def evaluate_agent(agent, env, num_episodes=100):
    total_rewards = []
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset(seed=42)
        episode_reward = 0
        
        while True:
            action = agent.predict_one(obs) 
            obs, rew, terminated, truncated, info = env.step(action)
            episode_reward += rew
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    return avg_reward, std_reward

def main():
    n = 2500
    lr = 0.0001
    e = 50
    patience = 5
    print({'n': n, 'lr': lr, 'e': e, 'patience': patience})
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action="store_true", default=False, help='Whether to use dagger')
    parser.add_argument('-c', action="store_true", default=False, help='Whether to use cluster selection')
    parser.add_argument('-u', action="store_true", default=False, help='Whether to use uncertainty-based selection')
    args = parser.parse_args()

    env = gym.make('LunarLander-v2')
    # Load data
    data = pd.read_csv('LunarLander_cluster.csv')
    data['observations'] = data['observations'].apply(eval)
    if args.c: # uncertainty and dagger doesn't influence initial data selection
        init_data, remaining_data = select_cluster(data, n)
    init_data = data.sample(n)
    remaining_data = data.drop(init_data.index)
    avg, std = [], []
    # Train behavior cloning model
    bc = BehaviorCloningModel(8, 4)
    model = Trainer(bc, lr)
    X, Y = np.vstack(init_data['observations'].values), init_data['actions'].values
    print('-' * 20, len(init_data), '-' * 20)
    model.train(X, Y, epochs=e, save_path=f'models/bc_model_d{int(args.d)}_c{int(args.c)}_u{int(args.u)}_0.pth', interval=e//10, patience=patience)
    avg_reward, std_reward = evaluate_agent(model, env)
    avg.append(avg_reward)
    std.append(std_reward)
    path = os.path.abspath('') + '/logs/ppo/LunarLander-v3_1/best_model'
    expert = PPO.load(path, device='cpu')
    max_reward = -np.inf
    former_data = init_data
    cnt = 0
    for i in range(int(100000/n-1)):
        selected_data, remaining_data = select_data(model, expert, env, remaining_data, n, args)
        former_data = pd.concat([former_data, selected_data])
        X, Y = np.vstack(former_data['observations'].values), former_data['actions'].values
        # model.load_model(f'models/bc_model_d{int(args.d)}_c{int(args.c)}_u{int(args.u)}.pth')
        bc = BehaviorCloningModel(8, 4)
        model = Trainer(bc, lr)
        print('-' * 20, len(former_data), '-' * 20)
        model.train(X, Y, epochs=e, patience=patience, save_path=f'models/bc_model_d{int(args.d)}_c{int(args.c)}_u{int(args.u)}_{i+1}.pth', interval=e//10)
        avg_reward, std_reward = evaluate_agent(model, env)
        avg.append(avg_reward)
        std.append(std_reward)
        rew = avg_reward - std_reward
        if rew >= max_reward:
            max_reward = rew
            if rew >= 200:
                cnt += 1
        elif max_reward >= 200:
            break
        if cnt == 2:
            break
    print('Average rewards:', avg)
    print('Standard deviations:', std)

if __name__ == '__main__':
    main()