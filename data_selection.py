import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.cluster import KMeans

def select_cluster(df, num, score='random'):
    ids = df['cluster_id'].unique()
    # Randomly select clusters
    if len(ids) >= num:
        selected_ids = random.sample(list(ids), num)
        df_selected = df[df['cluster_id'].isin(selected_ids)]
        df_unselected = df[~df['cluster_id'].isin(selected_ids)]
        if score == 'random':
            selected = df_selected.groupby('cluster_id').apply(lambda x: x.sample(1)).reset_index(drop=True)
        else:
            selected = df_selected.groupby('cluster_id').apply(lambda x: x.nlargest(1, score)).reset_index(drop=True)
    else:
        if score == 'random':
            selected = pd.DataFrame()
            to_sel = pd.DataFrame()
            while len(selected) + len(to_sel) < num:
                to_sel = df.groupby('cluster_id', group_keys=False).apply(lambda x: x.sample(n=1))
                selected = pd.concat([selected, to_sel])
                # Remove selected clusters
                df = df[~df.index.isin(to_sel.index)]
            to_sel = to_sel.sample(n=num-len(selected))
            selected = pd.concat([selected, to_sel])
        else:
            selected = pd.DataFrame()
            to_sel = pd.DataFrame()
            while len(selected) + len(to_sel) < num:
                to_sel = df.groupby('cluster_id', group_keys=False).apply(lambda x: x.nlargest(1, score))
                selected = pd.concat([selected, to_sel])
                # Remove selected clusters
                df = df[~df.index.isin(to_sel.index)]
            to_sel = to_sel.nlargest(num-len(selected), score)
            selected = pd.concat([selected, to_sel])
        df_unselected = df
    return selected, df_unselected

def dagger_step(env, model, expert, num1, cluster=False, num2=None):
    """
    Perform one step of the DAgger algorithm.
    Args:
        env: Gym environment.
        model: Behavior cloning model.
        expert: Expert model.
        num1: Number of expert demonstrations to collect.
        num2: Number of selected demonstrations.
    Returns:
        pd.DataFrame: Expert demonstrations.
    """
    expert_data = []
    episode_id = 0
    cnt = 0
    with tqdm(total=num1, desc="Progress", unit="unit") as pbar:
        while len(expert_data) < num1:
            obs, _ = env.reset(seed=42)
            while True:
                action = model.predict_one(obs) 
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    episode_id += 1
                    break
                expert_data.append({'observations': obs, 'actions': int(expert.predict(obs, deterministic=True)[0]), 'episode_id': episode_id})
                cnt += 1
                pbar.update(1)
                if cnt >= num1:
                    break
                
    df = pd.DataFrame(expert_data)
    if cluster:
        print('Clustering states...', flush=True)
        states = np.vstack(df['observations'].values)
        df['cluster_id'] = KMeans(n_clusters=num2).fit_predict(states)
    # df['observations'] = df['observations'].apply(lambda x: str(x.tolist()))
    return df

def uncertainty_sampling(model, data, num):
    # print(np.vstack(data['observations'].values).shape)
    data['entropy'] = model.get_entropy(np.vstack(data['observations'].values))
    data_selected = data.nlargest(num, 'entropy')
    data_unselected = data.drop(data_selected.index)
    return data_selected, data_unselected

def select_data(model, expert, env, data, num, args):
    # dagger
    # uncertainty
    # cluster (uncertainty?)
    # dagger + cluster (uncertainty?)
    if not args.d:
        if not args.c:
            if not args.u:
                data_selected = data.sample(num)
                data_unselected = data.drop(data_selected.index)
            else:
                data_selected, data_unselected = uncertainty_sampling(model, data, num)
        else:
            if not args.u:
                data_selected, data_unselected = select_cluster(data, num)
            else:
                data['entropy'] = model.get_entropy(np.vstack(data['observations'].values))
                data_selected, data_unselected = select_cluster(data, num, score='entropy')
        return data_selected, data_unselected
    else: # dagger
        if not args.c:
            if not args.u:
                data_selected = dagger_step(env, model, expert, num)
            else:
                data_selected = dagger_step(env, model, expert, num * 5)
                data_selected, data_unselected = uncertainty_sampling(model, data_selected, num)
        else:
            if not args.u: # d, c
                data_selected = dagger_step(env, model, expert, num * 5, cluster=True, num2=num)
                data_selected, data_unselected = select_cluster(data_selected, num)
            else: # d, c, u
                data_selected = dagger_step(env, model, expert, num * 5, cluster=True, num2=num)
                data_selected['entropy'] = model.get_entropy(np.vstack(data_selected['observations'].values))
                data_selected, data_unselected = select_cluster(data_selected, num, score='entropy')
        return data_selected, None