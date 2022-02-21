import copy
import glob
import os
import pprint
import traceback
import time
from collections import deque
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import re
import scipy.interpolate
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
sys.path.insert(0, '..')

import gym
import gym_nav
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data



def print_runs(folder='../runs/', prin=True):
    '''
    print tensorboard runs directories
    
    prints out each unique experiment name and how many trials were associated
    next use load_exp_df with the name and trial number to load it into a df
    
    also plan to update this function as more runs are added with 
    descriptions of what each run is
    '''
    space =  '    '
    branch = '│   '
    tee =    '├── '
    last =   '└── '
    
    run_descriptions = {
        'invisible_shared': ''
        
    }
    
    
    path = Path(folder)
    if prin:
        print(path.name)
    
    unique_experiments = {}
    original_experiment_names = {}
    depth = 0
    for d in path.iterdir():
        if '__' in d.name:
            trial_name = d.name.split('__')[0]
            
            if re.match('.*\_\d*', trial_name):
                exp_name = '_'.join(trial_name.split('_')[:-1])
            else:
                exp_name = trial_name
                
            if exp_name in unique_experiments.keys():
                unique_experiments[exp_name] += 1
            else:
                unique_experiments[exp_name] = 1
                
    if prin:
        for key, value in unique_experiments.items():
            if value > 1:
                print(branch*depth + tee+'EXP', key + ':', value)
            else:
                print(branch*depth+tee+key)

    return unique_experiments



def load_exp_df(exp_name, trial_num=0, folder='../runs/', save_csv=True):
    '''
    load experiment tensorboard run into a dataframe
    if save_csv is True, also convert into a csv for faster loading later
    
    if trial_num is None, directly try to load the given exp_name
    '''
    dirs = os.listdir(folder)
    
    if trial_num is not None:
        results = []
        for d in dirs:
            if '__' in d:
                trial_name = d.split('__')[0]
                if re.match('.*\_\d*', trial_name):
                    name = '_'.join(trial_name.split('_')[:-1])
                    trial = trial_name.split('_')[-1]
                    if 't' in trial:
                        trial = trial.split('t')[-1]
                    trial = int(trial)

                    if exp_name == name and trial == trial_num:
                        results.append(folder + d)

        if len(results) > 1:
            print('Warning: more than one experiment with the name and trial num found')
        if len(results) == 0:
            print('No experiments found')
            return None

        path = results[0]
    else:
        path = folder + exp_name
    files = os.listdir(path)
    
    #look for a preconverted csv dataframe file
    for file in files:
        if '.csv' in file:
            df = pd.read_csv(path + '/' + file)
            return df
    
    #no preconverted csv found, convert the events file
    for file in files:
        if 'events.out.tfevents' in file:
            df = tflog2pandas(path + '/' + file)
            
            if save_csv:
                df.to_csv(path + '/' + 'tflog.csv')
            
            return df
        
    print('No tf events file correctly found')
    return results



def plot_exp_df(df, smoothing=0.1):
    '''
    Plot the experiments values from tensorboard df
    (get the df from load_exp_df)
    '''
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))

    for i, chart in enumerate(df['metric'].unique()):
        # print(chart)
        x = i // 3
        y = i % 3
        idx = df['metric'] == chart
        df.loc[idx, 'ewm'] = df.loc[idx, 'value'].ewm(alpha=smoothing).mean()
        d = df[df['metric'] == chart]

        ax[x, y].plot(d['step'], d['value'], c='C0', alpha=0.1)
        ax[x, y].plot(d['step'], d['ewm'], c='C0')
        ax[x, y].set_title(chart)

        
def average_runs(trial_name, metric='return', ax=None, ewm=0.01,
                label=None):
    '''
    Get the average over a bunch of trials of the same name
    
    trial_name: Name of experiment, should be one found from print_runs()
    metric: Name of metric to plot, some shortcuts can be passed like
        value_loss, policy_loss, return, length
    ax: Optionally pass ax to plot on
    ewm: whether to do an exponential average of metric
        if not wanted, pass False
    label: whether to plot with a label
    '''
    shortcut_to_key = {
        'value_loss': 'losses/value_loss',
        'policy_loss': 'losses/policy_loss',
        'return': 'charts/episodic_return',
        'length': 'charts/episodic_length'
    }
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    exps = print_runs(prin=False) #just borrow function to get number of experiments
    if metric in shortcut_to_key:
        metric = shortcut_to_key[metric]
    
    if trial_name not in exps:
        print('No experiments with the given name found in runs folder')    
    else:
        # num_trials = exps[trial_name]
        folder = '../runs/'
        files = os.listdir(folder)
        exps = []
        for file in files:
            if trial_name in file:
                exps.append(file)
        
        # Averaging code same as from plot_cloud_from_dict in data_visualize.ipynb
        first_xs = []
        last_xs = []
        inters = []
        num_trials = 0
        
        # for i in range(num_trials):
        for exp in exps:
            # Load df from run file
            # df = load_exp_df(trial_name, i)
            
            df = load_exp_df(exp, trial_num=None)
            df = df[df['metric'] == metric]
            if len(df) < 1:
                raise Exception('No metric called {} found in {}'.format(
                    metric, exp))
            
            first_xs.append(df.iloc[0]['step'])
            last_xs.append(df.iloc[-1]['step'])
            
            if ewm:
                df['ewm'] = df['value'].ewm(alpha=ewm).mean()
                inter = scipy.interpolate.interp1d(df['step'], df['ewm'])
                inters.append(inter)
            else:
                inter = scipy.interpolate.interp1d(df['step'], df['value'])
                
            num_trials += 1
            
        min_x = np.max(first_xs)
        max_x = np.min(last_xs)
        xs = np.arange(min_x, max_x, 200)
        ys = np.zeros((num_trials, len(xs)))
        
        for j in range(num_trials):
            ys[j] = inters[j](xs)
        
        if ewm:
            ax.fill_between(xs, ys.min(axis=0), ys.max(axis=0), alpha=0.1)
        ax.plot(xs, ys.mean(axis=0), label=label)
        
