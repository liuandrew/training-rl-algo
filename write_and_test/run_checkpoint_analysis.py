from tqdm import tqdm
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict

import numpy as np
import pickle
from pathlib import Path 

import time
# model, obs_rms, kwargs = load_model_and_env('nav_auxiliary_tasks/nav_aux_wall_1', 0)
# env = gym.make('NavEnv-v0', **kwargs)

save = 'plots/representation_heatmaps/'

from representation_analysis import *
from model_evaluation import *

num_clusters = 9

def gaussian_smooth(pos, y, extent=(5, 295), num_grid=30, sigma=10,
                    ret_hasval=False):
    """Convert a list of positions and values to a smoothed heatmap

    Args:
        pos (): _description_
        y (_type_): _description_
        extent (tuple, optional): _description_. Defaults to (5, 295).
        num_grid (int, optional): _description_. Defaults to 30.
        sigma (int, optional): _description_. Defaults to 10.
        ret_hasval (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # a = stacked['shared_activations'][0, :, 0].numpy()
    y = np.array(y)
    
    grid = np.linspace(extent[0], extent[1], num_grid)
    xs, ys = np.meshgrid(grid, grid)
    ys = ys[::-1]
    smoothed = np.zeros(xs.shape)
    hasval = np.zeros(xs.shape)
    for i in range(num_grid):
        for j in range(num_grid):
            p = np.array([xs[i, j], ys[i, j]])
            dists = np.sqrt(np.sum((pos - p)**2, axis=1))
            g = np.exp(-dists**2 / (2*sigma**2))
            
            if len(g[g > 0.1]) < 1:
                val = 0
            else:
                val = np.sum(y[g > 0.1] * g[g > 0.1]) / np.sum(g[g > 0.1])
                hasval[i, j] = 1

            smoothed[i, j] = val
    if ret_hasval:
        return smoothed, hasval
    else:
        return smoothed


def clean_eps(eps, prune_first=5, activations_key='shared_activations',
             activations_layer=0, clip=False,
             save_inview=True, save_seen=True):
    """Prepare a list of episode data to be used for Gaussian heatmapping
    For clustering we have been using prun_first=0, save_inview=False, save_seen=False

    Args:
        eps (dict): eps dictionary, often from stack_all_ep(all_ep)
        prune_first (int, optional): How many first steps of each episode to remove. Defaults to 5.
        activations_key (str, optional): Which set of activations to save. Defaults to 'shared_activations'.
        activations_layer (int, optional): Which layer of activations to save. Defaults to 0.
            E.g., could use activations_key='actor_activations', activations_layer=1
        clip (bool, optional): Whether to only keep non-negative activation values. Defaults to False.
        save_inview (bool, optional): Whether to add to return dict the steps where poster is in view. Defaults to True.
        save_seen (bool, optional): Whether to add to return dict the steps where poster has been seen. 
            Defaults to True.

    Returns:
        dict: eps ready for conversion to heatmap
    """
    dones = eps['dones'].copy()
    pos = np.vstack(eps['data']['pos'])
    stacked = stack_activations(eps['activations'])
    angles = eps['data']['angle']
    acts = eps['actions']
    
    activ = stacked[activations_key][activations_layer, :, :].numpy()
    pinview = np.array(eps['data']['poster_in_view'])
    pseen = np.array(eps['data']['poster_seen'])
    
    ep_activ = split_by_ep(activ, dones)
    ep_pos = split_by_ep(pos, dones)
    ep_pinview = split_by_ep(pinview, dones)
    ep_angle = split_by_ep(angles, dones)
    ep_pseen = split_by_ep(pseen, dones)
    ep_acts = split_by_ep(acts, dones)
    
    if prune_first and prune_first > 0:
        prune_first = 5
        pruned_ep_activ = [a[prune_first:] for a in ep_activ]
        pruned_activ = np.vstack(pruned_ep_activ)
        pruned_ep_pos = [p[prune_first:] for p in ep_pos]
        pruned_pos = np.vstack(pruned_ep_pos)
        pruned_ep_pinview = [p[prune_first:] for p in ep_pinview]
        pruned_pinview = np.concatenate(pruned_ep_pinview)
        pruned_ep_angles = [p[prune_first:] for p in ep_angle]
        pruned_angles = np.concatenate(pruned_ep_angles)
        pruned_ep_pseen = [p[prune_first:] for p in ep_pseen]
        pruned_pseen = np.concatenate(pruned_ep_pseen)
        pruned_ep_acts = [p[prune_first:] for p in ep_acts]
        pruned_acts = np.concatenate(pruned_ep_acts)
        
        pos = pruned_pos
        activ = pruned_activ
        pinview = pruned_pinview
        angles = pruned_angles
        pseen = pruned_pseen
        acts = pruned_acts
    
    if clip:
        activ = np.clip(activ, 0, 1)
    
    result_dict = {
        'pos': pos,
        'activ': activ,
        'pinview': pinview,
        'pseen': pseen,
        'angles': angles,
        'dones': dones,
        'actions': acts
    }
    
    if save_inview:
        result_dict.update({
            'pos_inview': pos[pinview],
            'pos_notinview': pos[~pinview],
            'activ_inview': activ[pinview],
            'activ_notinview': activ[~pinview],
            'angles_inview': angles[pinview],
            'angles_notinview': angles[~pinview],
        })
    if save_seen:
        result_dict.update({'pos_seen': pos[pseen],
        'pos_notseen': pos[~pseen],
        'activ_seen': activ[pseen],
        'activ_notseen': activ[~pseen],
        'angles_seen': angles[pseen],
        'angles_notseen': angles[~pseen],
        })
    
    return result_dict
    
    
def stack_all_ep(all_ep):
    '''
    When making a list of results from multiple evalu calls,
    this function can be called to put the relevant data into a single dict to be
    passed to clean_eps for processing
    
    E.g.
    all_ep = []
    for i in range(10):
        all_ep.append(forced_action_evaluate(...))
    eps = stack_all_ep(all_ep)
    '''
    dones = np.concatenate([ep['dones'] for ep in all_ep])
    pos = np.vstack([ep['data']['pos'] for ep in all_ep])
    angles = np.concatenate([ep['data']['angle'] for ep in all_ep])
    pseen = np.concatenate([ep['data']['poster_seen'] for ep in all_ep])
    pinview = np.concatenate([ep['data']['poster_in_view'] for ep in all_ep])
    actions = np.vstack([np.vstack(ep['actions']) for ep in all_ep]).squeeze()
    activations = []
    for ep in all_ep:
        activations += ep['activations']

    eps = {
        'dones': dones,
        'activations': activations,
        'actions': actions,
        'data': {
            'pos': pos,
            'angle': angles,
            'poster_seen': pseen,
            'poster_in_view': pinview
        }
    }
    return eps
    

    
def split_by_angle(target, angles):
    """Split a list of data (targets) into slices depending on what angle quadrant
    it was associated with. 

    Args:
        target (list, np.array): List of data to be split
        angles (list, np.array): List of angles to split target data into quadrant by

    Returns:
        dict: target data split into quadrants 0, 1, 2, 3
    """
    splits = {
        0: [-np.pi/4, np.pi/4],
        1: [np.pi/4, 3*np.pi/4],
        3: [-3*np.pi/4, -np.pi/4],
        2: None #this will use else statement otherwise bounds are annoying
    }
    all_trues = np.zeros(angles.shape) == 1
    result = {}
    
    for s in [0, 1, 3]:
        split = splits[s]
        split_idxs = (split[0] <= angles) & (angles <= split[1])
        all_trues = all_trues | split_idxs
        
        result[s] = target[split_idxs]
    #finally, the ones that didn't fit into any of the other quadrants
    result[2] = target[~all_trues]
    
    return result
    
        
    
def compute_directness(all_ep=None, ep=None, pos=None):
    '''
    Compute the directness of paths taken either from an all_ep (split up
    eps generated from appending evalu() calls) or from a single ep
    '''
    goal_loc = np.array([250, 70])
    if all_ep is None and ep is None and pos is None:
        raise Exception('No proper parameters given')

    if all_ep is not None:
        directnesses = []
        for i in range(len(all_ep)):
            p = np.vstack(all_ep[i]['data']['pos'])
            d = p - goal_loc
            d = np.sqrt(np.sum(d**2, axis=1))
            dist_changes = np.diff(d)
            directness = np.sum(dist_changes[:-1] < 0) / np.sum(dist_changes[:-1] != 0)
            directnesses.append(directness)
        return np.array(directnesses)
    else:
        if ep is not None:
            p = np.vstack(ep['data']['pos'])
        elif pos is not None:
            p = pos
        d = p - goal_loc
        d = np.sqrt(np.sum(d**2, axis=1))
        dist_changes = np.diff(d)
        directness = np.sum(dist_changes[:-1] < 0) / np.sum(dist_changes[:-1] != 0)
        return directness
    
        
            
            
def filter_all_ep_directness(all_ep, bound=0.9):
    d = compute_directness(all_ep)
    idxs = d > 0.9
    d_ep = [ep for i, ep in enumerate(all_ep) if idxs[i]]
    return d_ep



def load_heatmaps(file='data/pdistal_rim_heatmap/rim_heatmaps'):
    all_heatmaps = pickle.load(open(file, 'rb'))

    heatmaps = []
    heatmap_idx_to_model = []
    heatmap_model_to_idxs = {}
    widths = [4, 8, 16, 32, 64]
    trials = 3

    current_idx = 0
    for width in widths:
        heatmap_model_to_idxs[width] = []
        for trial in range(trials):
            heatmaps.append(all_heatmaps[width][trial])

            #create indexers to map back and forth between heatmap idxs and models
            for i in range(width):
                heatmap_idx_to_model.append([width, trial, i])
            heatmap_model_to_idxs[width].append([current_idx, current_idx+width])
            current_idx = current_idx + width

    heatmaps = np.clip(np.vstack(heatmaps).reshape(372, 900), 0, 1)
    return heatmaps, heatmap_idx_to_model, heatmap_model_to_idxs


def count_labels(clabels, ignore_cluster=None, remove_zeros=False):
    #Convert a list of cluster labels into ratios
    cluster_counts = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_counts[i] = np.sum(clabels == i)
        
    if ignore_cluster is not None:
        if type(ignore_cluster) == list:
            for c in ignore_cluster:
                cluster_counts[c] = 0
        elif type(ignore_cluster) == int:
            cluster_counts[ignore_cluster] = 0
    
    cluster_ratios = cluster_counts / np.sum(cluster_counts)
    
    if remove_zeros:
        cluster_ratios = cluster_ratios[cluster_ratios != 0]
        cluster_counts = cluster_counts[cluster_counts != 0]
    return cluster_counts, cluster_ratios




def collect_checkpoint_data(trial_name, trial, data_folder='data/pdistal_widthaux_heatmap/',
                            checkpoint_folder='../trained_models/checkpoint/nav_pdistal_widthaux/',
                            model_folder='nav_pdistal_widthaux/',
                            checkpoints=None):
    """Given an experimental trial_name and trial, load saved checkpoints and collect
    forced trajectory and natural trajectory data to 

    Args:
        trial_name (_type_): _description_
        trial (_type_): _description_
        data_folder (str, optional): _description_. Defaults to 'data/pdistal_widthaux_heatmap/'.
        checkpoint_folder (str, optional): _description_. Defaults to '../trained_models/checkpoint/nav_pdistal_widthaux/'.
        model_folder (str, optional): _description_. Defaults to 'nav_pdistal_widthaux/'.
    """
    print(f'Collecting trajectory data for {trial_name}:{trial}')
    start_time = time.time()
    
    path = checkpoint_folder + f'{trial_name}_t{trial}'
    # Get env_kwargs
    kwargs_name = f'../trained_models/ppo/{model_folder}{trial_name}_env'
    # print(model_name)
    # _, _, kwargs = load_model_and_env(model_name, trial)
    kwargs = pickle.load(open(kwargs_name, 'rb'))

    save_path = data_folder + f'{trial_name}_checkpoint'
    if not Path(save_path).exists():
        checkpoint_data = {'copied': {}, 'policy': {}}
    else:
        checkpoint_data = pickle.load(open(save_path, 'rb'))
        
        
    # Make sure trial exists in the data file
    if trial not in checkpoint_data['copied']:
        checkpoint_data['copied'][trial] = {}
        checkpoint_data['policy'][trial] = {}
    else:
        print('Already completed, skipping')
        return

    if checkpoints == None:
        checkpoints = list(Path(path).iterdir())
        checkpoints = [int(chk.name.split('.pt')[0]) for chk in checkpoints]

    with torch.no_grad():
        for chkp_val in tqdm(checkpoints):
            if chkp_val in checkpoint_data['policy'][trial]:
                #already ran this checkpoint
                continue
            
            checkpoint = Path(path)/f'{chkp_val}.pt'
            model, obs_rms = torch.load(checkpoint)

            #Generate copied activations
            all_ep = []
            for i in range(len(keep_start_points)):
                copied_actions = lambda step: combined_actions[i][step]
                kw = kwargs.copy()
                kw['fixed_reset'] = [keep_start_points[i].copy(), keep_start_angles[i].copy()]
                ep = forced_action_evaluate(model, obs_rms, seed=0, num_episodes=1, eval_log_dir='./',
                                            env_kwargs=kw, data_callback=poster_data_callback,
                                            with_activations=True, forced_actions=copied_actions)
                all_ep.append(ep)
            eps = clean_eps(stack_all_ep(all_ep), prune_first=0, save_inview=False,
                            save_seen=False)
            checkpoint_data['copied'][trial][chkp_val] = eps


            #Generate policy activations
            all_ep = []
            for i in range(len(start_points)):
                kw = kwargs.copy()
                kw['fixed_reset'] = [start_points[i].copy(), start_angles[i].copy()]
                ep = forced_action_evaluate(model, obs_rms, seed=0, num_episodes=1, eva
                                            env_kwargs=kw, data_callback=poster_data_callback,
                                            with_activations=True)
                all_ep.append(ep)
            eps = clean_eps(stack_all_ep(all_ep), prune_first=0, save_inview=False, save_seen=False)

            checkpoint_data['policy'][trial][chkp_val] = eps
            
            pickle.dump(checkpoint_data, open(save_path, 'wb'))
            
    end_time = time.time()
    print('Wall time:', round(end_time-start_time, 2))
        

def compute_heatmaps(trial_name, trial, data_folder='data/pdistal_widthaux_heatmap/'):
    print(f'Computing heatmaps for {trial_name}:{trial}')
    start_time = time.time()
    
    data_path = data_folder + f'{trial_name}_checkpoint'
    checkpoint_data = pickle.load(open(data_path, 'rb'))

    heatmap_path = data_folder + f'{trial_name}_checkpoint_hms'
    if not Path(heatmap_path).exists():
        heatmap_data = {}
    else:
        heatmap_data = pickle.load(open(heatmap_path, 'rb'))

    if trial not in heatmap_data:
        heatmap_data[trial] = {}
    else:
        print('Already completed, skipping')
        return

    for chkp_val in tqdm(checkpoint_data['copied'][trial]):
        eps = checkpoint_data['copied'][trial][chkp_val]
        heatmaps = []

        p = eps['pos']
        a = eps['activ']

        for i in range(a.shape[1]):
            heatmap = gaussian_smooth(p, a[:, i])
            heatmaps.append(heatmap)
        
        heatmap_data[trial][chkp_val] = heatmaps
            
    pickle.dump(heatmap_data, open(heatmap_path, 'wb'))

    end_time = time.time()
    print('Wall time:', round(end_time-start_time, 2))



def compute_summary_stats(trial_name, trial, data_folder='data/pdistal_widthaux_heatmap/'):
    print(f'Compute summary stats for width {trial_name}:{trial}')
    start_time = time.time()
    
    kmeans = pickle.load(open('data/pdistal_rim_heatmap/kmeans_heatmap_clusterer', 'rb'))

    data_path = data_folder + f'{trial_name}_checkpoint'
    checkpoint_data = pickle.load(open(data_path, 'rb'))

    heatmap_path = data_folder + f'{trial_name}_checkpoint_hms'
    summary_path = data_folder + f'{trial_name}_checkpoint_summ'

    heatmap_data = pickle.load(open(heatmap_path, 'rb'))
    if not Path(summary_path).exists():
        summary_data = {}
    else:
        summary_data = pickle.load(open(summary_path, 'rb'))

    if trial not in summary_data: 
        summary_data[trial] = {}
    else:
        print('Already completed, skipping')
        return

    for chkp_val in checkpoint_data['policy'][trial]:    
        eps = checkpoint_data['policy'][trial][chkp_val]
        
        
        directness = compute_directness(pos=eps['pos'])

        ep_dones = split_by_ep(eps['dones'], eps['dones'])
        ep_lens = np.array([ep.shape[0] for ep in ep_dones])
        success_rate = 1 - np.sum(ep_lens == 202) / len(ep_lens)
        average_ep_len = np.mean(ep_lens)
        average_succ_ep_len = np.mean(ep_lens[ep_lens < 202])

        acts = eps['actions']
        act_ratios = np.array([np.sum(acts == i) for i in range(4)]) / len(acts)

        hms = np.vstack([hm.reshape(1, -1) for hm in heatmap_data[trial][chkp_val]])
        labels = kmeans.predict(hms)
        _, ratios = count_labels(labels, remove_zeros=False)
        _, nonzero = count_labels(labels, remove_zeros=True)
        hprime = np.sum(-nonzero * np.log(nonzero))

        summary_data[trial][chkp_val] = {
            'cluster_labels': labels, 
            'cluster_ratios': ratios, 
            'directness': directness, 
            'success_rate': success_rate, 
            'avg_ep_len': average_ep_len, 
            'avg_succ_ep_len': average_succ_ep_len, 
            'act_ratios': act_ratios,
            'shannon': hprime
        }
        
    pickle.dump(summary_data, open(summary_path, 'wb'))
    end_time = time.time()
    print('Wall time:', round(end_time-start_time, 2))



if __name__ == '__main__':
    '''
    Setup points and trajectories

    combined_actions, keep_start_points, keep_start_angles used for
        forced trajectories
    start_points, start_angles used for natural policies
    '''
    combined_actions, keep_start_points, keep_start_angles = pickle.load(open(f'data/pdistal_rim_heatmap/width64_comb_acts', 'rb'))

    #Starting around rim - First generate start points and angles
    WINDOW_SIZE = (300, 300)
    step_size = 10.
    xs = np.arange(0+step_size, WINDOW_SIZE[0], step_size)
    ys = np.arange(0+step_size, WINDOW_SIZE[1], step_size)
    # thetas = np.linspace(0, 2*np.pi, 12, endpoint=False)
    start_points = []
    start_angles = []
    for x in xs:
        for y in [5., 295.]:
            point = np.array([x, y])
            angle = np.arctan2(150 - y, 150 - x)
            start_points.append(point)
            start_angles.append(angle)
    for y in ys:
        for x in [5, 295]:
            point = np.array([x, y])
            angle = np.arctan2(150 - y, 150 - x)
            start_points.append(point)
            start_angles.append(angle)
            
    start_points = np.vstack(start_points)
    
    # Settings for origianl pdistal_width{width}batch200 trials
    
    # checkpoint_folder = '../trained_models/checkpoint/nav_poster_netstructure/'
    # model_folder = ''../trained_models/nav_poster_netstructure/'
    # data_folder = 'data/pdistal_rim_heatmap/'
    # num_trials = 10
    # widths = [2, 3, 4, 8, 16, 32, 64]
    # for t in range(num_trials):
    #     for width in widths:
    #         trial_name = f'nav_pdistal_width{width}batch200'
    #         collect_checkpoint_data(trial_name, t)
    #         compute_heatmaps(trial_name, t)
    #         compute_summary_stats(trial_name, t)
    
    
    
    # # Settings for auxiliary task pdistal_widthaux
    # num_trials = 3
    # widths = [16, 32, 64]
    # auxiliary_task_names = ['wall0', 'wall1', 'wall01', 'goaldist']
    # checkpoint_folder = '../trained_models/checkpoint/nav_pdistal_widthaux/'
    # data_folder = 'data/pdistal_widthaux_heatmap/'
    # model_folder = 'nav_pdistal_widthaux/'
    
    # for t in range(num_trials):
    #     for width in widths:
    #         for aux in auxiliary_task_names:
    #             trial_name = f'nav_pdistal_width{width}aux{aux}'
    #             collect_checkpoint_data(trial_name, t, data_folder=data_folder,
    #                                     checkpoint_folder=checkpoint_folder,
    #                                     model_folder=model_folder)
    #             compute_heatmaps(trial_name, t, data_folder=data_folder)
    #             compute_summary_stats(trial_name, t, data_folder=data_folder)
                
                
    # Settings for auxiliary task pdistal_width16batchaux
    num_trials = 3
    # widths = [16, 32, 64]
    batch_sizes = [8, 16, 32, 64, 96, 128]
    checkpoints = [
        [0, 320, 640, 960, 1280, 1600, 1920, 2240, 2560, 2880, 3200, 3749],
        [0, 160, 320, 480, 640, 800, 960, 1120, 1280, 1440, 1600, 1760, 1874],
        [0, 80, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880, 936],
        [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 467],
        [0, 30, 60, 80, 110, 140, 170, 200, 220, 250, 280, 300, 311],
        [0, 20, 40, 80, 100, 120, 140, 160, 180, 200, 220, 233],
    ]
    
    auxiliary_task_names = ['wall0', 'wall1', 'wall01', 'goaldist', 'none']
    checkpoint_folder = '../trained_models/checkpoint/nav_pdistal_batchaux/'
    data_folder = 'data/pdistal_batchaux_heatmap/'
    model_folder = 'nav_pdistal_batchaux/'
    
    for t in range(num_trials):
        for n, batch in enumerate(batch_sizes):
            for aux in auxiliary_task_names:
                trial_name = f'nav_pdistal_batch{batch}aux{aux}'
                collect_checkpoint_data(trial_name, t, data_folder=data_folder,
                                        checkpoint_folder=checkpoint_folder,
                                        model_folder=model_folder,
                                        checkpoints=checkpoints[n])
                compute_heatmaps(trial_name, t, data_folder=data_folder)
                compute_summary_stats(trial_name, t, data_folder=data_folder)