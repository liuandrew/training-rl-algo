from tqdm import tqdm
from model_evaluation import forced_action_evaluate, load_model_and_env, nav_data_callback
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
import proplot as pplt

import sys
sys.path.append('..')
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs

WINDOW_SIZE = (300, 300)

num_grid_slices = 5 # how many grid squares to slice maze into
num_grid_points = num_grid_slices**2

xs_grid = np.linspace(0, WINDOW_SIZE[0], num_grid_slices, endpoint=False)
ys_grid = np.linspace(0, WINDOW_SIZE[1], num_grid_slices, endpoint=False)

softmax = nn.Softmax(dim=1)


def activation_testing(model, env, x, y, angle):
    """
    Pass a model and corresponding environment, x, y and angle, then 
    get the observation and perform a prediction with model to get activations
    
    returns:
        outputs object from model.base.forward
    
    outputs['activations'] is the activation dict
    note that outputs['activations']['shared_activations'] is the same as the 
        rnn hidden state output
    """
    vis_walls = env.vis_walls
    vis_wall_refs = env.vis_wall_refs
    
    env.character.pos = np.array([x, y])
    env.character.angle = angle
    env.character.update_rays(vis_walls, vis_wall_refs)
    
    obs = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
    rnn_hxs = torch.zeros(1, model.recurrent_hidden_state_size, dtype=torch.float32)
    masks = torch.zeros(1, 1, dtype=torch.float32)
    
    outputs = model.base.forward(obs, rnn_hxs, masks, with_activations=True)
    outputs['obs'] = obs
    return outputs



def stack_activations(activation_dict, also_ret_list=False):
    '''
    Activations passed back from a FlexBase forward() call can be appended, e.g.
    all_activations = []
    for ...:
        all_activations.append(actor_critic.act(..., with_activations=True)['activations'])
        
    This will result in a list of dictionaries
    
    This function converts all_activations constructed in this way into a dictionary,
    where each value of the dictionary is a tensor of shape
    [layer_num, seq_index, activation_size]
    
    Args:
        also_ret_list: If True, will also return activations in a list one-by-one
            rather than dict form. Good for matching up with labels and classifier ordering
            from train classifiers function
    '''
    stacked_activations = defaultdict(list)
    list_activations = []
    keys = activation_dict[0].keys()
    
    for i in range(len(activation_dict)):
        for key in keys:
            num_layers = len(activation_dict[i][key])
            
            if num_layers > 0:
                # activation: 1 x num_layers x activation_size
                activation = torch.vstack(activation_dict[i][key]).reshape(1, num_layers, -1)
                # stacked_activations: list (1 x num_layers x activation_size)
                stacked_activations[key].append(activation)
    
    for key in stacked_activations:
        activations = torch.vstack(stacked_activations[key]) # seq_len x num_layers x activation_size
        activations = activations.transpose(0, 1) # num_layers x seq_len x activation_size
        stacked_activations[key] = activations
        
        #Generate activations in list form
        if also_ret_list:
            for i in range(activations.shape[0]):
                list_activations.append(activations[i])
    
    if also_ret_list:
        return stacked_activations, list_activations
    else:
        return stacked_activations


def find_grid_index(point=None, x=None, y=None):
    '''
    Take a point or x and y coordinates and return the grid index
    that the position falls into. Note that coarseness of grid
    depends on parameter set in representation_analysis.py file
    '''
    if point is not None:
        x = point[0]
        y = point[1]
    elif x is not None and y is not None:
        pass
    else:
        raise Exception('No valid argument combination given')
        
    x_grid_idx = np.max(np.argwhere(x >= xs_grid))
    y_grid_idx = np.max(np.argwhere(y >= ys_grid))
    total_grid_idx = x_grid_idx * num_grid_slices + y_grid_idx
    return total_grid_idx



def train_classifier(x, labels, valid_x=None, valid_labels=None, num_labels=None,
                     lr=0.1, epochs=1000, prog=False):
    '''
    Train an arbitrary classifier
        x (tensor): train X data
        labels (tensor): train labels
        valid_x (tensor, optional): train valid X data
        valid_labels (tensor, optional): train valid labels
        num_labels (int, optional): number of output labels
            for model to output. If not provided, default is
            given by the largest label value in labels
    '''
    if num_labels is None:
        num_labels = labels.max().item() + 1
    linear = nn.Linear(x.shape[1], num_labels)
    optimizer = optim.Adam(linear.parameters(), lr=lr)
    softmax = nn.Softmax(dim=1)
    # optimizer = optim.SGD(linear.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    valid_accuracies = []
    if prog:
        it = tqdm(range(epochs))
    else:
        it = range(epochs)
        
    for i in it:
        y = linear(x)
        loss = criterion(y, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(accuracy_score(softmax(y.detach()).argmax(axis=1), labels))

        if valid_x is not None and valid_labels is not None:
            pred_valid = softmax(linear(valid_x)).argmax(axis=1)
            valid_accuracies.append(accuracy_score(pred_valid, valid_labels))
            
    return linear, losses, accuracies, valid_accuracies


def draw_character(pos, angle, size=10, ax=None, color=None):
    '''
    Given a position and angle, draw character to the given axis
    '''
    angle1 = angle - 0.3
    angle2 = angle + 0.3
    point1 = [pos[0], pos[1]]
    point2 = [pos[0] - np.cos(angle1)*size, pos[1] - np.sin(angle1)*size]
    point3 = [pos[0] - np.cos(angle2)*size, pos[1] - np.sin(angle2)*size]

    if color is None:
        color = np.array([0.9, 0.9, 0])

    poly = plt.Polygon([point1, point2, point3], fc=color)
    if ax is None:
        plt.gca().add_patch(poly)
    else:
        ax.add_patch(poly)

        
# def get_activations(model, obs, masks, pos, device=torch.device('cpu')):
#     eval_recurrent_hidden_states = torch.zeros(
#         num_processes, model.recurrent_hidden_state_size, device=device)

#     with torch.no_grad():
#         outputs = model.base(obs, eval_recurrent_hidden_states, 
#                                masks, deterministic=True, with_activations=True)
    
#     #For some reason, the first ~100ish activations are misaligned with the original
#     #activations if the agent is run with the episodes live, so remove some early misalignment
#     skip = 0
    
#     activations = outputs['activations']
#     stacked = {}
#     for key in activations:
#         substacked = []
#         for i in range(len(activations[key])):
#             activ = activations[key][i][skip:, :]
#             shape = activ.shape
#             substacked.append(activ.reshape(1, shape[0], shape[1]))
#         stacked[key] = torch.vstack(substacked)
#     pos = pos[skip:]
#     # angle = angle[skip:]

#     return stacked, pos
    

def train_position_classifier(model, obs_rms=None, kwargs=None, model_num=None, 
                              train_episodes=30, valid_episodes=5, epochs=100, 
                              seed=0, random_actions=True):
    '''
    Complete automated training of a position decoder
    model (str or Policy): the model to be trained, can either be a
        path to trained model from ../trained_models/ppo/ or a policy directly
    
    model (str):
        model_num (int): Need to also pass the trial number to load
    model (Policy):
        obs_rms, kwargs: These are usually loaded from load_model_and_env
            so when a Policy is passed, need to pass these as well
    '''
    
    if model_num is not None and type(model) == str:
        model, obs_rms, kwargs = load_model_and_env(model, model_num)
    elif obs_rms is not None and kwargs is not None \
        and type(model) == Policy:
        pass
    else:
        raise ValueError('Must pass either model (str) + model_num or model (Policy) + obs_rms + kwargs')
    
    # Generate activations and positions for training and validation
    train_results = perform_ep_collection(model, obs_rms, kwargs,
                                     seed=seed, num_episodes=train_episodes, random_actions=random_actions)
    stacked, grid_indexes = train_results['stacked'], train_results['grid_indexes']

    valid_results = perform_ep_collection(model, obs_rms, kwargs,
                                     seed=seed, num_episodes=valid_episodes, random_actions=random_actions)
    valid_stacked, valid_grid_indexes = train_results['stacked'], train_results['grid_indexes']
    
    # Train position decoders on activations
    classifiers = {}

    all_losses = []
    all_accuracies = []
    labels = []
    valid_accuracies = []
    
    valid_preds = []

    for key in stacked:
        # print(key)
        num_layers = stacked[key].shape[0]
        for i in range(num_layers):            
            x = stacked[key][i]
            valid_x = valid_stacked[key][i]
            
            linear, losses, accuracies, _ = train_classifier(x, grid_indexes, epochs=epochs)
            classifiers[f'{key}_{i}'] = linear
            
            labels.append(f'{key}_{i}')
            all_losses.append(losses)
            all_accuracies.append(accuracies)
    
            pred_valid = linear(valid_x).argmax(axis=1)
        
            valid_preds.append(pred_valid)
            
            accuracy_score(pred_valid, valid_grid_indexes)
            valid_accuracies.append(accuracy_score(pred_valid, valid_grid_indexes))
    return {
        'labels': labels,
        'losses': all_losses,
        'training_acc': all_accuracies,
        'final_valid_acc': valid_accuracies,
        'classifiers': classifiers,
        
        'valid_preds': valid_preds,
        'valid_grid_indexes': valid_grid_indexes
    }
    
    
def quick_vec_env(obs_rms, env_kwargs={}, env_name='NavEnv-v0', seed=0,
                 num_processes=1, eval_log_dir='/tmp/gym/_eval',
                 device=torch.device('cpu'), capture_video=False):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                          None, eval_log_dir, device, True, 
                          capture_video=capture_video, 
                          env_kwargs=env_kwargs)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms
        
    return eval_envs

    
def perform_ep_collection(model, obs_rms=None, kwargs=None, model_num=None, 
                          num_episodes=1, seed=0, random_actions=False,
                         data_callback=nav_data_callback):
    '''
    Collect activations, positions, etc. for a number of episodes
    Used by train_position_classifier to collect activation and pos data
    Can also be used to generate single episodes of data to evaluate classifiers etc.
    
    Same model (str / Policy) parameters as train_position_classifier
    '''
    if model_num is not None and type(model) == str:
        model, obs_rms, kwargs = load_model_and_env(model, model_num)
    elif obs_rms is not None and kwargs is not None \
        and type(model) == Policy:
        pass
    else:
        raise ValueError('Must pass either model (str) + model_num or model (Policy) + obs_rms + kwargs')

    
    # Generate random episodes to train on
    if seed is not None:
        np.random.seed(seed)
    action_randomizer = lambda step: np.random.choice([0, 1, 2])
    
    if random_actions:
        results = forced_action_evaluate(model, obs_rms, forced_actions=action_randomizer, env_kwargs=kwargs, seed=seed,
                                        data_callback=data_callback, num_episodes=num_episodes, with_activations=True)
    else:
        results = forced_action_evaluate(model, obs_rms, env_kwargs=kwargs, seed=seed,
                                        data_callback=data_callback, num_episodes=num_episodes, with_activations=True)

    pos = np.vstack(results['data']['pos'])
    angle = torch.tensor(np.vstack(results['data']['angle']))
    stacked, listed_activ = stack_activations(results['activations'], also_ret_list=True)
    grid_indexes = torch.tensor([find_grid_index(p) for p in pos])
    
    results['pos'] = pos
    results['angle'] = angle
    results['stacked'], results['listed_activ'] = stacked, listed_activ
    results['grid_indexes'] = grid_indexes
    
    return results
    
    
    
def prune_fail_episodes(targets, dones):
    '''
    Remove episodes where the agent failed. Need to pass 'dones'
    from episode evaluations, which lets us determine when episode
    reached max length, as well as targets where the corresponding
    data points of failed episodes will be removed from
    
    E.g.
    
    '''
    max_ep_len = 202
    done_idxs = np.where(np.vstack(dones))[0]
    diffs = np.diff(done_idxs)
    incomplete_idxs = np.where(diffs == max_ep_len)[0]
    
    pruned_targets = []
    
    for i in range(len(done_idxs)):
        if i == 0:
            done_targets = targets[:done_idxs[i]]
        else:
            done_targets = targets[done_idxs[i-1]:done_idxs[i]]
        
        if i-1 not in incomplete_idxs:
            pruned_targets.append(done_targets)
    
    return pruned_targets



def split_by_ep(targets, dones):
    '''
    Similar to prune_fail_episodes, use targets=dones to split a
    recorded set of values from evaluation by episodes
    '''
    done_idxs = np.where(np.vstack(dones))[0]
    split_targets = []
    for i in range(len(done_idxs)):
        if i == 0:
            done_targets = targets[:done_idxs[i]]
        else:
            done_targets = targets[done_idxs[i-1]:done_idxs[i]]

        split_targets.append(done_targets)
    return split_targets


def draw_box(env=None, corner=None, size=None, ax=None, c='y'):
    '''
    Draw a box to an axis. By default, use static platform location
    and size for platform
    '''
    if env is not None:
        for box in env.boxes:
            if box.is_goal:
                corner = box.corner
                size = box.size
                break
    else:
        if corner is None:
            corner = np.array([240, 60])
        if size == None:
            size = [20, 20]
    if type(size) != list:
        size = [size, size]
        
    box_array = np.vstack([
        corner,
        corner+np.array([0, size[1]]),
        corner+np.array([size[0], size[1]]),
        corner+np.array([size[0], 0]),
        corner
    ]).T
    
    if ax == None:
        plt.plot(box_array[0], box_array[1], c=c)
    else:
        ax.plot(box_array[0], box_array[1], c=c)
    
    
        
def dones_to_timesteps(dones):
    '''
    Given a set of dones, create episode-split timesteps that increment
    for each timestep (just used for plotting)
    '''
    i=1
    steps = []
    for j, done in enumerate(dones):
        if done[0] == True:
            i = 1
        steps.append(i)
        i += 1
    steps = np.array(steps)
    return steps