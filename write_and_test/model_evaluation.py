import gym
import gym_nav
import torch
import sys
sys.path.insert(0, '..')
from evaluation import evaluate
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.envs import make_vec_envs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
import re
from matplotlib import animation
from IPython.display import HTML

from scipy.ndimage.filters import gaussian_filter
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

eval_log_dir = '/tmp/gym/_eval'
device = torch.device("cpu")

save_folder = 'plots/proof_of_concept/'

'''
General operation of functions in this file:
print_trained_models(): Run this first to get a list of trained models
    in the trained_models folder.
model, obs_rms, env_kwargs = load_model_and_env(model_name):
    For model name, pass the entire path to the trained model from ppo folder
    e.g., model_name = 'nav_auxiliary_tasks/nav_aux_wall_1_t1.pt'

Now the model has been loaded and can be evaluated
results = evalu(model, obs_rms, n=10, env_name='NavEnv-v0', env_kwargs=env_kwargs)
    This will actually evaluate the model on the given environment
    Note that we evaluate with a single environment at a time to make data_callback
        easier to work with, but if no data_callback is passed we can parallelize.
        Note also that capture video only works on the first vectorized env
    The results will be a dictionary with information about the episodes
    We can run this with capture_video=True to record videos
    We can optionally pass a data_callback to collect extra information
        (see example data_callback functions)

To dig into the results of a single episode, use the following
get_ep(results['all_obs'], results['ep_ends'], ep_num=2)
The evalu() function includes the episode ending numbers for this purpose
'''


def print_trained_models(folder='../trained_models/ppo/', ignore_non_pt=True,
                        ignore_non_dir_in_first=True):
    '''
    Read the trained_models folder to see what models have been trained
    ignore_non_pt: don't print files without .pt extension
    ignore_non_dir_in_first: don't print files in the parent folder, skip straight to directories
    '''
    
    space =  '    '
    branch = '│   '
    tee =    '├── '
    last =   '└── '
    
    path = Path(folder)
    print(path.name)
    
    def inner_print(path, depth, ignore_non_pt=ignore_non_pt, ignore_non_dir_in_first=ignore_non_dir_in_first):
        directories = []
        unique_experiments = {}
        original_experiment_names = {}
        for d in path.iterdir():
            if d.is_dir():
                directories.append(d)
            elif d.suffix == '.pt':
                if not re.match('.*\d*\.pt', d.name) and not (ignore_non_dir_in_first and depth == 0):
                    #not a trial, simply print
                    print(branch*depth+tee+d.name)
                exp_name = '_'.join(d.name.split('_')[:-1])
                if exp_name in unique_experiments.keys():
                    unique_experiments[exp_name] += 1
                else:
                    unique_experiments[exp_name] = 1
                    original_experiment_names[exp_name] = d.name
            elif not ignore_non_pt:
                print(branch*depth+tee+d.name)
        for key, value in unique_experiments.items():
            if ignore_non_dir_in_first and depth == 0:
                break
            if value > 1:
                print(branch*depth + tee+'EXP', key + ':', value)
            else:
                print(branch*depth+tee+original_experiment_names[key])
        for d in directories:
            print(branch*depth + tee+d.name)
            inner_print(d, depth+1, ignore_non_pt, ignore_non_dir_in_first)
            
    inner_print(path, 0, ignore_non_pt, ignore_non_dir_in_first)    
    
    


def load_model_and_env(experiment, 
                       base_folder='../trained_models/ppo/'):
    '''
    Load a model along with its environment
    Use the fact that all saved models end with _i where i indicates the trial number
    
    For scenario name, include any folder directions starting from trained_models/ppo/
    E.g., for the poster scenario might use scenario_name = 'invisible_poster/invisible_poster_0_shape_0_0.pt'
    
    Note that after training has been completed, we need to manually go back to write_experiments.ipynb
    to add the appropriate env kwargs to the trained models folder
    
    Return: model, obs_rms, env_kwargs
    '''
    
    env_file = '_'.join(experiment.split('_')[:-1]) + '_env'
    try:
        env_kwargs = pickle.load(open(base_folder + env_file, 'rb'))
    except:
        Exception('Error loading env_kwargs, make sure that they have been saved from write_experiments.ipynb')
    # env = gym.make('Gridworld-v0', **env_kwargs)
    
    model, obs_rms = torch.load(base_folder + experiment)
    
    return model, obs_rms, env_kwargs




def evalu(model, obs_rms, n=100, env_name='Gridworld-v0', env_kwargs={},
          data_callback=None, capture_video=False, verbose=False):
    '''
    Evaluate using the current global model, obs_rms, and env_kwargs
    Load ep_ends, ep_lens into global vars to be used by get_ep
        as well as all_obs, all_actions, all_rewards, all_hidden_states, all_dones, eval_envs, data
        
    capture_video: whether video should be captured
    verbose: print every episode
    '''
    
    results = evaluate(model, obs_rms, env_name, 1, 1, eval_log_dir, device, 
         env_kwargs=env_kwargs, data_callback=data_callback, num_episodes=n, capture_video=capture_video,
         verbose=verbose)

    #ep_ends and ep_lens used to easily pull data for single episodes
    results['ep_ends'] = np.where(np.array(results['dones']).flatten())[0]
    results['lens'] = np.diff(results['ep_ends'])
    
    # env = gym.make(env_name, **env_kwargs)

    #code for computing PCA of hidden_states over all evaluated episodes
    
    # global all_trajectories, hidden_states
    
    # hidden_states = np.zeros((len(all_hidden_states), all_hidden_states[0].shape[1]))
    # for i in range(len(all_hidden_states)):
    #     hidden_states[i] = all_hidden_states[i].numpy()[0]

    # pca = PCA(n_components=2)
    # pca.fit(hidden_states)

    # all_trajectories = pca.transform(hidden_states)

    # len(pca.transform(hidden_states))
    
    return results
    
    
    
def get_ep(data, ep_ends, ep_num=0):
    '''
    Pass data and a data block to grab data from this episode n alone
    E.g., call get_ep(hidden_states, 1) to get the hidden states for the 2nd episode
    '''
    if ep_num == 0:
        ep_start = 0
    else:
        ep_start = ep_ends[ep_num - 1]
    
    ep_end = ep_ends[ep_num]
    return data[ep_start:ep_end]

    
    
    
    
def animate_episode(ep_num=0, trajectory=False):
    #generate frames of episode
    rgb_array = []
    agent = get_ep(data['agent'], ep_num)
    goal = get_ep(data['goal'], ep_num)[0]
    for i in range(1, env.world_size[0]-1):
        for j in range(1, env.world_size[1]-1):
            env.objects[i, j] = 0
            env.visible[i, j] = 0
    env.objects[goal[0], goal[1]] = 1
    env.visible[goal[0], goal[1]] = 6

    for a in agent:
        env.agent[0][0] = a[0][0]
        env.agent[0][1] = a[0][1]
        env.agent[1] = a[1]
        rgb_array.append(env.render('rgb_array'))

    rgb_array = np.array(rgb_array)
    
    if trajectory:
        #generate trajectory
        trajectory = get_ep(all_trajectories, ep_num)
        scat_min = np.min(all_trajectories)
        scat_max = np.max(all_trajectories)


        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        
        im = ax[0].imshow(rgb_array[0,:,:,:])
        x, y = np.full(trajectory.shape[0], -100.0), np.full(trajectory.shape[0], -100.0)
        scatter = ax[1].scatter(x, y)
        ax[1].set_xlim([scat_min, scat_max])
        ax[1].set_ylim([scat_min, scat_max])

        plt.close() # this is required to not display the generated image

        def init():
            im.set_data(rgb_array[0,:,:,:])
            scatter.set_offsets(np.c_[x, y])

        def animate(i):
            im.set_data(rgb_array[i,:,:,:])
            x[:i+1] = trajectory.T[0][:i+1]
            y[:i+1] = trajectory.T[1][:i+1]
            scatter.set_offsets(np.c_[x, y])
            # print(np.c_[x, y])
            return im, scatter,

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=rgb_array.shape[0],
                                       interval=500)
    else:
        fig = plt.figure(figsize=(6,6))
        
        im = plt.imshow(rgb_array[0,:,:,:])
        plt.close() # this is required to not display the generated image

        def init():
            im.set_data(rgb_array[0,:,:,:])

        def animate(i):
            im.set_data(rgb_array[i,:,:,:])
            return im

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=rgb_array.shape[0],
                                       interval=500)

    # fig = plt.figure()
    return HTML(anim.to_html5_video())


    
    
def nav_data_callback(actor_critic, vec_envs, recurrent_hidden_states,
                                  obs, action, reward, data):
    if data == {}:
        data['pos'] = []
        # data['layer_activations'] = []
    
    pos = vec_envs.get_attr('character')[0].pos.copy()
    data['pos'].append(pos)
    
    return data
        
    
    
    
    
def data_callback(actor_critic, vec_envs, recurrent_hidden_states, 
                  obs, action, reward, data):
    """Example of a data callback, this basic one is intended
    for 'GridNav-v0'. We take these arguments and can perform
    evaluations on them to collect data during episodes

    Args:
        These arguments will always be passed by the evaluation
        function (evaluation.py)
        data: we pass in the previous dictionary of data

    Returns:
        data: pass back updated dictionary of data
    """
    # Initialize the data dictionary if it has not been initialized yet
    if data == {}:
        data['goal'] = []
        data['agent'] = []
        data['facing_goal'] = []
        data['dist_from_goal'] = []
    
    agent = vec_envs.get_attr('agent')[0]
    objects = vec_envs.get_attr('objects')[0]
    
    goal_y = (objects == 2).argmax(axis=0).max()
    goal_x = (objects == 2).argmax(axis=1).max()
    data['goal'].append([goal_y, goal_x])
    
    #check if agent is facing the goal
    correct_directions = np.array([False, False, False, False])
    if goal_y > agent[0][0]:
        correct_directions[1] = True
    if goal_y < agent[0][0]:
        correct_directions[3] = True
    if goal_x > agent[0][1]:
        correct_directions[0] = True
    if goal_x < agent[0][1]:
        correct_directions[2] = True
    
    if correct_directions[agent[1]]:
        data['facing_goal'].append(True)
    else:
        data['facing_goal'].append(False)
        
        
    dist_from_goal = np.abs(goal_y - agent[0][0]) + np.abs(goal_x - agent[0][1])
    data['dist_from_goal'].append(dist_from_goal)
    # vec_envs.env_method('render')
    
    data['agent'].append(agent.copy())
    
    return data

    
    
def get_activations(model, results):
    """After running evalu, pass the model and results
    to collect the layer activations at each timestep during experimentation
    Assuming the model is a FlexBase network

    Args:
        model (FlexBaseNN): loaded policy and model
        results (dict): dictionary from evalu() function

    Returns:
        activations: dictionary with activations from each layer
    """
    obs = torch.vstack(results['obs'][:])
    hidden_state = torch.vstack(results['hidden_states'][:])
    masks = torch.vstack(results['masks'][:])
    
    activations = model.base.forward_with_activations(obs, hidden_state, 
                                                    masks, deterministic=True)
    
    return activations
    
    
def determine_linear_fit(layer_activations, quantity):
    """Perform a linear regression between a layer's activations
    and some quantity to see if it could be linearly encoded
    within the weights

    Args:
        layer_activations (array: (samples, layer_size)): 
            activations collected over samples running
        quantity (array: (samples, quantity_size)): 
            quantity desired to check fit of
    """
    reg = LinearRegression()
    reg.fit(layer_activations, quantity)
    prediction = reg.predict(layer_activations)
    
    print(f'r2 score: {r2_score(prediction, quantity)}')
    print(f'MSE: {mean_squared_error(prediction, quantity)}')
    
    return reg


def plot_2d_activations(layer_activations, quantity, min_val=0, max_val=300,
                        grid_samples=300, sigma=10):
    """Attempt to fit a gradient boosted tree that takes a 2D
    quantity and fits it with layer activations
    Designed to see if we can predict a node activation based on position

    Args:
        layer_activations (array: (samples, layer_size)): 
            activations collected over samples running
        quantity (array: (samples, 2)): 
            2D quantity to fit and plot against
        min: min value to make grid
        max: max value to make grid
    """
    num_samples = layer_activations.shape[0]
    num_nodes = layer_activations.shape[1]
    plot_size = int(np.ceil(np.sqrt(num_nodes)))
    
    regressors = []
    
    fig, ax = plt.subplots(plot_size, plot_size, figsize=(20, 20),
                           sharex=True, sharey=True)
    grid = np.zeros((grid_samples * 2, grid_samples)).reshape(-1, 2)
    
    xs = np.linspace(min_val, max_val, grid_samples)
    ys = np.linspace(min_val, max_val, grid_samples)
    
    for i, y in enumerate(ys):
        grid[i*grid_samples:(i+1)*grid_samples] = np.concatenate(
            [xs.reshape((-1, 1)), np.full((grid_samples, 1), y)], axis=1
        )
        
    for i in range(num_nodes):
        ax_x = i % plot_size
        ax_y = i // plot_size
        
        model = xgb.XGBRegressor(max_depth=2)
        y = layer_activations[:, i]
        model.fit(quantity, y)
        regressors.append(model)
        
        grid_activations = model.predict(grid).reshape(grid_samples, grid_samples)
        smoothed = gaussian_filter(grid_activations, sigma=sigma)
        ax[ax_x, ax_y].imshow(smoothed)
        ax[ax_x, ax_y].set_xticks([])
        ax[ax_x, ax_y].set_yticks([])
    
    plt.tight_layout()
    
    return regressors
        