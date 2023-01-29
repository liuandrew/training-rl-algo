import torch
import gym
import gym_nav
import sys
from pathlib import Path
sys.path.append('..')
from a2c_ppo_acktr.envs import make_vec_envs
import pickle
from gradients import initialize_ppo_training

model=None
obs_rms=None
env_name='NavEnv-v0'
env_kwargs={}
make_env=True

agent_base='LoudPPO'
nn_base_kwargs={}
recurrent=True

num_steps=10
num_processes=1
seed=0
ppo_epoch=4
clip_param=0.5

num_mini_batch=2
value_loss_coef=0.5
entropy_coef=0.01

auxiliary_loss_coef=0.3
gamma=0.99
lr=7e-4
eps=1e-5
max_grad_norm=0.5

log_dir='/tmp/gym/'
device=torch.device('cpu')

capture_video=False
take_optimizer_step=True

normalize=True
obs=None
aux_wrapper_kwargs={}
new_aux=True

auxiliary_truth_sizes=[]

batch = 16
aux = 'catfacewall'
trial = 0

exp_name = f'nav_pdistal_auxcat/nav_pdistal_batch{batch}aux{aux}'
exp_name_t = f'nav_pdistal_auxcat/nav_pdistal_batch{batch}aux{aux}_t{trial}'

# _, _, env_kwargs = load_model_and_env(exp_name, 0)
env_kwargs = pickle.load(open(f'../trained_models/ppo/{exp_name}_env', 'rb'))
chk_path = Path('../trained_models/checkpoint/')/exp_name_t

print(1)
#Load first checkpoint and warm up reward normalization
model, obs_rms = torch.load(chk_path/f'{0}.pt')

if __name__ == '__main__':
    collect_processes = 2
    # envs = make_vec_envs(env_name, seed, num_processes, gamma, log_dir, device, False,
    #                             capture_video=capture_video, env_kwargs=env_kwargs, normalize=normalize,
    #                             **aux_wrapper_kwargs)
    agent, envs, storage = initialize_ppo_training(model, obs_rms, env_kwargs=env_kwargs, num_steps=128,
                                            num_processes=collect_processes, ppo_epoch=1, take_optimizer_step=False, normalize=True,
                                                    agent_base='DecomposeGradPPOAux', new_aux=True,
                                                    auxiliary_truth_sizes=[1])
    print('success')