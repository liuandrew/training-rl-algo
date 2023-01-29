from gradients import *
from read_experiments import *
from representation_analysis import *
import pickle
from tqdm import tqdm

# Old ref grad collection sizes
# collect_batch = 2048
# collect_processes = 4

# Collect batches more akin to training
ref_collect_batch = 256
collect_batch = 16
collect_processes = 100

"""Set up storage object
"""
batch_sizes = [16, 32]
# aux_tasks = ['catfacewall', 'catquad', 'catwall01', 'catwall0', 'catwall1']
# aux_tasks = ['wall0', 'wall1', 'wall01', 'goaldist', 'terminal']
# auxiliary_truth_sizes = [[1], [1], [1, 1],  [1], [1]]

aux_tasks = ['none', 'catfacewall', 'catquad', 'catwall01', 'catwall0', 'catwall1']


trials = range(10)
# all_chks = {16: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
#         170, 190, 210, 230, 250, 270, 300, 350, 400, 450, 500, 550, 600,
#         650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800],
#             32: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
#         170, 190, 210, 230, 250, 270, 300, 350, 400, 450, 500, 550, 600,
#         650, 700, 750, 800, 850, 900]}

#Much fewer checkpoints to test since they mostly result in the same things anyways
all_chks = {16: [0, 50, 100, 150, 300, 600, 1000, 1500],
            32: [0, 20, 40, 80, 150, 300, 600, 900]}


keep_keys = ['actions',  'data',  'dones']


all_res = {}
all_res_ref = {}

for i, batch in enumerate(batch_sizes):
    all_res[batch] = {}
    all_res_ref[batch] = {}
    for aux in aux_tasks:
        all_res[batch][aux] = {}
        all_res_ref[batch][aux] = {}
        for trial in trials:
            all_res[batch][aux][trial] = {}
            all_res_ref[batch][aux][trial] = {}
            
            
if __name__ == '__main__':
    for i, batch in enumerate(batch_sizes):
        for j, aux in enumerate(aux_tasks):
            for trial in tqdm(trials):
                print(f'Batch {batch}, aux {aux}, trial {trial}')
                
                '''Compute grads ref grads for categorical grads'''
                
                # exp_name = f'nav_pdistal_auxcat/nav_pdistal_batch{batch}aux{aux}coef1'
                # exp_name_t = f'{exp_name}_t{trial}'
                
                # # _, _, env_kwargs = load_model_and_env(exp_name, 0)
                # env_kwargs = pickle.load(open(f'../trained_models/ppo/{exp_name}_env', 'rb'))
                # chk_path = Path('../trained_models/checkpoint/')/exp_name_t
                # chks = all_chks[batch]
                
                # #Load first checkpoint and warm up reward normalization
                # model, obs_rms = torch.load(chk_path/f'{chks[0]}.pt')
                # agent, envs, storage = initialize_ppo_training(model, obs_rms, env_kwargs=env_kwargs, num_steps=128,
                #                                         num_processes=collect_processes, ppo_epoch=1, take_optimizer_step=False, normalize=True,
                #                                             agent_base='DecomposeGradPPOAux', new_aux=True,
                #                                             auxiliary_truth_sizes=auxiliary_truth_sizes[j], log_dir='tmp')
                # obs = storage.obs[0].clone()
                # res = collect_batches_and_grads(agent, envs, storage, 20, decompose_grads=True, new_aux=True)

                # cs_means = []
                # for chk in tqdm(chks):
                #     model, obs_rms = torch.load(chk_path/f'{chk}.pt')
                #     obs = storage.obs[-1].clone()
                #     envs.venv.obs_rms = obs_rms
                    
                #     #Collect small batches
                #     agent, _, storage = initialize_ppo_training(model, obs_rms, env_kwargs=env_kwargs, num_steps=collect_batch,
                #                                             num_processes=collect_processes, ppo_epoch=1, take_optimizer_step=False, normalize=True,
                #                                             make_env=False, obs=obs, agent_base='DecomposeGradPPOAux', new_aux=True, 
                #                                                 auxiliary_truth_sizes=auxiliary_truth_sizes[j])


                #     res = collect_batches_and_grads(agent, envs, storage, num_batches=20, decompose_grads=True,
                #                                     new_aux=True)
                #     all_res[batch][aux][trial][chk] = res
                    
                #     #Collect ref batches
                #     agent, _, storage = initialize_ppo_training(model, obs_rms, env_kwargs=env_kwargs, num_steps=ref_collect_batch,
                #                                             num_processes=collect_processes, ppo_epoch=1, take_optimizer_step=False, normalize=True,
                #                                             make_env=False, obs=obs, agent_base='DecomposeGradPPOAux', new_aux=True, 
                #                                                 auxiliary_truth_sizes=auxiliary_truth_sizes[j])


                #     res = collect_batches_and_grads(agent, envs, storage, num_batches=3, decompose_grads=True,
                #                                     new_aux=True)
                #     all_res_ref[batch][aux][trial][chk] = res
                
                
                ''' Compute ref grads for old auxiliary task methods '''
                    
                # exp_name = f'nav_pdistal_batchauxcoef1/nav_pdistal_batch{batch}aux{aux}coef1'
                # exp_name_t = f'{exp_name}_t{trial}'
                
                # env_kwargs = pickle.load(open(f'../trained_models/ppo/{exp_name}_env', 'rb'))
                # chk_path = Path('../trained_models/checkpoint/')/exp_name_t
                # chks = all_chks[batch]
                

                # #Load first checkpoint and warm up reward normalization
                # model, obs_rms = torch.load(chk_path/f'{chks[0]}.pt')
                # agent, envs, storage = initialize_ppo_training(model, obs_rms, env_kwargs=env_kwargs, num_steps=128,
                #                                         num_processes=collect_processes, ppo_epoch=1, take_optimizer_step=False, normalize=True,
                #                                             agent_base='DecomposeGradPPO', log_dir='tmp', new_aux=False)
                # obs = storage.obs[0].clone()
                # res = collect_batches_and_grads(agent, envs, storage, 20, decompose_grads=True)


                # cs_means = []
                # for chk in tqdm(chks):
                #     model, obs_rms = torch.load(chk_path/f'{chk}.pt')
                #     obs = storage.obs[-1].clone()
                #     envs.venv.obs_rms = obs_rms
                #     agent, _, storage = initialize_ppo_training(model, obs_rms, env_kwargs=env_kwargs, num_steps=collect_batch,
                #                                             num_processes=collect_processes, ppo_epoch=1, take_optimizer_step=False, normalize=True,
                #                                             make_env=False, obs=obs, agent_base='DecomposeGradPPO', new_aux=False)


                #     res = collect_batches_and_grads(agent, envs, storage, num_batches=20,
                #                                     decompose_grads=True)
                #     all_res[batch][aux][trial][chk] = res


                #     # Collect ref batches
                #     agent, _, storage = initialize_ppo_training(model, obs_rms, env_kwargs=env_kwargs, num_steps=ref_collect_batch,
                #                                             num_processes=collect_processes, ppo_epoch=1, take_optimizer_step=False, normalize=True,
                #                                             make_env=False, obs=obs, agent_base='DecomposeGradPPO', new_aux=False)


                #     res = collect_batches_and_grads(agent, envs, storage, num_batches=3,
                #                                     decompose_grads=True)
                #     all_res_ref[batch][aux][trial][chk] = res


                '''Collect trajectories'''
                chks = all_chks[batch]
                
                new_aux = True
                if aux == 'none':
                    exp_name = f'nav_pdistal_batchaux/nav_pdistal_batch{batch}aux{aux}'
                    new_aux = False
                else:
                    exp_name = f'nav_pdistal_auxcat/nav_pdistal_batch{batch}aux{aux}coef1'
                    
                exp_name_t = f'{exp_name}_t{trial}'
                env_kwargs = pickle.load(open(f'../trained_models/ppo/{exp_name}_env', 'rb'))
                chk_path = Path('../trained_models/checkpoint/')/exp_name_t
                
                for chk in chks:
                    if chk in all_res[batch][aux][trial]:
                        continue
                    model, obs_rms = torch.load(chk_path/f'{chk}.pt')
                    res = evalu(model, obs_rms, env_kwargs=env_kwargs, new_aux=new_aux, 
                        auxiliary_truth_sizes=[1], n=100, data_callback=nav_data_callback,
                        eval_log_dir='tmp')
                    
                    for key in list(res.keys()):
                        if key not in keep_keys:
                            del res[key]
                            
                    all_res[batch][aux][trial][chk] = res

                
            print('saving...')
            # pickle.dump(all_res, open(f'data/grads/auxcat_100grads', 'wb'))
            # pickle.dump(all_res_ref, open(f'data/grads/auxcat_100refgrads', 'wb'))
            
            # pickle.dump(all_res, open(f'data/grads/aux_100grads', 'wb'))
            # pickle.dump(all_res_ref, open(f'data/grads/aux_100refgrads', 'wb'))
            
            
    '''Clone a condensed version of trajectory data by stacking arrays'''
    condensed = {}
    for batch in batch_sizes:
        condensed[batch] = {}
        for aux in aux_tasks:
            condensed[batch][aux] = {}
            for trial in trials:
                condensed[batch][aux][trial] = {}

    for batch in all_res:
        for aux in all_res[batch]:
            for trial in all_res[batch][aux]:
                for chk in all_res[batch][aux][trial]:
                    condensed[batch][aux][trial][chk] = {}
                    condensed[batch][aux][trial][chk]['actions'] = torch.vstack(all_res[batch][aux][trial][chk]['actions'])
                    condensed[batch][aux][trial][chk]['dones'] = np.vstack(all_res[batch][aux][trial][chk]['dones'])
                    condensed[batch][aux][trial][chk]['data'] = {}
                    condensed[batch][aux][trial][chk]['data']['pos'] = np.vstack(all_res[batch][aux][trial][chk]['data']['pos'])
                    condensed[batch][aux][trial][chk]['data']['angle'] = np.vstack(all_res[batch][aux][trial][chk]['data']['angle'])
                    
          
    pickle.dump(condensed, open('data/trajectories/auxcatcoef1', 'wb'))
