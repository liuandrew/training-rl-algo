import gym
import gym_nav
from tqdm import tqdm
# env = gym.make('CartPole-v0')
# capture_video = 1
# env = gym.wrappers.Monitor(env, './video', 
#     video_callable=lambda t:t%capture_video==0, force=True)

# obs = env.reset()
# for i in range(202):
#     obs, reward, done, info = env.step(env.action_space.sample())
#     if done:
#         break
# env.close()

import torch
from a2c_ppo_acktr.envs import make_vec_envs
from torch.utils.tensorboard import SummaryWriter
from a2c_ppo_acktr.arguments import get_args
import os

if __name__ == '__main__':
    env_name = 'NavEnv-v0'
    seed = 1
    num_processes = 2
    gamma = 0.99
    log_dir = 'logs/'
    device = torch.device('cpu')
    capture_video = 1
    env_kwargs = None

    vid_dir = 'video'

    for f in os.listdir(vid_dir):
        os.remove(os.path.join(vid_dir, f))

    args = get_args()

    import wandb

    wandb.init(
        project='test',
        entity=None,
        sync_tensorboard=True,
        config=vars(args),
        name='test_record',
        monitor_gym=True,
        save_code=True,
    )
    logdir = wandb.run.dir
    writer = SummaryWriter(logdir)

    envs = make_vec_envs(env_name, seed, num_processes,
                         gamma, log_dir, device, False, capture_video=capture_video,
                         env_kwargs=env_kwargs)
    envs.reset()

    def file_available(f):
        if os.path.exists(f):
            try:
                os.rename(f, f)
                return True
            except OSError as e:
                return False

    def upload_videos():
        for f in os.listdir(vid_dir):
            if '.mp4' in f:
                if f not in uploaded_vids:
                    path = os.path.join(vid_dir, f)
                    if file_available(path):
                        wandb.log({'video': wandb.Video(path),
                            'format': 'gif'})
                        uploaded_vids.append(f)




    uploaded_vids = []

    for i in tqdm(range(1000)):
        actions = torch.tensor([envs.action_space.sample() for i in range(num_processes)]).to(device)
        _, _, dones, _ = envs.step(actions.reshape(2, 1))
        if dones[0]:
            print('done')
        upload_videos()


    envs.close()