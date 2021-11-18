import copy
import glob
import os
import time
from collections import deque

import gym
import gym_nav
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from scheduler import write_latest_exp_complete


def main():
    args = get_args()

    #Andy: setup W&B and tensorboard
    if args.exp_name is not None:
        run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = f"{args.env_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        logdir = wandb.run.dir
        writer = SummaryWriter(logdir)
    else:
        writer = SummaryWriter(f"runs/{run_name}")

    #Setup video uploading
    vid_dir = 'video'

    for f in os.listdir(vid_dir):
        os.remove(os.path.join(vid_dir, f))


    # def file_available(f):
    #     if os.path.exists(f):
    #         try:
    #             os.rename(f, f)
    #             return True
    #         except OSError as e:
    #             return False

    def available_videos():
        vid_dir = 'video'
        unuploaded_vids = []
        idxs = []
        for f in os.listdir(vid_dir):
            if '.mp4' in f and f not in logged_videos:
                unuploaded_vids.append(f)

        if len(unuploaded_vids) > 1:
            for f in unuploaded_vids:
                idxs.append(int(f.split('.mp4')[0].split('-')[-1]))
            f = unuploaded_vids[np.argmin(idxs)]
            return os.path.join(vid_dir, f), f
        else:
            return False

    def upload_videos():
        available = available_videos()
        if available is not False:
            path = available[0]
            f = available[1]
            wandb.log({'video': wandb.Video(path),
                'format': 'gif'})
            logged_videos.append(f)
                        

    logged_videos = []
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    # Andy: generate save path ahead of time
    if args.save_subdir is not None:
        save_path = os.path.join(args.save_dir, args.save_subdir)
        try:
            os.makedirs(save_path)
        except OSError:
            pass
    else:
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

    if args.save_name is not None:
        save_path = os.path.join(save_path, args.save_name + '.pt')
    else:
        save_path = os.path.join(save_path, args.env_name)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, capture_video=args.capture_video,
                         env_kwargs=args.env_kwargs)

    loaded_model = False
    if args.cont:
        loaded_model = True
        actor_critic, obs_rms = torch.load(save_path)
    if not loaded_model or not args.cont
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    #Andy: add global step
    global_step = 0
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            #Andy: add global step
            global_step += 1 * args.num_processes
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    print(f'global_step={global_step}')
                    # Andy: add tensorboard writing episode returns
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], 
                        global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], 
                        global_step)



            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        if args.algo == 'ppo':
            value_loss, action_loss, dist_entropy, approx_kl, clipfracs = \
            agent.update(rollouts)

        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        #Andy: add tensorboard data tracking
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss, global_step)
        writer.add_scalar("losses/policy_loss", action_loss, global_step)
        writer.add_scalar("losses/entropy", dist_entropy, global_step)

        if args.algo == 'ppo':
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start)), global_step)

        #Attempt to manually log any videos to W&B if they exist
        if args.track and args.capture_video and 'video' in os.listdir():
            upload_videos()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":

            # Andy: change save path to use more args
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], save_path)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))



        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

    #Finally log all remaining videos
    if args.track and args.capture_video and 'video' in os.listdir():
        time.sleep(5)
        upload_videos()

    if args.config_file_name is not None:
        write_latest_exp_complete(args.config_file_name)
        print('Experiment completed, experiment log updated')


if __name__ == "__main__":
    main()
