import gym
from trainer import Trainer
from model import SLAC
import argparse
from mockenv import MockEnv
import torch
from config import Trainer_config, SLAC_config
# from env import make_dmc


def main(args):
    if args.debug:
        env = MockEnv(obs_shape=(3, 64, 64), action_shape=(6, ), dtype="uint8")
        env_test = MockEnv(obs_shape=(3, 64, 64),
                           action_shape=(6, ), dtype="uint8")

    else:
        from env import make_dmc
        env = make_dmc(
            domain_name=args.env,
            task_name=args.task_name,
            action_repeat=args.action_repeat,
            image_size=64,
        )
        env_test = make_dmc(
            domain_name=args.env,
            task_name=args.task_name,
            action_repeat=args.action_repeat,
            image_size=64,
        )

    slac = SLAC(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        gamma=SLAC_config['gamma'],
        batch_size_sac=SLAC_config['batch_size_sac'],
        batch_size_latent=SLAC_config['batch_size_latent'],
        buffer_size=SLAC_config['buffer_size'],
        num_sequences=SLAC_config['num_sequences'],
        lr_sac=SLAC_config['lr_sac'],
        lr_latent=SLAC_config['lr_latent'],
        tau=SLAC_config['tau'],
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=slac,
        log_dir=args.log_dir,
        seed=args.seed,
        num_steps=args.num_steps,
        initial_collection_steps=Trainer_config['initial_collection_steps'],
        initial_learning_steps=Trainer_config['initial_learning_steps'],
        num_sequences=Trainer_config['num_sequences'],
        eval_interval=Trainer_config['eval_interval'],
        num_eval_episodes=Trainer_config['num_eval_episodes'])

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default='./')
    parser.add_argument("--env", type=str, default='cheetah')
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--action_repeat", type=int, default=4)
    args = parser.parse_args()
    main(args)
