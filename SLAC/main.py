import gym
from trainer import Trainer
from model import SLAC
import argparse
import yaml
from mockenv import MockEnv
import torch
# from env import make_dmc


def main(args):
    if args.debug:
        env = MockEnv(obs_shape=(3, 64, 64), action_shape=(6, ), dtype="uint8")
        env_test = MockEnv(obs_shape=(3, 64, 64),
                           action_shape=(6, ), dtype="uint8")

    else:
        env = make_dmc(
            domain_name=args.domain_name,
            task_name=args.task_name,
            action_repeat=args.action_repeat,
            image_size=64,
        )
        env_test = make_dmc(
            domain_name=args.domain_name,
            task_name=args.task_name,
            action_repeat=args.action_repeat,
            image_size=64,
        )

    with open('config.yml', 'r') as yml:
        config = yaml.load(yml)

    slac_conf = config['SLAC']
    slac = SLAC(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        tau=slac_conf['tau'],
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed
    )

    trainer_conf = config['Trainer']
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=slac,
        log_dir=args.log_dir,
        seed=args.seed,
        num_steps=args.num_steps,
        initial_collection_steps=trainer_conf['initial_collection_steps'],
        initial_learning_steps=trainer_conf['initial_learning_steps'],
        num_sequences=trainer_conf['num_sequences'],
        eval_interval=trainer_conf['eval_interval'],
        num_eval_episodes=trainer_conf['num_eval_episodes'])

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
