import gym
from trainer import Trainer
from model import SLAC
from constants import constants
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default='./')
    parser.add_argument("--env", type=str, default='HalfCheetah-v2')
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6)
    parser.add_argument("--cuda", action="store_true")
    args = paser.parse_args()

    env = gym.make(args.env)
    env_test = gym.make('HalfCheetah-v2')

    with open('config.yaml', 'r') as yml:
        constants = yaml.load(yml)

    slac_conf = config['SLAC']
    slac = SLAC(
        action_dim=1,
        tau=tau
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
