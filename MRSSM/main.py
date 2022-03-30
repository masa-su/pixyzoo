import sys
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import wandb

from model import RSSM
from memory import FixedExperienceReplay_Multimodal
from utils import log_params_from_omegaconf_dict, get_base_folder_name, get_git_hash

torch.backends.cudnn.benchmark = True


def train(cfg, cwd):
    # Setup
    print("Setup")
    results_dir = get_base_folder_name(cwd, cfg.main.experiment_name, cfg.env.env_name)
    os.makedirs(results_dir, exist_ok=True)
    cfg.main.log_dir = results_dir
    OmegaConf.save(cfg, "{}/config.yaml".format(results_dir))

    np.random.seed(cfg.main.seed)
    torch.manual_seed(cfg.main.seed)
    if torch.cuda.is_available() and not cfg.main.disable_cuda:
        print("using {}".format(cfg.main.device))
        device = torch.device(cfg.main.device)
        torch.cuda.manual_seed(cfg.main.seed)
    else:
        print("using CPU")
        device = torch.device("cpu")

    metrics = {"steps": [], "episodes": [], "observations_loss_sum": [], "reward_loss": [], "kl_loss": [], "learned_step": 0}
    for name in cfg.model.observation_names:
        metrics["observation_{}_loss".format(name)] = []

    writer = SummaryWriter(results_dir)

    # Initialise training environment and experience replay memory
    print("Initialise training environment and experience replay memory")
    # env = []
    dataset_dir = os.path.join(cwd, cfg.train.experience_replay)
    print("load dataset from {}".format(dataset_dir))
    if cfg.train.experience_replay != "" and os.path.exists(dataset_dir):
        print("load dataset from {}".format(dataset_dir))
        D = FixedExperienceReplay_Multimodal(
            size=cfg.train.experience_size,
            observation_names=cfg.model.observation_names,
            observation_shapes=cfg.env.observation_shapes,
            action_size=cfg.env.action_size,
            bit_depth=cfg.env.bit_depth,
            device=device,
            dataset_type=cfg.env.dataset_type,
        )

        D.load_dataset(
            dataset_dir,
            int(cfg.env.max_episode_length / cfg.env.action_repeat),
            cfg.train.n_episode,
            cfg.train.n_episode_per_data,
            cfg.train.n_augment,
        )

        metrics["steps"], metrics["episodes"] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
    elif not cfg.main.test:
        raise NotImplementedError

    # Initialise model parameters randomly
    print("Initialise model parameters randomly")
    model = RSSM(cfg, device)

    if cfg.main.wandb:
        wandb.watch(model.model_modules)

    for itr in tqdm(range(1, cfg.train.train_iteration + 1), desc="train"):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.fix_sample(cfg.train.batch_size, cfg.train.chunk_size, itr)  # Transitions start at time t = 0

        step = itr * cfg.train.batch_size * cfg.train.chunk_size
        loss_info = model.optimize(observations, actions, rewards, nonterminals, writer, step)

        # Logging losses
        for key in loss_info:
            if key not in metrics.keys():
                metrics[key] = []
            metrics[key].append(loss_info[key])
            if cfg.main.wandb:
                wandb.log(data={key: loss_info[key]}, step=step)
        metrics["learned_step"] = step

        # Checkpoint models
        if itr % cfg.main.checkpoint_interval == 0:
            model.save_model(results_dir, itr)
            np.save(os.path.join(results_dir, "metrics_%d.npy" % itr), metrics)

    # Close training environment
    # env.close()


def run(cfg):
    if cfg.main.experiment_name == None:
        print("Please set experiment_name")
        quit()
    # Overshooting distance cannot be greater than chunk size
    cfg.model.overshooting_distance = min(cfg.train.chunk_size, cfg.model.overshooting_distance)

    cwd = hydra.utils.get_original_cwd()

    # ---------- ML Flow setting ----------
    # mlrunsディレクトリ指定
    tracking_uri = cwd + "/mlruns"  # パス
    mlflow.set_tracking_uri(tracking_uri)
    # experiment指定
    mlflow.set_experiment(cfg.main.experiment_name)

    mlflow.start_run()

    if cfg.main.wandb:
        wandb_dir = os.path.join(cwd, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(project=cfg.main.experiment_name, config=cfg, dir=wandb_dir)

    hash = get_git_hash()
    mlflow.log_param("git_hash", hash)

    print(" " * 26 + "Options")
    for k, v in cfg.items():
        print(" " * 26 + k + ": " + str(v))

    log_params_from_omegaconf_dict(cfg)

    # ---------- train RSSM ----------
    train(cfg, cwd)

    # ---------- ML Flow log ----------
    mlflow.log_artifact(os.path.join(os.getcwd(), ".hydra/config.yaml"))
    mlflow.log_artifact(os.path.join(os.getcwd(), ".hydra/hydra.yaml"))
    mlflow.log_artifact(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
    mlflow.log_artifact(os.path.join(os.getcwd(), "main.log"))

    mlflow.end_run()

    if cfg.main.wandb:
        wandb.finish(exit_code=0)


@hydra.main(config_path="", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
