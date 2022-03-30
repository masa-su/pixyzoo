import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from model import RSSM
from memory import EpisodeBuffer
from utils import get_pca_model, plot_pca_result, plot_observation_reconstruction_imagination_pca, plot_image_and_sound, sound_plot_process, convined_mp4_wav


def run(cfg, model_path):
    cfg.main.wandb = False

    for k, v in cfg.items():
        print(" " * 26 + k + ": " + str(v))

    cwd = hydra.utils.get_original_cwd()
    results_dir = "{}/results/multimodal_results".format(cwd)
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device(cfg.main.device)

    logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)

    # fft parameter
    sr = 16000
    fft_size = 1024
    frame_period = 5  # ms
    target_hz = 10
    hop_length = int(0.001 * sr * frame_period)
    frame_num = int((1 / target_hz) / (0.001 * frame_period))

    # Load model
    env = None
    rssm_model = RSSM(cfg, device)
    rssm_model.load_model(os.path.join(cwd, model_path))

    print(" " * 26 + "Options")
    # print(cfg)
    train_dataset_dir = "dataset/train/pack"
    test_dataset_dir = "dataset/test/pack"
    D = EpisodeBuffer(cfg, os.path.join(cwd, train_dataset_dir), os.path.join(cwd, test_dataset_dir), device)
    D.load_dataset_cobbota()

    n_data = D.n_train_epi
    recons = []
    for epi_idx in tqdm(range(n_data)):
        observations, actions, rewards, nonterminals = D.train_epi(epi_idx)
        recon = rssm_model.reconstruction_multimodal(observations, actions, rewards, nonterminals)
        recons.append(recon)
    beliefs = [recons[i]["states"]["belief"][:, 0] for i in range(n_data)]
    post_mean = [recons[i]["states"]["post"]["mean"][:, 0] for i in range(n_data)]

    pca_belief = get_pca_model(beliefs)
    pca_post_mean = get_pca_model(post_mean)

    plot_pca_result(results_dir, pca_belief, pca_post_mean, beliefs, post_mean, n_data)

    t_imag_start = 0
    n_frame = None  # if n_frame is not None, the animation length is limited the n_frame.

    pbar = tqdm(range(D.n_train_epi), desc="plot train data result")
    for idx_epi in pbar:
        observations, actions, rewards, nonterminals = D.train_epi(idx_epi)
        planning_horizon = len(observations["image"]) - t_imag_start - 2

        output = rssm_model.observation_reconstruction_imagination(observations, actions, rewards, nonterminals, t_imag_start, planning_horizon)
        observations_clip, recons_clip, imags = output
        save_folder_name = "{}/{}".format(results_dir, D.result_path_train(idx_epi))
        os.makedirs(save_folder_name, exist_ok=True)
        pbar.set_description("plot_observation_reconstruction_imagination_pca")
        plot_observation_reconstruction_imagination_pca(pca_belief, pca_post_mean, observations_clip, recons_clip, imags, planning_horizon, t_imag_start, save_folder_name, n_frame=n_frame)
        plot_observation_reconstruction_imagination_pca(pca_belief, pca_post_mean, observations_clip, recons_clip, imags, planning_horizon, t_imag_start, save_folder_name, n_frame=n_frame, dim=3)
        pbar.set_description("plot_image_and_sound")
        plot_image_and_sound(observations_clip, "Observation", planning_horizon, t_imag_start, save_folder_name, n_frame=n_frame)
        plot_image_and_sound(recons_clip, "Reconstruction", planning_horizon, t_imag_start, save_folder_name, n_frame=n_frame)
        plot_image_and_sound(imags, "Imagination", planning_horizon, t_imag_start, save_folder_name, n_frame=n_frame)
        pbar.set_description("sound_plot_process")
        sound_plot_process(observations_clip, recons_clip, imags, save_folder_name)
        pbar.set_description("convined_mp4_wav")
        convined_mp4_wav(save_folder_name, results_dir, cwd)
    pbar.close()

    pbar = tqdm(range(D.n_test_epi), desc="plot test data result")
    for idx_epi in pbar:
        observations, actions, rewards, nonterminals = D.test_epi(idx_epi)
        planning_horizon = len(observations["image"]) - t_imag_start - 2

        output = rssm_model.observation_reconstruction_imagination(observations, actions, rewards, nonterminals, t_imag_start, planning_horizon)
        observations_clip, recons_clip, imags = output
        save_folder_name = "{}/{}".format(results_dir, D.result_path_test(idx_epi))
        os.makedirs(save_folder_name, exist_ok=True)
        pbar.set_description("plot_observation_reconstruction_imagination_pca")
        plot_observation_reconstruction_imagination_pca(pca_belief, pca_post_mean, observations_clip, recons_clip, imags, planning_horizon, t_imag_start, save_folder_name, n_frame=n_frame)
        plot_observation_reconstruction_imagination_pca(pca_belief, pca_post_mean, observations_clip, recons_clip, imags, planning_horizon, t_imag_start, save_folder_name, n_frame=n_frame, dim=3)
        pbar.set_description("plot_image_and_sound")
        plot_image_and_sound(observations_clip, "Observation", planning_horizon, t_imag_start, save_folder_name, n_frame=n_frame)
        plot_image_and_sound(recons_clip, "Reconstruction", planning_horizon, t_imag_start, save_folder_name, n_frame=n_frame)
        plot_image_and_sound(imags, "Imagination", planning_horizon, t_imag_start, save_folder_name, n_frame=n_frame)
        pbar.set_description("sound_plot_process")
        sound_plot_process(observations_clip, recons_clip, imags, save_folder_name)
        pbar.set_description("convined_mp4_wav")
        convined_mp4_wav(save_folder_name, results_dir, cwd)
    pbar.close()


@hydra.main(config_path="", config_name="config")
def main(cfg: DictConfig):
    model_path = "results/excute_test/cobotta/2022-03-30/run_0/models_20000.pth"
    # result_path = "./Sound_Visual_RSSM"
    run(cfg, model_path)


if __name__ == "__main__":
    main()
