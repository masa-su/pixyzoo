import os
import numpy as np
import torch
import librosa
import librosa.display
import soundfile as sf
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mlflow
from omegaconf import DictConfig, ListConfig, OmegaConf
import datetime
import subprocess
import hydra
import wandb


# --------------------------------------------------------- #
#                       preprocess                          #
# --------------------------------------------------------- #
# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


# --------------------------------------------------------- #
#                       logger                              #
# --------------------------------------------------------- #
def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                mlflow.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f"{parent_name}.{i}", v)


def get_base_folder_name(cwd=".", experiment_name=".", task="."):
    dt_now = datetime.date.today()
    count = 0
    while True:
        base_folder_name = "{}/results/{}/{}/{}/run_{}".format(cwd, experiment_name, task, dt_now, count)
        if not os.path.exists(base_folder_name):
            print("base_folder_name: {}".format(base_folder_name))
            break
        else:
            count += 1
    return base_folder_name


def get_git_hash():
    cmd = "git rev-parse --short HEAD"
    hash = subprocess.check_output(cmd.split()).strip().decode("utf-8")
    return hash


def init_logger(cfg):
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
        wandb.init(project=cfg.main.experiment_name, config=cfg)
    hash = get_git_hash()
    mlflow.log_param("git_hash", hash)
    print(" " * 26 + "Options")
    # print(cfg)
    for k, v in cfg.items():
        print(" " * 26 + k + ": " + str(v))
    log_params_from_omegaconf_dict(cfg)
    return cwd


def end_logger():
    mlflow.log_artifact(os.path.join(os.getcwd(), ".hydra/config.yaml"))
    mlflow.log_artifact(os.path.join(os.getcwd(), ".hydra/hydra.yaml"))
    mlflow.log_artifact(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
    mlflow.log_artifact(os.path.join(os.getcwd(), "main.log"))
    mlflow.end_run()


def setup(cfg, cwd):
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
    return results_dir, device


# --------------------------------------------------------- #
#                       Visualizer                          #
# --------------------------------------------------------- #
def to_np(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return data


def image_postprocess(image, bit_depth=5):
    if torch.is_tensor(image):
        image = to_np(image)
    image = postprocess_observation(image, bit_depth=bit_depth).transpose(1, 2, 0)
    return image


def sound_postprocess(sound):
    if torch.is_tensor(sound):
        sound = to_np(sound)
    sound_mel = np.multiply(sound, -80)
    return sound_mel


def mlsp2wav(sound, sr, fft_size, hop_length):
    import librosa

    if torch.is_tensor(sound):
        sound = to_np(sound)
    sound_mel = np.multiply(sound, -80)
    sound_mel = librosa.db_to_power(sound_mel)
    sound_wav = librosa.feature.inverse.mel_to_audio(sound_mel, sr=sr, n_fft=fft_size, hop_length=hop_length)
    return sound_wav, sound_mel


def tensor_cat(tensor):
    n_frame = tensor.size()[0]
    for i in range(n_frame):
        if i == 0:
            cat_tensor = tensor[i]
        else:
            cat_tensor = torch.cat((cat_tensor, tensor[i]), 1)
    return cat_tensor


def get_xyz(feat):
    feat_flat = flat(feat)
    if not (feat_flat.shape[1] == 3):
        print("error")
    return feat_flat[:, 0], feat_flat[:, 1], feat_flat[:, 2]


def flat(feat):
    feat_size = feat.shape[-1]
    return feat.reshape(-1, feat_size)


def to_np(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return data


def create_video(video_path, wav_path, out_path, overwite=True) -> None:
    import ffmpeg

    instream_v = ffmpeg.input(video_path)
    instream_a = ffmpeg.input(wav_path)
    stream = ffmpeg.output(instream_v, instream_a, out_path, vcodec="copy", acodec="aac")
    ffmpeg.run(stream, overwrite_output=overwite, quiet=True)


def sound_plot_process_image(observations_clip, recons_clip, imags, save_folder_name):
    # fft parameter
    sr = 16000
    fft_size = 1024
    frame_period = 5  # ms
    target_hz = 10
    hop_length = int(0.001 * sr * frame_period)
    frame_num = int((1 / target_hz) / (0.001 * frame_period))

    obs_sound_wav, obs_sound_mel = mlsp2wav(tensor_cat(observations_clip["sound"].squeeze(1)), sr=sr, fft_size=fft_size, hop_length=hop_length)
    obs_wav_wavname = "{}/obs_wavfile.wav".format(save_folder_name)
    sf.write(obs_wav_wavname, obs_sound_wav, sr)
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    img = librosa.display.specshow(obs_sound_mel, x_axis="time", y_axis="mel", sr=sr, hop_length=hop_length)
    fig.colorbar(img, ax=ax1, format="%+2.0f dB")
    ax1.set_title("Ground Truth Mel-frequency spectrogram")
    save_file_name = "{}/melspectrogram.png".format(save_folder_name)
    fig.savefig(save_file_name)
    plt.close()
    # plot waveform
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    librosa.display.waveplot(obs_sound_wav, sr=sr)
    ax1.set_title("Ground Truth Waveform")
    save_file_name = "{}/waveform.png".format(save_folder_name)
    fig.savefig(save_file_name)
    plt.close()


def sound_plot_process_sound(observations_clip, recons_clip, imags, save_folder_name):
    # fft parameter
    sr = 16000
    fft_size = 1024
    frame_period = 5  # ms
    target_hz = 10
    hop_length = int(0.001 * sr * frame_period)
    frame_num = int((1 / target_hz) / (0.001 * frame_period))

    obs_sound_wav, obs_sound_mel = mlsp2wav(tensor_cat(observations_clip["sound"].squeeze(1)), sr=sr, fft_size=fft_size, hop_length=hop_length)
    recon_sound_wav, recon_sound_mel = mlsp2wav(tensor_cat(recons_clip["sound"].squeeze(1)), sr=sr, fft_size=fft_size, hop_length=hop_length)
    imag_sound_wav, imag_sound_mel = mlsp2wav(tensor_cat(imags["sound"].squeeze(1)), sr=sr, fft_size=fft_size, hop_length=hop_length)

    obs_wav_wavname = "{}/obs_wavfile.wav".format(save_folder_name)
    recon_wav_wavname = "{}/recon_wavfile.wav".format(save_folder_name)
    imag_wav_wavname = "{}/imag_wavfile.wav".format(save_folder_name)

    sf.write(obs_wav_wavname, obs_sound_wav, sr)
    sf.write(recon_wav_wavname, recon_sound_wav, sr)
    sf.write(imag_wav_wavname, imag_sound_wav, sr)

    # plot mel-spectrogram
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(3, 1, 1)
    img = librosa.display.specshow(obs_sound_mel, x_axis="time", y_axis="mel", sr=sr, hop_length=hop_length)
    fig.colorbar(img, ax=ax1, format="%+2.0f dB")
    ax1.set_title("Grand Truth Mel-frequency spectrogram")
    ax2 = fig.add_subplot(3, 1, 2)
    img = librosa.display.specshow(recon_sound_mel, x_axis="time", y_axis="mel", sr=sr, hop_length=hop_length)
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Reconstruction(post) Mel-frequency spectrogram")
    ax3 = fig.add_subplot(3, 1, 3)
    img = librosa.display.specshow(imag_sound_mel, x_axis="time", y_axis="mel", sr=sr, hop_length=hop_length)
    fig.colorbar(img, ax=ax3, format="%+2.0f dB")
    ax3.set_title("Imagination Mel-frequency spectrogram")
    save_file_name = "{}/melspectrogram.png".format(save_folder_name)
    fig.savefig(save_file_name)
    plt.close()

    # plot waveform
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(3, 1, 1)
    librosa.display.waveplot(obs_sound_wav, sr=sr)
    ax1.set_title("Ground Truth Waveform")
    ax2 = fig.add_subplot(3, 1, 2)
    librosa.display.waveplot(recon_sound_wav, sr=sr)
    ax2.set_title("Reconstruction(post) Waveform")
    ax3 = fig.add_subplot(3, 1, 3)
    librosa.display.waveplot(imag_sound_wav, sr=sr)
    ax3.set_title("Imagination Waveform")
    save_file_name = "{}/waveform.png".format(save_folder_name)
    fig.savefig(save_file_name)
    plt.close()


def sound_plot_process(observations_clip, recons_clip, imags, save_folder_name, ptype=None):
    if ptype == "image":
        sound_plot_process_image(observations_clip, recons_clip, imags, save_folder_name)
    else:
        sound_plot_process_sound(observations_clip, recons_clip, imags, save_folder_name)


def convined_mp4_wav_image(save_folder_name, result_dir, cwd):
    # obs
    create_video(os.path.join(save_folder_name, "image_and_sound-Observation.mp4"), os.path.join(save_folder_name, "obs_wavfile.wav"), os.path.join(save_folder_name, "image_and_sound-Observation_sound.mp4"))
    # recon
    create_video(os.path.join(save_folder_name, "image_and_sound-Reconstruction.mp4"), os.path.join(save_folder_name, "obs_wavfile.wav"), os.path.join(save_folder_name, "image_and_sound-Reconstruction_sound.mp4"))
    # imag
    create_video(os.path.join(save_folder_name, "image_and_sound-Imagination.mp4"), os.path.join(save_folder_name, "obs_wavfile.wav"), os.path.join(save_folder_name, "image_and_sound-Imagination_sound.mp4"))
    # pca_2d <- obs, recon, imag
    create_video(os.path.join(save_folder_name, "image-imag_PCA.mp4"), os.path.join(save_folder_name, "obs_wavfile.wav"), os.path.join(save_folder_name, "image-imag_PCA_sound_truth.mp4"))
    # pca_3d <- obs, recon, imag
    create_video(os.path.join(save_folder_name, "image-imag_PCA_3d.mp4"), os.path.join(save_folder_name, "obs_wavfile.wav"), os.path.join(save_folder_name, "image-imag_PCA_3d_sound_truth.mp4"))


def convined_mp4_wav_sound(save_folder_name, result_dir, cwd):
    # obs
    create_video(os.path.join(save_folder_name, "image_and_sound-Observation.mp4"), os.path.join(save_folder_name, "obs_wavfile.wav"), os.path.join(save_folder_name, "image_and_sound-Observation_sound.mp4"))
    # recon
    create_video(os.path.join(save_folder_name, "image_and_sound-Reconstruction.mp4"), os.path.join(save_folder_name, "recon_wavfile.wav"), os.path.join(save_folder_name, "image_and_sound-Reconstruction_sound.mp4"))
    # imag
    create_video(os.path.join(save_folder_name, "image_and_sound-Imagination.mp4"), os.path.join(save_folder_name, "imag_wavfile.wav"), os.path.join(save_folder_name, "image_and_sound-Imagination_sound.mp4"))
    # pca_2d <- obs, recon, imag
    create_video(os.path.join(save_folder_name, "image-imag_PCA.mp4"), os.path.join(save_folder_name, "obs_wavfile.wav"), os.path.join(save_folder_name, "image-imag_PCA_sound_truth.mp4"))
    create_video(os.path.join(save_folder_name, "image-imag_PCA.mp4"), os.path.join(save_folder_name, "recon_wavfile.wav"), os.path.join(save_folder_name, "mimage-imag_PCA_sound_recon.mp4"))
    create_video(os.path.join(save_folder_name, "image-imag_PCA.mp4"), os.path.join(save_folder_name, "imag_wavfile.wav"), os.path.join(save_folder_name, "image-imag_PCA_sound_imag.mp4"))
    # pca_3d <- obs, recon, imag
    create_video(os.path.join(save_folder_name, "image-imag_PCA_3d.mp4"), os.path.join(save_folder_name, "obs_wavfile.wav"), os.path.join(save_folder_name, "image-imag_PCA_3d_sound_truth.mp4"))
    create_video(os.path.join(save_folder_name, "image-imag_PCA_3d.mp4"), os.path.join(save_folder_name, "recon_wavfile.wav"), os.path.join(save_folder_name, "image-imag_PCA_3d_sound_recon.mp4"))
    create_video(os.path.join(save_folder_name, "image-imag_PCA_3d.mp4"), os.path.join(save_folder_name, "imag_wavfile.wav"), os.path.join(save_folder_name, "image-imag_PCA_3d_sound_imag.mp4"))


def convined_mp4_wav(save_folder_name, result_dir, cwd, ptype=None):
    if ptype == "image":
        convined_mp4_wav_image(save_folder_name, result_dir, cwd)
    else:
        convined_mp4_wav_sound(save_folder_name, result_dir, cwd)


def get_pca_model(feat):
    pca = PCA(n_components=3)
    feat = to_np(torch.vstack(feat))
    feat_flat = flat(feat)
    pca.fit(feat_flat)
    return pca


def plot_pca_result(results_dir, pca_belief, pca_post_mean, beliefs, post_mean, n_data):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    coords = []
    for i in range(n_data):
        feat = to_np(beliefs[i])
        feat_pca = pca_belief.transform(feat)
        coords.append(get_xyz(feat_pca))
    #     x, y, z = get_xyz(feat_pca)
    for i in range(n_data):
        x, y, z = coords[i]
        ax.scatter(x, y, z, label="episode:{}".format(i), marker="x", alpha=0.1)
    plt.legend()
    plt.savefig("{}/image-result_pca_train_beliefs.png".format(results_dir))
    plt.close()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    coords = []
    for i in range(n_data):
        feat = to_np(post_mean[i])
        feat_pca = pca_post_mean.transform(feat)
        coords.append(get_xyz(feat_pca))
    #     x, y, z = get_xyz(feat_pca)
    for i in range(n_data):
        x, y, z = coords[i]
        ax.scatter(x, y, z, label="episode:{}".format(i), marker="x", alpha=0.1)
    plt.legend()
    plt.savefig("{}/image-result_pca_train_post_mean.png".format(results_dir))
    plt.close()


def plot_pca_traj(ax, traj_rec, traj_imag, t, dim=2):
    x_rec, y_rec, z_rec = get_xyz(traj_rec[: t + 1])
    x_imag, y_imag, z_imag = get_xyz(traj_imag[: t + 1])
    if dim == 3:
        ax.plot(x_rec, y_rec, z_rec, label="rec", marker="x")
        ax.plot(x_imag, y_imag, z_imag, label="imag", marker="x")
    else:
        ax.plot(x_rec, y_rec, label="rec", marker="x")
        ax.plot(x_imag, y_imag, label="imag", marker="x")
    #     ax.plot(x_imag, y_imag, z_imag, label="imag", marker=".")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    if dim == 3:
        ax.set_zlim(-10, 10)
        # ax.set_zlabel("Z-axis")


def plot_observation_reconstruction_imagination_pca(pca_belief, pca_post_mean, observations_clip, recons_clip, imags, planning_horizon, t_imag_start, save_folder_name, n_frame=None, dim=2):

    idx = 0
    if n_frame == None:
        n_frame = planning_horizon

    t_start = 0

    n_graph = 4
    # fft parameter
    sr = 16000
    fft_size = 1024
    frame_period = 5  # ms
    target_hz = 10
    hop_length = int(0.001 * sr * frame_period)
    frame_num = int((1 / target_hz) / (0.001 * frame_period))

    feat_rec_belief = to_np(recons_clip["states"]["belief"][:, 0])
    traj_rec_belief = pca_belief.transform(feat_rec_belief)

    feat_imag_belief = to_np(imags["states"]["belief"][:, 0])
    traj_imag_belief = pca_belief.transform(feat_imag_belief)

    feat_rec_post_mean = to_np(recons_clip["states"]["post"]["mean"][:, 0])
    traj_rec_post_mean = pca_post_mean.transform(feat_rec_post_mean)

    feat_imag_prior_mean = to_np(imags["states"]["prior"]["mean"][:, 0])
    traj_imag_prior_mean = pca_post_mean.transform(feat_imag_prior_mean)

    fig = plt.figure(figsize=(2.5 * n_graph, 2.5 * 2))
    ax1 = fig.add_subplot(2, n_graph, 1)
    ax1.axis("off")
    ax1.set_title("Grand Truth")
    ax2 = fig.add_subplot(2, n_graph, 2)
    ax2.axis("off")
    ax2.set_title("Reconstruction")
    ax3 = fig.add_subplot(2, n_graph, 3)
    ax3.axis("off")
    ax3.set_title("Imagination")

    if dim == 2:
        ax4 = fig.add_subplot(2, n_graph, 4)
    elif dim == 3:
        ax4 = fig.add_subplot(2, n_graph, 4, projection="3d")
    ax4.set_title("Deterministic State")

    ax5 = fig.add_subplot(2, n_graph, 5)
    ax6 = fig.add_subplot(2, n_graph, 6)
    ax6.axis("off")
    ax7 = fig.add_subplot(2, n_graph, 7)
    ax7.axis("off")
    if dim == 2:
        ax8 = fig.add_subplot(2, n_graph, 8)
    elif dim == 3:
        ax8 = fig.add_subplot(2, n_graph, 8, projection="3d")
    ax8.set_title("Stchastic State")

    def plot(t):
        t_observation = t + 2 + t_imag_start + t_start
        t_reconstruction = t_observation - 1
        t_imagination = t_observation - 2 - t_imag_start

        plt.cla()

        plt.axis("off")
        fig.suptitle("image only t={}".format(t_observation))

        ax1.imshow(image_postprocess(observations_clip["image"].squeeze(1)[t]))
        ax2.imshow(image_postprocess(recons_clip["image"].squeeze(1)[t]))
        ax3.imshow(image_postprocess(imags["image"].squeeze(1)[t]))

        ax4.cla()
        plot_pca_traj(ax4, traj_rec_belief, traj_imag_belief, t, dim=dim)

        mlsp = sound_postprocess(observations_clip["sound"].squeeze(1)[t])
        librosa.display.specshow(mlsp, x_axis="ms", y_axis="mel", sr=sr, hop_length=hop_length, ax=ax5)

        mlsp = sound_postprocess(recons_clip["sound"].squeeze(1)[t])
        librosa.display.specshow(mlsp, x_axis="ms", y_axis="mel", sr=sr, hop_length=hop_length, ax=ax6)

        mlsp = sound_postprocess(imags["sound"].squeeze(1)[t])
        librosa.display.specshow(mlsp, x_axis="ms", y_axis="mel", sr=sr, hop_length=hop_length, ax=ax7)

        ax8.cla()
        plot_pca_traj(ax8, traj_rec_post_mean, traj_imag_prior_mean, t, dim=dim)

    # create animation 10Hz
    anim = FuncAnimation(fig, plot, frames=n_frame, interval=100)
    if dim == 2:
        save_file_name = "{}/image-imag_PCA.mp4".format(save_folder_name)
    else:
        save_file_name = "{}/image-imag_PCA_3d.mp4".format(save_folder_name)
    anim.save(save_file_name, writer="ffmpeg")
    plt.close()


def plot_image_and_sound(data, data_name, planning_horizon, t_imag_start, save_folder_name, n_frame=None):
    if n_frame == None:
        n_frame = planning_horizon

    t_start = 0

    # fft parameter
    sr = 16000
    fft_size = 1024
    frame_period = 5  # ms
    target_hz = 10
    hop_length = int(0.001 * sr * frame_period)
    frame_num = int((1 / target_hz) / (0.001 * frame_period))

    fig = plt.figure(figsize=(2.5, 2.5 * 2))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.axis("off")
    ax1.set_title(data_name)
    ax2 = fig.add_subplot(2, 1, 2)

    def plot(t):
        t_observation = t + 2 + t_imag_start + t_start
        plt.cla()
        plt.axis("off")
        fig.suptitle("Image Only t={}".format(t_observation))
        ax1.imshow(image_postprocess(data["image"].squeeze(1)[t]))
        mlsp = sound_postprocess(data["sound"].squeeze(1)[t])
        librosa.display.specshow(mlsp, x_axis="ms", y_axis="mel", sr=sr, hop_length=hop_length, ax=ax2)

    # create animation 10Hz
    anim = FuncAnimation(fig, plot, frames=n_frame, interval=100)
    save_file_name = "{}/image_and_sound-{}.mp4".format(save_folder_name, data_name)
    anim.save(save_file_name, writer="ffmpeg")

    plt.close()
