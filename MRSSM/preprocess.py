# MP4 files tranceform wav files and finally convert numpy array
# And Packing same time spectrogram and observation
# -Directory--------------------------------------- #
# dataset
# ├── test
# │   └── mp4
# │       ├── ***.mp4
# │       ├── ***.mp4
# │       ...
# │       └── ***.mp4
# └── train
#     └── mp4
#         ├── ***.mp4
#         ├── ***.mp4
#         ...
#         └── ***.mp4
# ------------------------------------------------- #
import os
import cv2
import glob
import ffmpeg
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm


def make_wavfile(mp4_path, wav_path, overwite=True) -> None:
    # save as wav file
    mv = ffmpeg.input(mp4_path)
    mv = ffmpeg.output(mv, wav_path)
    ffmpeg.run(mv, overwrite_output=overwite, quiet=True)


def make_dummy(data_length):
    # dummy data
    action = np.zeros((data_length, 1)).astype(np.float32)
    reward = np.zeros((data_length,)).astype(np.float32)
    done = np.zeros((data_length,)).astype(np.float32)
    done[-1] = 1.0
    return action, reward, done


def image_preprocess(file_path):
    # load mp4
    cap = cv2.VideoCapture(file_path)
    # Warning : Please Check cap_file.isOpened() is True!
    if cap.isOpened() == False:
        print(file_path + " is opened : " + str(cap.isOpened()))
    # image to numpy
    image_array = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    tt_list = list(range(0, int((total_frames / fps) // 0.1)))
    target_time = tt_list.pop(0) * 0.1
    # target fps is 10Hz
    for i in range(total_frames):
        ret, frame = cap.read()
        if (i + 1) / fps >= target_time and i / fps <= target_time:
            if not tt_list:
                # last frame is ignore
                # mv_array.append(frame)
                break
            else:
                image_array.append(frame)
                target_time = tt_list.pop(0) * 0.1
    image_array = np.array(image_array).astype(np.uint8)
    return image_array


def sound_preprocess(wav_path, plot_path=None):
    # fft parameter
    sr = 16000
    fft_size = 1024
    frame_period = 5  # ms
    target_hz = 10
    hop_length = int(0.001 * sr * frame_period)
    frame_num = int((1 / target_hz) / (0.001 * frame_period))
    # create mel-spectrogram
    wav, sr = librosa.load(wav_path, sr=sr)
    mlsp = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=fft_size, hop_length=hop_length)
    mlsp = librosa.power_to_db(mlsp, ref=np.max)
    if plot_path != None:
        mlsp_plot(mlsp, plot_path, sr=sr, hop_length=hop_length)
    # divide 10Hz
    sound_array = []
    freq, total_frame = mlsp.shape
    for i in range(int(total_frame // frame_num)):
        temp = mlsp.T[frame_num * i : frame_num * (i + 1)]
        sound_array.append(temp.T)
    sound_array = np.array(sound_array).astype(np.float32)
    # sound preprocess [-0 ~ -80] -> [0 ~ 1]
    sound_array = np.divide(np.abs(sound_array), 80).astype(np.float32)
    return sound_array


def mlsp_plot(mlsp, plot_path=None, sr=16000, hop_length=80):
    fig, ax = plt.subplots(figsize=(15, 5))
    img = librosa.display.specshow(mlsp, x_axis="time", y_axis="mel", sr=sr, hop_length=hop_length, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel-frequency spectrogram")
    plt.savefig(plot_path)


def main():
    train_dataset_dir = "dataset/train"
    test_dataset_dir = "dataset/test"
    result_dir = "plot"

    train_mp4_dir = os.path.join(train_dataset_dir, "mp4")
    train_wav_dir = os.path.join(train_dataset_dir, "wav")
    train_pack_dir = os.path.join(train_dataset_dir, "pack")
    test_mp4_dir = os.path.join(test_dataset_dir, "mp4")
    test_wav_dir = os.path.join(test_dataset_dir, "wav")
    test_pack_dir = os.path.join(test_dataset_dir, "pack")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(train_wav_dir, exist_ok=True)
    os.makedirs(train_pack_dir, exist_ok=True)
    os.makedirs(test_wav_dir, exist_ok=True)
    os.makedirs(test_pack_dir, exist_ok=True)

    # find train mp4 files
    train_file_names = glob.glob(os.path.join(train_mp4_dir, "*.mp4"))
    print("find %d train mp4 files!" % len(train_file_names))
    # train process
    for file_path in tqdm(train_file_names, desc="train dataset"):
        file_title = file_path.replace(train_mp4_dir, "").replace(".mp4", "").replace("/", "")
        wav_path = os.path.join(train_wav_dir, file_title + ".wav")
        plot_path = os.path.join(result_dir, file_title + "_mel-spectrogram" + ".png")

        image_array = image_preprocess(file_path)
        make_wavfile(file_path, wav_path)
        sound_array = sound_preprocess(wav_path, plot_path)
        action, reward, done = make_dummy(data_length=len(image_array))
        save_dict = {"image": image_array, "sound": sound_array, "action": action, "reward": reward, "done": done}
        np.save(os.path.join(train_pack_dir, file_title + ".npy"), save_dict)

    # find test mp4 files
    test_file_names = glob.glob(os.path.join(test_mp4_dir, "*.mp4"))
    print("find %d test mp4 files!" % len(test_file_names))
    # test process
    for file_path in tqdm(test_file_names, desc="test dataset"):
        file_title = file_path.replace(test_mp4_dir, "").replace(".mp4", "").replace("/", "")
        wav_path = os.path.join(test_wav_dir, file_title + ".wav")
        plot_path = os.path.join(result_dir, file_title + "_mel-spectrogram" + ".png")

        image_array = image_preprocess(file_path)
        make_wavfile(file_path, wav_path)
        sound_array = sound_preprocess(wav_path, plot_path)
        action, reward, done = make_dummy(data_length=len(image_array))
        save_dict = {"image": image_array, "sound": sound_array, "action": action, "reward": reward, "done": done}
        np.save(os.path.join(test_pack_dir, file_title + ".npy"), save_dict)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
