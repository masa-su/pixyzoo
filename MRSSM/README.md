# Pixyz implementation of Multimodal RSSM
#### RSSM Paper : Learning Latent Dynamics for Planning from Pixels, Hafner Danijar et al, 2019 (https://arxiv.org/abs/1811.04551)

## Requirements
- Python >=3.6
- Pytorch
- Pixyz
- TensorBoardX
- tqdm
- librosa
- ffmpeg
- hydra
- plotly
- wandb

## Dataset

You can use mp4 files for dataset. Please locate the files as follows.

```
dataset
├── test
│   └── mp4
│       ├── ***.mp4
│       ├── ***.mp4
│       ...
│       └── ***.mp4
└── train
    └── mp4
        ├── ***.mp4
        ├── ***.mp4
        ...
        └── ***.mp4
```

you can use these mp4 files.

https://drive.google.com/drive/folders/1Xr5f6N30XAqDl-gFWsS4K75dCS7L7C5s

## How to Train

### first step : preprocess mp4
preprocess.py convert from mp4 files to npy files
```
python preprocess.py
```
### second step : train the model
```
python main.py
```

### third step : visualize latent state

```
python visualizer.py
```

