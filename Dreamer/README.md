# Pixyz implementation of RL Dreamer
### Original Paper:  Dream to control: Learning behaviors by latent imagination(Hafner, et al., 2019)
#### https://deepmind.com/research/publications/Dream-to-Control-Learning-Behaviors-by-Latent-Imagination

#### Original Implementation (Pytorch): https://github.com/yusukeurakami/dreamer-pytorch

## Requirements
- Python >=3.6
- Pytorch
- Pixyz = 0.3.1
- TensorBoardX
- tqdm
- torchvision
- gym
- dm_control

You can try dreamer on gym envs and deepmind control suite envs. Available envs are listed in the env.py file.
## How to Train
```
python main.py --algo=dreamer --env=walker-walk --action-repeat=2 --experience-size 1000 --seed 0
```
Note: In case you run out of memory during training, try smaller experience-size option.
## Included Files
### env.py
Wrappers of gym/dmc envs

### main.py
Training scripts

### memory.py
Replay Buffer

### models.py
Networks

### planner.py
Model-predictive control planner

### utils.py
Utility functions
