# DynaNet
- Original paper: DynaNet: Neural Kalman Dynamical Model for Motion Estimation and Prediction, Changhao Chen et al., 2019
- https://arxiv.org/abs/1908.03918

WARNING: Pixyz version >= 0.1.0 is required

* dynanet.ipynb: implementation of simple Dynanet network in Pixyz

## Experiment setting
- Generate MNIST by stacking row images(consider row as time step)
    - Compared to the original paper's setting, y (output of the predictor) equals x (input of the encoder)
- Compared to the original training, we trained dynanet end-to-end
    - According to the author, for more stable training, the encoder is fixed and initiated with the pertrained weights, but not trained along with other modules.