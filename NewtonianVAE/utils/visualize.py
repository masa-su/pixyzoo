import os
import numpy as np

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import LinearSegmentedColormap

cdict = {
    "red": [
        (0.0, 0.0, 0.3),
        (0.25, 0.2, 0.4),
        (0.5, 0.8, 0.9),
        (0.75, 0.9, 1.0),
        (1.0, 0.4, 1.0),
    ],
    "green": [
        (0.0, 0.0, 0.2),
        (0.25, 0.2, 0.5),
        (0.5, 0.5, 0.8),
        (0.75, 0.8, 0.9),
        (1.0, 0.9, 1.0),
    ],
    "blue": [
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
}

cmap = LinearSegmentedColormap("custom", cdict, 12)


class Visualization:
    def __init__(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))

        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4

        self.frames: list = []
        self.reconstruction_images = []

    def append(self, I_t, rec_I_t, points=None):

        self.ax1.set_title(r"$ I_t $")
        self.ax1.axis('off')
        self.ax2.set_title(r"$ \hat{I}_t $")
        self.ax2.axis('off')
        self.ax4.set_title(r"$ trajectory $")

        art_1 = self.ax1.imshow(I_t)
        art_2 = self.ax2.imshow(rec_I_t)
        if points is not None:
            art_4 = self.ax4.scatter(
                points[:, 0], points[:, 1], s=1., c=cmap(np.arange(0, len(points))/100))

            self.frames.append([art_1, art_2, art_4])
        else:
            self.frames.append([art_1, art_2])

        self.reconstruction_images.append(rec_I_t)

    def encode(self, save_path, file_name):
        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass

        ani = ArtistAnimation(self.fig, self.frames, interval=100)
        ani.save(f"{save_path}/{file_name}", writer="ffmpeg")
        plt.cla()
        self.frames = []

    def add_images(self, writer, epoch):
        writer.add_images("reconstruction_images", np.stack(self.reconstruction_images), epoch, dataformats="NHWC")
        self.reconstruction_images = []
