import torch
import numpy as np
import av
from PIL import Image


def resize_frame(frame, target_size=(256, 256)):
    pil_image = Image.fromarray(frame)
    resized_image = pil_image.resize(target_size)
    return np.array(resized_image)


def open_video(file):
    container = av.open(file)
    video = []

    for frame in container.decode(video=0):
        # Convert frame to numpy array in RGB format
        rgb_image = frame.to_rgb().to_ndarray()
        rgb_image = resize_frame(rgb_image)  # ! resize processing
        video.append(rgb_image)

    container.close()
    return torch.from_numpy(np.stack(video))
