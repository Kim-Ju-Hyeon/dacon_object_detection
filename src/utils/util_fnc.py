import numpy as np
import torch
import cv2

def squeeze_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor.squeeze()
    return tensor


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    video = np.stack(frames)
    print(video.shape)
    return video.swapaxes(0, 3)


def box_denormalize(x1, y1, x2, y2, width, height, img_size=512):
    x1 = (x1 / img_size) * width
    y1 = (y1 / img_size) * height
    x2 = (x2 / img_size) * width
    y2 = (y2 / img_size) * height
    return x1.item(), y1.item(), x2.item(), y2.item()