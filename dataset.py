import os
import cv2
import numpy as np
from PIL import Image
import random

def load_frame(video_path, num_clips=1, num_frms=4):
    # Currently, this function supports only 1 clip
    assert num_clips == 1

    frame_names = sorted(os.listdir(video_path))
    total_num_frames = len(frame_names)

    # Calculate desired number of frames to extract
    desired_num_frames = min(total_num_frames, num_frms)

    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_num_frames, desired_num_frames)

    # Extract frames and get original sizes
    clip_imgs = []
    original_sizes = []
    for i in frame_idx:
        img = Image.open(os.path.join(video_path, frame_names[i]))
        clip_imgs.append(img)
        original_sizes.append(img.size)
    original_sizes = tuple(original_sizes)

    return clip_imgs, original_sizes

def read_jpg_frame( video_path, keyframe,bound=None, fps=3):
        # print(video_path)
        frame_indices = keyframe
        max_frame = len(os.listdir(video_path))
        frames = list()
        # frame_indices = get_index(None, fps, max_frame, first_idx=1) # frame_idx starts from 1
        # print(frame_indices)
        original_sizes = []
        for frame_index in frame_indices:
            frame_idx = frame_index[0]+1
            print(frame_idx)
            img = Image.open(os.path.join(video_path, f"{frame_idx:05d}.jpg"))
            print(f"{frame_idx:05d}.jpg")
            frames.append(img)
            original_sizes.append(img.size)
        # print('len(frames)',len(frames), original_sizes)
        # exit(0)
        return frames,tuple(original_sizes)

def load_video(video_path, keyframe=None, num_clips=1, num_frms=6):
    """
    Load video frames from a video file.

    Parameters:
    video_path (str): Path to the video file.
    num_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frms (int): Number of frames to extract from each clip. Defaults to 6.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video frame from a directory
    if os.path.isdir(video_path):
        return read_jpg_frame(video_path,keyframe)
    
    # Load video with OpenCV  if your device support decord package, we recommend to use decord instead of opencv: import decord
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert num_clips == 1

    desired_num_frames = num_frms

    frame_idx = []
    if keyframe:
        for k in keyframe:
            frame_idx.append(k[0])
        frame_idx.sort()
    else:
        n = total_num_frames
        m = desired_num_frames
        interval = (n - 0) / m  # Calculate interval
        frame_idx = [int(0 + i * interval) for i in range(m)]
        frame_idx.sort()

    clip_imgs = []
    original_sizes = []

    for idx in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: failed to read frame {idx} from {video_path}")
            continue
        # Convert BGR (OpenCV format) to RGB (PIL format)
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            clip_imgs.append(img)
            original_sizes.append(img.size)
        except:
            print('video_path', video_path,'\n','idx',idx)

    cap.release()

    original_sizes = tuple(original_sizes)

    return clip_imgs, original_sizes


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq