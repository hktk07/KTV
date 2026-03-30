import os
import csv
import cv2
import av
import pickle
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu,gpu
from transformers import Dinov2Model, AutoImageProcessor
from sklearn.cluster import KMeans
import json
from torchvision import transforms
fast_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# model_name = "dinov2_vitl14"
# model = torch.hub.load('facebookresearch/dinov2', model_name)
# model.eval()
# model_path = '/home/pengjun/.cache/torch/hub/facebookresearch_dinov2_main'
# # print(2)
# # 如果有CUDA可用，则将模型移动到GPU
# device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

model_name_hf = "facebook/dinov2-large"
model = Dinov2Model.from_pretrained(model_name_hf)
hf_processor = AutoImageProcessor.from_pretrained(model_name_hf)
model.eval()
device = torch.device("cuda") #if you use ASCEND npu  device = torch.device("npu")
model.to(device)


def get_frame_indices(total_frames, max_frames):
    if total_frames <= max_frames:
        return np.arange(total_frames)
    else:
        return np.linspace(0, total_frames - 1, max_frames, dtype=int)

def get_index( bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        frame_indices = np.arange(start_idx, end_idx + 1)

        return frame_indices
    
def read_jpg_frame( video_path, bound=None, fps=3):
        print(video_path)
        max_frame = len(os.listdir(video_path))
        frames = list()
        frame_indices = get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        print(frame_indices)
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            frames.append(img)

        return frames
    
# def extract_selected_frames(video_path, indices):
#     cap = cv2.VideoCapture(video_path)
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total == 0:
#         return []

#     frames = []
#     indices_set = set(indices)
#     i = 0
#     while i < total:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if i in indices_set:
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(Image.fromarray(image))
#         i += 1
#     cap.release()
#     return frames


# def extract_selected_frames(video_path, indices):
#     """
#     用 PyAV 高效提取视频中指定索引的帧。
#     :param video_path: 视频路径
#     :param indices: 要提取的帧索引列表
#     :return: PIL.Image 列表
#     """
#     container = av.open(video_path)
#     stream = container.streams.video[0]

#     # 强制精确解码
#     stream.codec_context.skip_frame = "NONE"
#     stream.thread_type = "AUTO"

#     indices_set = set(indices)
#     frames = []
#     current_idx = 0

#     for frame in container.decode(stream):
#         if current_idx in indices_set:
#             img = frame.to_ndarray(format='rgb24')
#             frames.append(Image.fromarray(img))
#         current_idx += 1
#         if current_idx > max(indices):
#             break

#     container.close()
#     return frames

# def extract_selected_frames(video_path, indices):
#     # 创建视频读取器（可改为 gpu(0) 启用GPU加速）
#     vr = VideoReader(video_path, ctx=cpu(0))
#     total = len(vr)

#     # 过滤并排序索引
#     valid_indices = sorted([i for i in indices if 0 <= i < total])
#     if not valid_indices:
#         return []

#     # 批量读取指定帧（一次性提取，速度非常快）
#     frames = vr.get_batch(valid_indices).asnumpy()  # shape: (num_frames, H, W, 3)

#     # 转为 PIL.Image 列表
#     images = [Image.fromarray(frame) for frame in frames]
#     return images


def dino_feature_stream(video_path, model, device, indices, batch_size=128):
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []
    features = []
    current_idx = 0
    indices_set = set(indices)
    max_idx = max(indices)

    with torch.no_grad():
        for frame in container.decode(stream):
            if current_idx in indices_set:
                img = frame.to_ndarray(format='rgb24')
                img = Image.fromarray(img)
                frames.append(fast_transform(img))
                # 到达批次大小 → 提特征 + 清理
                if len(frames) >= batch_size:
                    batch_tensors = torch.stack(frames).to(device)
                    outputs = model(pixel_values=batch_tensors)
                    batch_features = outputs.pooler_output.cpu().numpy()
                    features.extend(batch_features)
                    print('process done')
                    # 清空缓存
                    del batch_tensors, frames[:]
                    # torch.cuda.empty_cache()
            current_idx += 1
            if current_idx > max_idx:
                break

        # 剩余帧处理
        if frames:
            batch_tensors = torch.stack(frames).to(device)
            outputs = model(pixel_values=batch_tensors)
            batch_features = outputs.pooler_output.cpu().numpy()
            features.extend(batch_features)
            del batch_tensors
            torch.cuda.empty_cache()

    container.close()
    return features


# def dino_feature_stream_decord(video_path, model, device, indices, batch_size=128):
#     """
#     使用 Decord 高效提帧 + DINOv2 提特征
#     自动选择 GPU 解码（NVDEC）或 CPU 解码
#     """


#     # ✅ 自动检测 GPU 是否可用
#     # if torch.cuda.is_available() and decord.gpu_enabled():
#     #     ctx = gpu(0)
#     #     print("✅ Using Decord GPU decoding (NVDEC)")
#     # else:
#     ctx = cpu(0)


#     # 读取视频
#     vr = VideoReader(video_path, ctx=ctx)
#     total_frames = len(vr)
#     if total_frames == 0:
#         print(f"⚠️ Empty or unreadable video: {video_path}")
#         return []

#     # 过滤有效索引
#     indices = np.array([i for i in indices if 0 <= i < total_frames])
#     if len(indices) == 0:
#         print(f"⚠️ No valid frame indices for {video_path}")
#         return []

#     features = []

#     # ⚡ 按 batch 提取帧并处理（高效）
#     with torch.no_grad():
#         for i in range(0, len(indices), batch_size):
#             batch_idx = indices[i:i + batch_size]

#             # Decord 批量取帧 (GPU/CPU 自动)
#             batch_frames = vr.get_batch(batch_idx).asnumpy()  # (B, H, W, 3)
#             # 转 PIL + transform
#             imgs = [fast_transform(Image.fromarray(img)) for img in batch_frames]
#             batch_tensors = torch.stack(imgs).to(device, non_blocking=True)

#             # DINO 提特征
#             outputs = model(pixel_values=batch_tensors)
#             batch_features = outputs.pooler_output.cpu().numpy()
#             features.extend(batch_features)

#             del batch_tensors, imgs, batch_frames
#             torch.cuda.empty_cache()

#             print(f"Processed {i+len(batch_idx)}/{len(indices)} frames")

#     return features

def dino_feature_optimized(video_path, data_type=None, ts=None,batch_size=256, target_frames=5400):
    if data_type=='frame':
        print('frame')
        frames = read_jpg_frame(video_path, ts)
    else:
        # print(video_path)
        # exit(0)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total frames",total_frames)
        cap.release()

        if total_frames <= 0:
            return []

        indices = get_frame_indices(total_frames, target_frames)
        print(len(indices))
        print(video_path)
        # frames = extract_selected_frames(video_path, indices)
    print(indices)
    # features = []
    features = dino_feature_stream(video_path, model, device, indices, batch_size=512)

    # for i in range(0, len(frames), batch_size):
    #     batch = frames[i:i + batch_size]
    #     # inputs = hf_processor(images=batch, return_tensors="pt").to(device)
    #     inputs = torch.stack([fast_transform(img) for img in batch])
    #     batch_tensors = inputs.to(device, non_blocking=True)  
    #     print('process done')
    #     with torch.no_grad():
    #         outputs = model(pixel_values=batch_tensors)   # <-- 注意：用 keyword 参数
    #         batch_features = outputs.pooler_output

    #     # with torch.no_grad():
    #     #     outputs = model(**inputs)
    #     #     batch_features = outputs.pooler_output
    #     features.extend(batch_features.cpu().numpy())

    return features


def dinov2(json_path, video_path, save_tensor_path, dataset):
    
    if not os.path.exists(save_tensor_path):
        os.makedirs(save_tensor_path)
    # video_total = set()
    video_total = []
    procesed_videos = set()
    with open(json_path ,'r')as f:
        data = json.load(f)
    for i in data:
        video_total.append(i['question_id'])
    # data = data[250:]
    for video in tqdm(data, total=len(video_total)):
        if dataset=='MLVU_Test':
            video_name = video['video']
        else:
            video_name = video['video_name']
        # if video_name=='test_TFS-5.mp4' or video_name=='test_BWB-5.mp4'or video_name=='test_movie101_91.mp4' or video_name=='test_AWD-3.mp4':
        #     continue
        # full_video_path = video['video_path']
        # if dataset=='MLVU_Test':
        #     folders = [name for name in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, name))]
        #     # print(folders)
        #     for folder in folders:
        #         if video['question_type'] in folder:
        #             full_video_path = os.path.join(video_path, folder,video_name[5:])
        #             # print(full_video_path)
        #     # exit(0)
        # else:
        #     full_video_path = os.path.join(video_path, video_name)
        full_video_path = os.path.join(video_path, video_name)
        if not os.path.exists(full_video_path):
            print('path not exist',full_video_path)
        video_name = video_name.replace('.mp4','')
        output_path = os.path.join(save_tensor_path, f'{video_name}.pkl')
        # print(save_tensor_path)
        # print(output_path)
        # print(output_path)
        if os.path.exists(output_path):
            continue
        if video_name in procesed_videos:
            continue
        procesed_videos.add(video_name)
        if os.path.exists(output_path):
            continue
        # data_type = video['data_type']
        # print('type',video['data_type'])
        # if video['ts']:
        #     ts = [video['start'], video['end']]
        # else:
        #     ts = None
        # frame_features = dino_feature_optimized(full_video_path, data_type, ts,batch_size=2000)
        frame_features = dino_feature_optimized(full_video_path,batch_size=1000)
        if frame_features:
            temp = {video_name: frame_features}
            with open(output_path, 'wb') as f:
                pickle.dump(temp, f)

json_path_videomme = 'ktv/playground/gt_qa_files/Videomme/val_qa.json'
video_path_videomme = 'datasets/Video-MME/data'
save_tensor_path_videomme = 'ktv/save_tensor/Videomme'
dinov2(json_path_videomme, video_path_videomme, save_tensor_path_videomme, 'Videomme')


