import torch
import clip
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import av
from tqdm import tqdm
import csv
import os
import json
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
import os
import numpy as np
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-L/14", device=device)
processed_video = dict()

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
def get_index( bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        frame_indices = np.arange(start_idx, end_idx + 1)
        total_frames = len(frame_indices)
        selected_indices_float = np.linspace(0, total_frames - 1, 50)

        selected_indices_int = np.round(selected_indices_float).astype(int)


        selected_frames = frame_indices[selected_indices_int]
        print(selected_frames)
        
        return selected_frames

def read_jpg_frame( video_path, keyframe,bound=None, fps=3):
        print(video_path)
        frame_indices = keyframe
        max_frame = len(os.listdir(video_path))
        frames = list()
        
        original_sizes = []
        for frame_index in frame_indices:
            frame_index+=1
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            print(f"{frame_index:05d}.jpg")
            frames.append(img)
            original_sizes.append(img.size)
        
        return frames,tuple(original_sizes)

def load_video(video_path, keyframe=None, num_clips=1, num_frms=6, start=None, end=None):
    """
    Load video frames from a video file.

    Parameters:
    - video_path (str): Path to the video file.
    - keyframe (list): List of keyframe tuples like [(10,), (20,), ...] (optional).
    - num_clips (int): Number of clips to extract. Only 1 is supported.
    - num_frms (int): Number of frames to extract.
    - start (float or None): Start time in seconds. None means from beginning.
    - end (float or None): End time in seconds. None means till end.

    Returns:
    - clip_imgs (list[PIL.Image.Image]): List of extracted frames.
    - original_sizes (tuple): Tuple of frame sizes.
    """
    if os.path.isdir(video_path):
        ts = [start, end]
        frames,original_sizes = read_jpg_frame(video_path, keyframe,ts)
        return frames,original_sizes
        # print(frame_idx)
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('total_num_frames',total_num_frames)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_num_frames / fps

        # Convert start/end time (in seconds) to frame indices
        clip_start = 0 if start is None else int(start * fps)
        clip_end = total_num_frames if end is None else int(end * fps)

        clip_start = max(0, min(clip_start, total_num_frames - 1))
        clip_end = max(clip_start + 1, min(clip_end, total_num_frames))
        # print('clip_start',clip_start,clip_end)
        if clip_end <= clip_start:
            raise ValueError(f"Invalid start/end seconds: start={start}s, end={end}s")

        # Compute frame indices
        print('keyframe',keyframe)
        if keyframe:
            # frame_idx = sorted([k[0] for k in keyframe if clip_start <= k[0] < clip_end])
            # frame_idx = sorted([k for k in keyframe if clip_start <= k < clip_end])
            frame_idx = sorted([k for k in keyframe])
        else:
            n = clip_end - clip_start
            m = min(num_frms, n)
            interval = n / m
            frame_idx = [int(clip_start + i * interval) for i in range(m)]
    print('frame_idx',frame_idx)
    clip_imgs = []
    original_sizes = []

    for idx in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: failed to read frame {idx} from {video_path}")
            continue
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            clip_imgs.append(img)
            original_sizes.append(img.size)
        except Exception as e:
            print('video_path', video_path, '\n', 'idx', idx, '\n', e)

    cap.release()
    original_sizes = tuple(original_sizes)
    print('len',len(clip_imgs))
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







def extract_selected_frames(video_path, indices):
    extracted_frames_pil = []
    indices_set = set(indices)
    max_needed_index = -1
    if indices_set: 
        max_needed_index = max(indices_set)

    try:
        with av.open(video_path) as container:
            try:
                stream = container.streams.video[0]
            except IndexError:
                return []
            if stream.frames == 0:
                frame_idx_counter = 0
                frames_found_count = 0
            for frame in container.decode(stream):
                if frame_idx_counter in indices_set:
                    pil_image = frame.to_image() 
                    extracted_frames_pil.append(pil_image)
                    frames_found_count += 1
            
                if frames_found_count == len(indices_set):
                    break 

                if max_needed_index != -1 and frame_idx_counter >= max_needed_index:
                    break
                
                frame_idx_counter += 1
            
            if frames_found_count < len(indices_set):
                actual_video_frames = frame_idx_counter 



    except FileNotFoundError:
        return []
    except av.AVError as e:

        return []
    except Exception as e: 

        return []
            
    return extracted_frames_pil

def video_frame_clustering(frame_features, num_cluster=5):
    

    total_cluster = []

    kmeans = KMeans(n_clusters=num_cluster, random_state=0, init='k-means++', n_init=10).fit(frame_features)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    distances = np.linalg.norm(frame_features - cluster_centers[:, np.newaxis, :], axis=2)
    closest_frames = np.argmin(distances, axis=1)
    clusters = [[] for _ in range(num_cluster)]
    cluster_center_indices = [[] for _ in range(num_cluster)] 
    for j, label in enumerate(labels):
        clusters[label].append(j)
    for j in range (num_cluster):
        cluster_center_indices[j] = closest_frames[j]
    for j in range(len(cluster_center_indices)):
        cluster_center_indices[j] = int(cluster_center_indices[j])
    return cluster_center_indices


    
def get_original_frame_number(
    total_original_frames: int,
    index_in_extracted_list: int,  
    ts: list = None, 
    fps: int = None,  
    max_frames_to_extract: int = 5400
) -> int:  
    print(total_original_frames)

    num_actually_extracted: int
    if total_original_frames <= max_frames_to_extract:
        num_actually_extracted = total_original_frames
    else:
        num_actually_extracted = max_frames_to_extract

    if index_in_extracted_list >= num_actually_extracted:
        raise ValueError(
            f"index_in_extracted_list ({index_in_extracted_list}) out range"
        )

    original_frame_index_0_based: int

    if total_original_frames <= max_frames_to_extract:

        if ts:
            original_frame_index_0_based = int(ts[0]*fps)+ index_in_extracted_list
        else:
            original_frame_index_0_based =  index_in_extracted_list
    else:

        if num_actually_extracted == 1:

            if index_in_extracted_list == 0:
                original_frame_index_0_based = 0
            else:
 
                raise ValueError("If only one frame is extracted, index_in_extracted_list must be 0.")
        else:

            # original_frame_index_0_based = round(
            #     index_in_extracted_list * (total_original_frames - 1) / (num_actually_extracted - 1)
            # )
            # original_frame_index_0_based = int(original_frame_index_0_based) 
            changed_list = np.linspace(0, total_original_frames - 1, max_frames_to_extract, dtype=int)
            # print(index_in_extracted_list)
            original_frame_index_0_based = changed_list[index_in_extracted_list]
            # print(original_frame_index_0_based)
      
    return original_frame_index_0_based


def cluster(json_path, video_path, video_frame_tensor_path, save_cluster_path,dataset):
    print(1)
    if not os.path.exists(save_cluster_path):
        os.makedirs(save_cluster_path)
    with open(video_frame_tensor_path, 'rb') as f:
        data = pickle.load(f)
    video_frame_tensor = dict()
    for key, value in data.items():
        video_frame_tensor[key] = value
    video_name_total = []
    prompt_total = []
    question_id_total = []
    data_tpye_total = []
    ts_total = []
    start_total = []
    end_total = []
    with open(json_path, 'r')as f:
        data = json.load(f)
    for i in data:
        # video_path_total.append(i['video_path'])
        video_name_total.append(i['video_name'])
        # video_name_total.append(i['video'])
        prompt_total.append(i['question'])
        question_id_total.append(i['question_id'])
        # data_tpye_total.append(i['data_type'])
        # ts_total.append(i['ts'])
        # start_total.append(i['start'])
        # end_total.append(i['end'])
    print(2)
    save_cluster = []
    save_cluster_prompt = []
    save_cluster_6 = []
    video_order = []
    save_list = []

    num_cluster = [12]
    m=0
    # for video_path, prompt, question_id,data_type,ts,start,end in tqdm(zip(video_path_total, prompt_total, question_id_total,data_tpye_total,ts_total,start_total,end_total), total=len(video_path_total)):
    for video_name, prompt, question_id in tqdm(zip(video_name_total, prompt_total, question_id_total), total=len(video_name_total)):
        m+=1
        full_video_path = os.path.join(video_path,video_name)
        output_path = os.path.join(save_cluster_path, f'{question_id}.json')
        if os.path.exists(output_path):
            continue
    # for video in tqdm(video_total, total=len(video_total)):
        cluster_frame_total = []
        data_type = None
        if data_type =='frame':
            fps = 3
            total_frames = len(os.listdir(video_path))
        else:
            cap = cv2.VideoCapture(full_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
        # full_video_path = video_path
        
        # if ts:
        #     ts = [start, end]
        # else:
        #     ts =None
        # cluster_num = 6
        try:
            # tensor = video_frame_tensor[question_id]
            tensor = video_frame_tensor[video_name]
        except:
            print('error', question_id)
            continue
  
        # if ts and type!='frame' and (int(end*fps)-int(start*fps)>6):
        #     tensor = tensor[int(start*fps):int(end*fps)]
        # try:
        print(len(tensor))
        cluster_frame_temp = video_frame_clustering(tensor,num_cluster[0])
        cluster_frame_total.append(cluster_frame_temp)
        print('cluster_frame_temp',cluster_frame_temp)
        cluster_frame_total_original = []

        for i in cluster_frame_total[0]:
            cluster_frame_total_original.append(get_original_frame_number(total_frames,i,fps=fps,max_frames_to_extract=5400))
        set_temp = set(cluster_frame_total_original)
        # print('cluster_frame_total_original',cluster_frame_total_original)
        if len(set_temp)!=12:
            temp = np.linspace(0, float(total_frames), 12, endpoint=False)
            cluster_frame_total_original = []
            for i in temp:
                cluster_frame_total_original.append(int(i))
        print('original_frame',cluster_frame_total_original)
        # exit(0)
        # frames_cluster, _ = load_video(full_video_path,cluster_frame_total_original,start=start,end=end)
        frames_cluster, _ = load_video(full_video_path,cluster_frame_total_original,start=None,end=None)
        # except:
            # # continue
            # # error+=1
            # temp = np.linspace(0, float(total_frames), 6, endpoint=False)
            # cluster_frame_total_original = []
            # for i in temp:
            #     cluster_frame_total_original.append(int(i))
            # frames_cluster, _ = load_video(full_video_path,cluster_frame_total_original,start=None,end=None)
            
        temp = []
        batch = []
        for i in frames_cluster:
            batch.append(preprocess_clip(i))
        prompt_new = prompt.split(' ')
        print('prompt_len',len(prompt_new))
        if len(prompt_new)>40:
            prompt_new = prompt_new[10:50]
            prompt = ""
            for j in prompt_new:
                prompt+=j
                prompt+=' '
            prompt=prompt[:-1]

        

        for i in range(0, len(batch), 12):
            image_input = torch.tensor(np.stack(batch[i:i+12])).to(device)
            with torch.no_grad():
                image_features = model_clip.encode_image(image_input.to(device))
                print('question',prompt)
                text_input = clip.tokenize([prompt]).to(device)
                text_features = model_clip.encode_text(text_input.to(device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = 100.0 * image_features @ text_features.T
       
        print(similarity)
        top_k_values, top_k_indices = torch.topk(similarity.squeeze(), 12)
        print(top_k_indices)
        top_k_cluster_indices = [cluster_frame_total_original[i] for i in top_k_indices]
        key_frame_order = []
        for index, i in enumerate(top_k_cluster_indices):
            key_frame_order.append([i,index])
        temp = dict()
        temp[question_id] = key_frame_order
        print(temp)
        # with open(output_path, 'w') as f:
        #     json.dump(temp, f, ensure_ascii=False, indent=4)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(temp, f, ensure_ascii=False, indent=4, default=lambda o: int(o) if isinstance(o, np.integer) else o)


json_path_videomme_test = 'ktv/playground/gt_qa_files/Videomme/val_qa.json'
save_tensor_path_videomme_test = 'ktv/save_tensor/Videomme.pkl'
video_path_videomme = 'datasets/Video-MME/data'
save_cluster_path_videomme_test = 'videomme_test_json_temp12'
cluster(json_path_videomme_test, video_path_videomme,save_tensor_path_videomme_test, save_cluster_path_videomme_test,'videomme')

