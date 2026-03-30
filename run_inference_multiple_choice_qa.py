#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import os
import sys
from pathlib import Path
# sys.path.insert(0, Path(__file__).parent.as_posix())
sys.path.insert(0, os.path.join(Path(__file__).parent.as_posix(), "ktv"))
import json
from tqdm import tqdm
import torch       #for cuda device
# import torch_npu  for npu device
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from dataset import load_video
from prompt import get_multiple_choice_prompt
from utils import get_chunk


def llava_inference(
    video_frames,
    question,
    candidates,
    conv_mode,
    model,
    tokenizer,
    image_processor,
    image_sizes,
    temperature,
    top_p,
    num_beams,
    temporal_aggregation,
    keyframe_order=None,
    num_frames=None,
    prune_mode = None,
    global_rate = None,
    tokens_num = None,
):
    # Get multiple choice prompt
    prompt = get_multiple_choice_prompt(model, conv_mode, question, candidates)
    # print(prompt)
    # Get text inputs
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).cuda()

    # Get image inputs
    image_tensor = process_images(video_frames, image_processor, model.config)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=128,
            use_cache=True,
            temporal_aggregation=temporal_aggregation,
            keyframe_order = keyframe_order,
            num_frames=num_frames,
            prune_mode = prune_mode,
            global_rate = global_rate,
            tokens_num = tokens_num
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def run_inference(args):
    """
    Run inference on Video QA Dataset.

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()

    # Load tokenizer, model and image processor
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        device = torch.device("cuda"),
        device_map="auto",
        rope_scaling_factor=args.rope_scaling_factor,
    )
    key_frame_path = args.key_frame_path
    key_frame = {}
    if key_frame_path:
        with open(key_frame_path, 'r') as f:
            key_frame = json.load(f)
        for key, value in key_frame.items():
            key_frame[key] = value
    else :
        key_frame = None
    # Override image aspect ratio if needed
    if args.image_aspect_ratio:
        model.config.image_aspect_ratio = args.image_aspect_ratio

    # Load questions and answers
    gt_qa_pairs = json.load(open(args.gt_file, "r"))
    gt_qa_pairs = get_chunk(gt_qa_pairs, args.num_chunks, args.chunk_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    # ans_file = open(
    #     os.path.join(args.output_dir, f"{args.output_name}.json"), "w")
    output_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    if os.path.exists(output_path):
        ans_file = open(
            os.path.join(args.output_dir, f"{args.output_name}.json"), "a")
        generated_id = []
        with open(output_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                generated_id.append(data['id'])
    else:
        ans_file = open(
            os.path.join(args.output_dir, f"{args.output_name}.json"), "w")
        generated_id = []
    # Iterate over each sample in the ground truth file
    for index, sample in enumerate(tqdm(gt_qa_pairs)):
        # print(sample)
        task_name = sample["task_name"]
        video_name = sample["video_name"]
        question_id = sample["question_id"]
        if question_id in generated_id:
            continue
        question = sample["question"]
        answer_number = sample["answer_number"]
        candidates = sample["candidates"]
        answer = sample["answer"]
        # task_type = sample['task_type']
        # video_path = sample["video_path"]
        # print('video_path', video_path)
        # if question_id =='question_548':
        #     continue
        sample_set = {
            "task_name": task_name,
            "question": question,
            "id": question_id,
            "answer_number": answer_number,
            "candidates": candidates,
            "answer": answer,
            # "task_type": task_type,
            # "video_path": video_path,
        }

        # Load video
        # if task_name=='STAR':
        #     start = video_name.find('_')
        #     end = video_name.find('.mp4')
        #     start_end = video_name[start+1:end]
        #     time = start_end.split('_')
        #     time1 = float(time[0])
        #     time2 = float(time[1])
        #     video_name = video_name[:start]+'.mp4'
        #     video_path = os.path.join(args.video_dir, video_name)
        # else:
        video_path = os.path.join(args.video_dir, video_name)
        video_path = video_path
        try:
            if key_frame[question_id]:
                keyframe = key_frame[question_id]

                keyframe_order = []
                temp1 = [i[0] for i in keyframe]
                temp_dict = dict()
                for i in keyframe:
                    temp_dict[i[0]] = i[1]
                temp1.sort()
                for i in temp1:
                    keyframe_order.append(temp_dict[i])
            else:
                keyframe = None
                keyframe_order=None
        except:
            keyframe = None
            keyframe_order=None
        if os.path.exists(video_path):
            # try:
            video_frames, sizes = load_video(video_path, keyframe, num_frms=args.num_frames)
            # except Exception as e:
            #     print(f"Failed to load {video_path}, continue...")
            #     continue

            # Run inference on the video
            output = llava_inference(
                video_frames,
                question,
                candidates,
                args.conv_mode,
                model,
                tokenizer,
                image_processor,
                sizes,
                args.temperature,
                args.top_p,
                args.num_beams,
                args.temporal_aggregation,
                keyframe_order,
                args.num_frames,
                args.prune_mode,
                global_rate = args.rate,
                tokens_num = args.tokens_num
            )
            output = output.replace("In the image", "In the video")
            print(output)
            sample_set["pred"] = output
            ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question and answer.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--input_structure", type=str, default="image_seq")
    parser.add_argument("--image_aspect_ratio", type=str, default=None)
    parser.add_argument("--temporal_aggregation", type=str, default=None)
    parser.add_argument("--rope_scaling_factor", type=int, default=1)
    parser.add_argument("--key_frame_path", type=str, default=None)
    parser.add_argument("--prune_mode", type=str, default=None)
    parser.add_argument("--rate", help='this_global_rate', type=float,default=None)
    parser.add_argument("--tokens_num", help='tokens_num', type=int,default=936)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
