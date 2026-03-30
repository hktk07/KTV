import argparse
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pred_path", default="/mnt/data/sby/qwen2.5-vl/Qwen2.5-VL-main/qwen2.5-vl-7b_4090/mlvu_test_qwen7b_rate_0.3_1000tokens.json", help="The path to file containing prediction.")
    # parser.add_argument("--old_path", default="/mnt/data/sby/qwen2.5-vl/Qwen2.5-VL-main/qwen2.5-vl-7b_4090/mlvu_test_6keyframe_qwen2.5-vl-7b_4090_rate_0.2_4090.json", help="The path to file containing ground truth.")
    parser.add_argument("--pred_path", default="/mnt/data/sby/DYTO-main/llava_34b_4090/mlvu_test_6keyframe_dyto_llava34b_4090_rate_1_4090.json", help="The path to file containing prediction.")
    parser.add_argument("--old_path", default="/mnt/data/sby/qwen2.5-vl/Qwen2.5-VL-main/qwen2.5-vl-7b_4090/videomme_6keyframe_qwen2.5-vl-7b_4090_rate_0.2_4090.json", help="The path to file containing ground truth.")
    # parser.add_argument("--pred_path", default="/mnt/data/sby/igvlm/qwen2.5-vl-7b_4090/ig_egoschema.json", help="The path to file containing prediction.")
    # parser.add_argument("--old_path", default="/mnt/data/sby/qwen2.5-vl/Qwen2.5-VL-main/qwen2.5-vl-7b_4090/egoschema_qwen7b_rate_0.3_1000tokens.json", help="The path to file containing ground truth.")
    # parser.add_argument("--pred_path", default="/mnt/data/sby/DYTO-main/llava_34b_4090/nextqa_6keyframe_dyto_llava34b_4090_rate_1_4090.json", help="The path to file containing prediction.")
    # parser.add_argument("--old_path", default="/mnt/data/sby/ktv/llava34b_4090/nextqa_6keyframe_llava34b_rate_1_1872_4090.json", help="The path to file containing ground truth.")
    # parser.add_argument("--pred_path", default="/mnt/data/sby/qwen2.5-vl/Qwen2.5-VL-main/qwen2.5-vl-7b_4090/intentqa_qwen7b_rate_0.6_1000tokens.json", help="The path to file containing prediction.")
    # parser.add_argument("--old_path", default="/mnt/data/sby/qwen2.5-vl/Qwen2.5-VL-main/qwen2.5-vl-7b_4090/intentqa_6keyframe_qwen2.5-vl-7b_4090_rate_0.3_4090.json", help="The path to file containing ground truth.")
    
    # parser.add_argument("--pred_path", default="/mnt/data/sby/qwen2.5-vl/Qwen2.5-VL-main/qwen2.5-vl-7b_4090/videomme_6keyframe_qwen2.5-vl-7b_4090_rate_0.2_4090.json", help="The path to file containing prediction.")
    # parser.add_argument("--old_path", default="/mnt/data/sby/qwen2.5-vl/Qwen2.5-VL-main/qwen2.5-vl-7b_4090/intentqa_6keyframe_qwen2.5-vl-7b_4090_rate_0.3_4090.json", help="The path to file containing ground truth.")
    args = parser.parse_args()
    return args
# /mnt/data/sby/DYTO-main/llava_7b_4090/nextqa_6keyframe_dyto_llava7b_4090_rate_1_4090.json

def map_prediction_to_option(pred):
    pred_option = "none"
    if isinstance(pred, str):
        prediction_letter = pred[0]
        if prediction_letter in "abcdefABCDEF":
            pred_option = prediction_letter.lower()
        if "answer is " in pred:
            pred = pred[pred.index("answer is"):]
        if "A:" in pred or "A)" in pred:
            pred_option = "a"
        elif "B:" in pred or "B)" in pred:
            pred_option = "b"
        elif "C:" in pred or "C)" in pred:
            pred_option = "c"
        elif "D:" in pred or "D)" in pred:
            pred_option = "d"
        elif "E:" in pred or "E)" in pred:
            pred_option = "e"
        elif "F:" in pred or "F)" in pred:
            pred_option = "f"
    return pred_option


def check_ans(pred, gt):
    flag = False

    pred_option = map_prediction_to_option(pred)
    # print(pred_option)
    if pred_option not in "abcdef":
        print(f"Model does not follow the instruction: {pred}")
    elif pred_option == gt.lower():
        flag = True

    return flag


def main():
    # Parse arguments.
    args = parse_args()

    file = open(args.pred_path)
    old_file = open(args.old_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]
    old_pred_contents = [eval(i.strip()) for i in old_file.readlines()]
    old_pred_contents = old_pred_contents[:len(new_pred_contents)]
    task_accuracy = {}
    for new_pred_content in tqdm(new_pred_contents):
        task_name = new_pred_content["task_name"]
        if task_name not in task_accuracy:
            task_accuracy[task_name] = {
                "yes_count": 0,
                "no_count": 0,
            }

        gt = chr(ord("a") + new_pred_content["answer_number"])
        if check_ans(
            pred=new_pred_content["pred"],
            gt=gt,
        ):
            task_accuracy[task_name]["yes_count"] += 1
        else:
            task_accuracy[task_name]["no_count"] += 1

    accuracy_list = []
    for task_name in task_accuracy:
        yes_count = task_accuracy[task_name]["yes_count"]
        no_count = task_accuracy[task_name]["no_count"]
        accuracy = yes_count / (yes_count + no_count)
        accuracy_list.append(accuracy)
        print(task_name)
        print("\tYes count:", yes_count)
        print("\tNo count:", no_count)
        print("\tAccuracy:", accuracy)
    if len(accuracy_list) > 1:
        print("Average accuracy:", np.mean(accuracy_list))

    task_accuracy = {}
    for old_pred_content in tqdm(old_pred_contents):
        task_name = old_pred_content["task_name"]
        if task_name not in task_accuracy:
            task_accuracy[task_name] = {
                "yes_count": 0,
                "no_count": 0,
            }

        gt = chr(ord("a") + old_pred_content["answer_number"])
        if check_ans(
            pred=old_pred_content["pred"],
            gt=gt,
        ):
            task_accuracy[task_name]["yes_count"] += 1
        else:
            task_accuracy[task_name]["no_count"] += 1

    accuracy_list = []
    for task_name in task_accuracy:
        yes_count = task_accuracy[task_name]["yes_count"]
        no_count = task_accuracy[task_name]["no_count"]
        accuracy = yes_count / (yes_count + no_count)
        accuracy_list.append(accuracy)
        print(task_name)
        print("\tYes count:", yes_count)
        print("\tNo count:", no_count)
        print("\tAccuracy:", accuracy)
    if len(accuracy_list) > 1:
        print("Average accuracy:", np.mean(accuracy_list))


if __name__ == "__main__":
    main()
