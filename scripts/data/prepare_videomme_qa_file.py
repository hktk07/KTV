#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import argparse
import csv
import json
import re
def main(args, task_name="Videomme"):
    data_list_info = []
    with open(args.qa_file, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        # ,video_name,question,question_id,answer,video_id,question_type,a0,a1,a2,a3,a4
        for idx, row in enumerate(spamreader):
            if idx == 0:
                continue
            # _, video_name, question, question_id, answer, video_id, _, a0, a1, a2, a3, a4 = row
            _, deration, domain, sub_category, url, video_name, question_id, task_type, question, options, answer= row
            candidates = options
            cleaned = options.strip("[]'")
            cleaned = cleaned.replace('\n','')
            cleaned = cleaned.replace('\'','')
            # print(cleaned)
            # exit(0)
            # print(cleaned)
            # 用正则按选项切分（匹配大写字母+句点开头的部分）
            items = re.findall(r"[A-Z]\.\s.*?(?=(?:[A-Z]\.|$))", cleaned)
            # 去掉多余空格
            candidates = [x.strip() for x in items]
            # print(candidates)
            # print(answer)
            # exit(0)
            if answer =='A':
                answer = candidates[0]
            elif answer =='B':
                answer = candidates[1]
            elif answer == 'C':
                answer = candidates[2]
            elif answer == 'D':
                answer = candidates[3]
            assert answer in candidates
        
            data_list_info.append({
                "task_name": task_name,
                "video_name": f"{video_name}.mp4",
                "question_id": question_id,
                "question": question,
                "answer_number": candidates.index(answer),
                "candidates": candidates,
                "answer": answer,
            })

    folder = f"/mnt/data/sby/ktv/playground/gt_qa_files/Videomme"
    os.makedirs(folder, exist_ok=True)
    with open(f"/mnt/data/sby/ktv/playground/gt_qa_files/Videomme/val_qa.json", "w") as f:
        json.dump(data_list_info, f, indent=4)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", help="Path to Videomme.csv", default='/mnt/data/sby/ktv/playground/gt_qa_files/Videomme/val_qa.csv')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
