# %%
import os
os.environ["LD_LIBRARY_PATH"] = "/miniconda3/envs/vclip/lib/x86_64-linux-gnu/libstdc++.so.6"
import collections
import json 
import numpy as np
import openai 
import concurrent
from tqdm import tqdm
import cv2
import requests


# %%
split = "val"
version = "v1"
question = "intention"
model_to_use = "blip2"
ncaption = 4
nframes = 1
offset_portion = 0.1

logfile = f"./captions/caption_{model_to_use}_{question}_ncap{ncaption}_{split}_{version}.txt"
remote = "ait-server-04.inf.ethz.ch:8964" # or "ip:port"

# %%
question_dict = {
    "detection" : "what objects are in this image",
    "action" : "what is the person doing in this image", 
    "location" : "where is the person",
    "interaction" : "what is the person interacting with in this image",
    "intention" : "what does the person want to do in this image",
    "prediction" : "what will the person do later in this image",
    "caption" : "A person is",
    "caption_iblip" : "A short image caption:",
    "caption_iblip_prefix" : "A person is",
    "img2llm_caption_action": "this is a description of the image: {}. In this image, a person is",
    "img2llm_caption_interaction": "this is a description of the image: {}. In this image, a person is",
    "img2llm_caption_intention": "this is a description of the image: {}. In this image, a person is",
    "img2llm_caption_location": "this is a description of the image: {}. In this image, a person is",
}

# %%
with open(f"/directory/to/annotations/fho_lta_{split}.json", "r") as f:
    dset = json.load(f)

annotations = collections.defaultdict(list)
for entry in dset["clips"]:
    annotations[entry['clip_uid']].append(entry)

# Sort windows by their PNR frame (windows can overlap, but PNR is distinct)
annotations = {
    clip_uid: sorted(annotations[clip_uid], key=lambda x: x['action_idx'])
    for clip_uid in annotations
}


# %%

http_path = remote

def generator_api(pack):
    if model_to_use == "instructblip":
        cond = "_instructblip"
    elif model_to_use == "git":
        cond = "_git"
    elif "video" in model_to_use:
        cond = "_video"
    else:
        cond = ""

    images, text, max_new_tokens, ncaption, top_k, top_p = pack
    decoded = requests.post("http://{}/caption{}/".format(http_path, cond), json={
        "image" : images,
        "prefix" : text,
        "max_new_tokens" : max_new_tokens,
        "ncaption" : ncaption,
        "top_k" : top_k,
        "top_p" : top_p,
    })
    #print(decoded.text)
    return json.loads(decoded.text)

generator = generator_api

# %%

keys_done = set()
if os.path.exists(logfile):
    with open(logfile, "r") as f:
        for l in f:
            keys_done.add(l.split(" ")[0])

torun = []
for clip_uid in annotations:
    for idx in range(len(annotations[clip_uid])):
        if clip_uid + "_" + str(annotations[clip_uid][idx]["action_idx"]) in keys_done: 
            continue 

        if clip_uid == "440656ae-cb82-464e-b320-25c8e693ad84":
            continue
        torun.append((clip_uid, idx))

def generate_per_clip_uid_action_idx(clip_uid_action_idx):
    clip_uid, idx = clip_uid_action_idx
    action = annotations[clip_uid][idx]
    video_file = "/directory/to/clip.mp4"
    cap = cv2.VideoCapture(video_file)

    if nframes == 1:
        frame_list = [
            (action["action_clip_start_frame"] + action["action_clip_end_frame"]) // 2,
        ]
    else:
        frame_list = np.linspace(
            int(action["action_clip_start_frame"] * (.5 + offset_portion) + action["action_clip_end_frame"] * (.5 - offset_portion)), 
            int(action["action_clip_start_frame"] * (.5 - offset_portion) + action["action_clip_end_frame"] * (.5 + offset_portion)),
            nframes,
        )
    for frameidx in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameidx)
        ret, frame = cap.read()
        try:
            h, w = frame.shape[:2]
        except:
            continue 

        if "caption" not in question:
            question_text = "Question: {}? Answer:".format(question_dict[question])
        else:
            question_text = question_dict[question]

        image_serilize = json.dumps(frame[..., ::-1].tolist())
        generated = generator_api((image_serilize, question_text, 70, ncaption, 5, 0.75))
        with open(logfile, "a") as f:
            f.write("{} {}\n".format(clip_uid + "_" + str(action["action_idx"]), generated))


def video_generate_per_clip_uid_action_idx(clip_uid_action_idx):
    from pytorchvideo.data.video import VideoPathHandler
    clip_uid, idx = clip_uid_action_idx
    action = annotations[clip_uid][idx]
    video_file = "/directory/to/video/clip.mp4"

    video_path_handler = VideoPathHandler()
    video = video_path_handler.video_from_path(video_file)

    clip = video.get_clip(action["action_clip_start_sec"], action["action_clip_end_sec"])

    if "caption" not in question:
        question_text = "Question: {}? Answer:".format(question_dict[question])
    else:
        question_text = question_dict[question]
    
    frames = clip["video"][:, ::30, ...]
    image_serilize = json.dumps(frames.tolist())
    generated = generator_api((image_serilize, question_text, 70, ncaption, 5, 0.75))
    with open(logfile, "a") as f:
        f.write("{} {}\n".format(clip_uid + "_" + str(action["action_idx"]), generated))


# %%
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    if "video" in model_to_use :
        results = list(tqdm(pool.map(video_generate_per_clip_uid_action_idx, torun), total=len(torun)))
    else:
        results = list(tqdm(pool.map(generate_per_clip_uid_action_idx, torun), total=len(torun)))

