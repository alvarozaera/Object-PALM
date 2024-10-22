import os
import cv2
import torch
import json
import concurrent
import collections
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

# NOTE: Variables to configure the input_dir, output_dir, and model_id
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
# Ego4D variables
split = "train"
nframes = 11
ego4d_base_path = "/cluster/scratch/azaera/ego4d/v2"
output_dir = "grounding_dino_hand_outputs"


processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

with open(f"{ego4d_base_path}/annotations/fho_lta_{split}.json", "r") as f:
    dset = json.load(f)

annotations = collections.defaultdict(list)
for entry in dset["clips"]:
    annotations[entry['clip_uid']].append(entry)

# Sort windows by their PNR frame (windows can overlap, but PNR is distinct)
annotations = {
    clip_uid: sorted(annotations[clip_uid], key=lambda x: x['action_idx'])
    for clip_uid in annotations
}


def generate_per_clip_uid_action_idx(clip_uid_action_idx):
    clip_uid, idx = clip_uid_action_idx

    # if os.path.exists(os.path.join(output_dir, f"{clip_uid}_{idx}.json")):
    #     return
    try:
        with open(os.path.join(output_dir, f"{clip_uid}_{idx}.json"), "r") as f:
            detections_dict = json.load(f)
        return
    except:
        pass    

    action = annotations[clip_uid][idx]
    action_idx = action["action_idx"]
    video_file = f"{ego4d_base_path}/clips/{clip_uid}.mp4"
    cap = cv2.VideoCapture(video_file)

    print(f"Processing {clip_uid}_{action_idx}", flush=True)


    if nframes == 1:
        frame_list = [
            (action["action_clip_start_frame"] + action["action_clip_end_frame"]) // 2,
        ]
    else:
        # frame_list = np.linspace(
        #     int(action["action_clip_start_frame"] * (.5 + offset_portion) + action["action_clip_end_frame"] * (.5 - offset_portion)), 
        #     int(action["action_clip_start_frame"] * (.5 - offset_portion) + action["action_clip_end_frame"] * (.5 + offset_portion)),
        #     nframes,
        # )
        frame_list = np.linspace(
            action["action_clip_start_frame"], 
            action["action_clip_end_frame"],
            nframes,
        ).astype(int)

    detections_dict = {}

    for i, frameidx in enumerate(frame_list):
        try:        
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameidx)
            ret, frame = cap.read()
        

            H, W = frame.shape[:2]
        
            # load image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame)
            
            inputs = processor(images=image_pil, text="hand.", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image_pil.size[::-1]]
            )

            # plt.imshow(image_pil)
            detections_dict[str(frameidx)] = results[0]
            bounding_boxes = results[0]['boxes'].cpu()
            bounding_boxes = expand_boxes(bounding_boxes, 0.15, 0.15, H=H, W=W)
            detections_dict[str(frameidx)]['expanded_boxes'] = bounding_boxes.cpu().numpy().tolist()
            detections_dict[str(frameidx)]['boxes'] = results[0]['boxes'].cpu().numpy().tolist()
            detections_dict[str(frameidx)]['scores'] = results[0]['scores'].cpu().numpy().tolist()

            # plot_boxes(bounding_boxes)    
        except Exception as e:
            print(f"Error in {clip_uid}_{action_idx}:", e, flush=True)

    cap.release()
    try:
        with open(os.path.join(output_dir, f"{clip_uid}_{idx}.json"), "r") as f:
            detections_dict = json.load(f)
        return
    except:
        pass   
    with open(os.path.join(output_dir, f"{clip_uid}_{action_idx}.json"), "w") as f:
        json.dump(detections_dict, f)


def expand_boxes(boxes, margin_x, margin_y, H=1080, W=1920):
    """Expand the bounding box by a certain margin."""
    # Same as above but box is a tensor of shape (N, 4)
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0] - margin_x * H
    boxes[:, 1] = boxes[:, 1] - margin_y * W
    boxes[:, 2] = boxes[:, 2] + margin_x * H
    boxes[:, 3] = boxes[:, 3] + margin_y * W
    # Ensure that the expanded box is within the image boundaries
    boxes[:, 0] = torch.clamp(boxes[:, 0], 0, W)
    boxes[:, 1] = torch.clamp(boxes[:, 1], 0, H)
    boxes[:, 2] = torch.clamp(boxes[:, 2], 0, W)
    boxes[:, 3] = torch.clamp(boxes[:, 3], 0, H)
    return boxes


def plot_boxes(bounding_boxes):
    """Plot bounding boxes """

    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = box
        plt.plot([x1, x2], [y1, y1], color='red')  # Top edge
        plt.plot([x1, x2], [y2, y2], color='red')  # Bottom edge
        plt.plot([x1, x1], [y1, y2], color='red')  # Left edge
        plt.plot([x2, x2], [y1, y2], color='red')  # Right edge

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Bounding Boxes')
    #plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    #plt.grid(True)
    # Not show axis
    plt.axis('off')
    plt.show()


torun = []
for clip_uid in annotations:
    for idx in range(len(annotations[clip_uid])):
        # try to open and load the json file (add to torun if it doesn't exist or is corrupted)
        try:
            with open(os.path.join(output_dir, f"{clip_uid}_{idx}.json"), "r") as f:
                detections_dict = json.load(f)
        except:
            torun.append((clip_uid, idx))
            continue
        # if not os.path.exists(os.path.join(output_dir, f"{clip_uid}_{idx}.json")):
        #     torun.append((clip_uid, idx))

# for clip_uid_action_idx in tqdm(torun):
#     generate_per_clip_uid_action_idx(clip_uid_action_idx)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(tqdm(pool.map(generate_per_clip_uid_action_idx, torun), total=len(torun)))