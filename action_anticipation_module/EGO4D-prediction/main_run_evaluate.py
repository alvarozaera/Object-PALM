# %%
import os
import random
import collections
import json 
import numpy as np
import torch
import argparse
import concurrent
import time 
from tqdm import tqdm
from collections import defaultdict
from transformers import pipeline, set_seed
from transformers import AutoTokenizer

from utils import annotation_utils, eval_utils, prompt_utils
#os.environ["LD_LIBRARY_PATH"] = "/local/home/sankim/miniconda3/envs/vclip/lib/x86_64-linux-gnu/libstdc++.so.6"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(16)

parser = argparse.ArgumentParser()
parser.add_argument('--split')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--ntot_example', type=int, default=8)
parser.add_argument('--nexample', type=int, default=1)
parser.add_argument('--numbercaptions', type=int, default=1)
parser.add_argument('--version', default="v2")
parser.add_argument('--use_narration', default="imagecaption")
parser.add_argument('--caption_file', default=None)
parser.add_argument('--noun_list', default=None)
parser.add_argument('--is_past_action', default="True")
parser.add_argument('--is_pred_action', default="True")
parser.add_argument('--pred_action_version', default="79")
parser.add_argument('--text_generation_model', default="EleutherAI/gpt-neo-1.3B")
parser.add_argument('--prompt_design', default="maxmargin")
parser.add_argument('--fetch_k', type=int, default=20)
parser.add_argument('--lambda_mult', type=float, default=0.5)
parser.add_argument('--embedding_model', default="mpnet")
parser.add_argument('--log_file', default="None")
parser.add_argument('--max_nprev', type=int, default=8)
parser.add_argument('--padding', default="last")
parser.add_argument('--remove_duplicate', default="True")
parser.add_argument('--action_feature', default="egovlp")
parser.add_argument('--wo_examples', default="False")
args = parser.parse_args()

# %%
device = args.device
nprev = 8
ntot_example = args.ntot_example
nexample = args.nexample
ncaption = 1
noun_list = args.noun_list if args.noun_list != "None" else None 
is_scenario = False
use_narration = args.use_narration
numbercaptions = args.numbercaptions
is_patch = True 
is_past_action = args.is_past_action == "True"
is_pred_action = args.is_pred_action
version = args.version
split = args.split
text_generation_model = args.text_generation_model
prompt_design = args.prompt_design
annotation_folder = "/cluster/scratch/azaera/ego4d/v2/annotations"
use_narration_summary = True


logfile = "{}.txt".format(
    str(time.time()) if args.log_file == "None" else args.log_file
)


#Evaluation part

import json 
import collections
with open(f"{annotation_folder}/fho_lta_val.json", "r") as f:
    dset = json.load(f)

annotations = collections.defaultdict(list)
for entry in dset["clips"]:
    annotations[entry['clip_uid']].append(entry)

annotations = {
    clip_uid: sorted(annotations[clip_uid], key=lambda x: x['action_idx'])
    for clip_uid in annotations
}


pred_all = {}
vres = []
nres = []
ares = []
import itertools
from utils import eval_utils
import numpy as np

action_ids_to_consider = json.load(open("./llm-log/llama8B_intention_final_report_common.json", "r"))

with open("./llm-log/" + logfile, "r") as f:
    for lines in itertools.zip_longest(*[f] * 10):
        clip_uid, idx = lines[0].split(" ")[0].split("_")
        if f"{clip_uid}_{idx}" not in action_ids_to_consider:
            continue
        idx = int(idx)
        vpd = [[int(x) for x in l.split(" ")[2:]] for l in lines[:5]]
        npd = [[int(x) for x in l.split(" ")[2:]] for l in lines[5:]]

        ts = annotations[clip_uid][(idx + 1):(idx + 21)]
        vts = [x["verb_label"] for x in ts]
        vres.append(eval_utils.AUED(np.array(vpd).transpose((1, 0))[None, ...], np.array(vts)[None, ...]))
        nts = [x["noun_label"] for x in ts]
        nres.append(eval_utils.AUED(np.array(npd).transpose((1, 0))[None, ...], np.array(nts)[None, ...]))

        ares.append(eval_utils.AUED(
            np.array(npd).transpose((1, 0))[None, ...] * 130 + np.array(vpd).transpose((1, 0))[None, ...],
            np.array(nts)[None, ...] * 130 + np.array(vts)[None, ...],
        ))

verb_list = [x["ED_19"] for x in vres]
noun_list = [x["ED_19"] for x in nres]
action_list = [x["ED_19"] for x in ares]

print("{:.4f} & {:.4f} & {:.4f}".format(np.mean(verb_list), np.mean(noun_list), np.mean(action_list)))
print(len(ares))
