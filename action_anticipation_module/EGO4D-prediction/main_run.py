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
annotation_folder = "/cluster/work/cvl/xiwang1/sankim/data/annotations"

logfile = "{}.txt".format(
    str(time.time()) if args.log_file == "None" else args.log_file
)

# top_k word p probablity threshold
top_k = 50
top_p = 0.5
print("hyperparams: top_k {} top_p {}".format(top_k, top_p))

is_vis = False 
if is_vis:
    print("visulization, saving everything")

# %%
#
if 'llama' in text_generation_model:
    generator, tokenizer = prompt_utils.get_generator(text_generation_model, device=device)
else: 
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    generator = prompt_utils.get_generator(text_generation_model, device=device)

# %%
with open("{}/fho_lta_taxonomy.json".format(annotation_folder), "r") as f:
    tax = json.load(f)
vmap, nmap = annotation_utils_new.get_vnmap(tax)

with open(annotation_folder + "/narration.json", "r") as f:
    narrations = json.load(f)

if args.action_feature == "egovlp":
    with open("/cluster/work/cvl/xiwang1/sankim/data/new_action_output_egovlp_best/outputs_{}.json".format(split.split("_")[0] if "val" in split else split), "r") as f:
        pred_action = json.load(f)
else: #egovlpv2, mvit, slowfast, stillfast
     with open(f"/cluster/work/cvl/xiwang1/sankim/data/annotations/action_recognition_outputs/outputs_val_{args.action_feature}.json", "r") as f:
        pred_action = json.load(f)
print(f"Using {args.action_feature} feature for action recognition!!!")

all_caption = collections.defaultdict(list)
all_caption = annotation_utils_new.update_captions(
    all_caption, 
    args.caption_file,
)

prompt_search_in_train = False 
if prompt_design in ["maxmargin", "semanticsim"]:
    print("try populate dataset with training set, are you sure?")
    import time 
    #time.sleep(10)
    try:
        all_caption = annotation_utils_new.update_captions(
            all_caption, 
            args.caption_file.replace("test_unannotated", "train").replace("val_small", "train").replace("val", "train"),
        )
        prompt_search_in_train = True 
    except Exception as e:
        print("failed, doing prompt search in validation set, err message", e)
        all_caption = annotation_utils_new_new.update_captions(
            all_caption, 
            "/cluster/work/cvl/xiwang1/sankim/data/llm-log/blip2-caption/caption_blip2_ncap4_train_v2.json",
        )
        prompt_search_in_train = True # False # just to make sure

if "val" not in split:
    try:
        all_caption = annotation_utils_new.update_captions(
            all_caption, 
            args.caption_file.replace("test_unannotated", "val"),
        )
    except:
        print("populating captions with val_small, NOT FULL VALIDATION SET!")
        all_caption = annotation_utils_new.update_captions(
            all_caption, 
            args.caption_file.replace("test_unannotated", "val_small"),
        )

all_question = {
    "action" : collections.defaultdict(list),
    "location" : collections.defaultdict(list),
    "intention" : collections.defaultdict(list),
    "interaction" : collections.defaultdict(list),
    "prediction" : collections.defaultdict(list),
    "ncaption" : numbercaptions,
    "narration" : narrations,
}

for _question in ["action", "location", "intention", "interaction", "prediction"]:
    if "_{}".format(_question) in use_narration:
        all__question_path = "/cluster/work/cvl/xiwang1/sankim/data/llm-log/{}{}_blip2_ncap1_npatch1_nframe8_{}_{}_best_param_midframe.json".format(
            _question,
            "_img2llm" if "intentionimg2llm" in use_narration else "",
            split,
            version,
        )
        print("loading {} from {}".format(_question, all_action_path))
        all_question[_question] = annotation_utils_new.update_captions(all_question[_question], all_action_path, non_exist_ok=True)


clip2video = {}
def get_dset(split):
    with open("{}/fho_lta_{}.json".format(annotation_folder, split), "r") as f:
        dset = json.load(f)

    annotations = collections.defaultdict(list)
    for entry in dset["clips"]:
        annotations[entry['clip_uid']].append(entry)

        clip2video[entry['clip_uid']] = entry['video_uid']

    # Sort windows by their PNR frame (windows can overlap, but PNR is distinct)
    annotations = {
        clip_uid: sorted(annotations[clip_uid], key=lambda x: x['action_idx'])
        for clip_uid in annotations
    }

    return annotations

ann_train = get_dset("train")
ann_val = get_dset("val")
ann_torun = get_dset(split)

def get_video2scenario():

    video2scenario = {}
    with open(f"{annotation_folder}/ego4d.json", "r") as f:
        ego4d = json.load(f)

    for video in ego4d["videos"]:
        video2scenario.update({video["video_uid"] : video["scenarios"]})

    return video2scenario
video2scenario = get_video2scenario()


# %%
if prompt_design == "GT":
    prompt = "You are going to complete an action sequence, an action is one (verb, noun) pair{}. A complete sequence consists of {} actions.{}\n".format(
        ", and the candidates for noun are shown in the Noun list" if noun_list is not None else "",
        (nprev if is_past_action else 0) + 20,
        " You will also be given a text description of the past actions for reference." if use_narration != "None" else "",
    )
else:
    prompt = prompt_utils.get_prompt(
        prompt_design=prompt_design,
        ann_for_prompt=ann_val if prompt_search_in_train == False else ann_train,
        nprev=nprev, nexample=ntot_example, ncaption=ncaption, fetch_k=args.fetch_k, lambda_mult=args.lambda_mult, embedding_model=args.embedding_model, 
        all_caption=all_caption, all_question=all_question, 
        clip2video=clip2video, video2scenario=video2scenario, pred_action=pred_action, tax=tax,
        noun_list=noun_list, is_scenario=is_scenario, is_past_action=is_past_action,
        use_narration=use_narration,
        device=device,
    )

# %%
keys_done = set()
if os.path.exists("./llm-log/" + logfile):
    with open("./llm-log/" + logfile, "r") as f:
        for l in f:
            keys_done.add(l.split(" ")[0])

torun = []
for clip_uid in ann_torun:
    l = len(ann_torun[clip_uid])
    for i in range(l - 20 - nprev):
        if clip_uid + "_" + str(ann_torun[clip_uid][i]["action_idx"] + nprev - 1) in keys_done:
            continue 
        torun.append((clip_uid, i))

random.shuffle(torun)

def generate_per_clip_uid_action_idx(clip_uid_action_idx):
    clip_uid, i = clip_uid_action_idx
    action_key = clip_uid + "_" + str(ann_torun[clip_uid][i]["action_idx"] + nprev - 1)
  
    #try:
    tr_prompt = annotation_utils_new.get_tr_prompt(
        clip_uid=clip_uid, 
        i=max(i - (args.max_nprev - nprev), 0), 
        nprev=min(args.max_nprev, ann_torun[clip_uid][i]["action_idx"] - ann_torun[clip_uid][0]["action_idx"] + nprev), 
        annotations=ann_torun,
        all_caption=all_caption,
        all_question=all_question,
        clip2video=clip2video,
        video2scenario=video2scenario,
        pred_action=pred_action,
        tax=tax,
        is_test=True, 
        noun_list=noun_list, 
        is_scenario=is_scenario,
        is_pred_action=is_pred_action,
        is_past_action=is_past_action,
        use_narration=use_narration,
        ncaption=ncaption,
    )
    if prompt_design in ["maxmargin", "semanticsim"]:
        captions_this, actions_this = tr_prompt.split("Narrations: ")[int(use_narration != "None")].split("Past actions: ")
        prompt_full = prompt.format(captions=captions_this, actions=actions_this) if use_narration != "None" else prompt.format(actions=actions_this)
    elif prompt_design == "GT":
        cnt = 0
        examples = ""
        while cnt < ntot_example:
            l = len(ann_torun[clip_uid])
            if(l <= 20 + nprev):
                continue 
            i_foridx = random.choice(range(l - 20 - nprev))
            if l >= 3*(20+nprev)  and i_foridx in range(i-nprev, i+nprev):
                continue
            temp_prompt = annotation_utils_new.get_tr_prompt(
                clip_uid=clip_uid, 
                i=i_foridx, 
                nprev=nprev, 
                annotations=ann_torun,
                all_caption=all_caption,
                all_question=all_question,
                clip2video=clip2video,
                video2scenario=video2scenario,
                pred_action=pred_action,
                tax=tax,
                is_test=False, 
                noun_list="GT" if noun_list is not None else None, 
                is_scenario=is_scenario,
                is_pred_action="False",
                is_past_action=is_past_action,
                use_narration=use_narration,
                ncaption=ncaption,
            )
            examples += temp_prompt
            cnt += 1
        prompt_full = prompt + examples + tr_prompt
    else:
        prompt_full = prompt + tr_prompt

    pred_all = {}
    pred = {"verb": [], "noun": []}
    cntt = 0
    nfail = 0

    if is_vis:
        gen_list = []

    # Get the last action of 8 previous actions for later
    if is_pred_action == "True":
        last_action_aggregated = {"verb": [], "noun": []}
        for vpd, npd in zip(pred_action[action_key]["verb"], pred_action[action_key]["noun"]):
            last_action_aggregated["verb"].append(vpd[-1]) # only interested in the last action
            last_action_aggregated["noun"].append(npd[-1])

        assert len(last_action_aggregated["verb"]) == 5

        get_most_common = lambda x : max(set(x), key=x.count)
        prev_vfind = get_most_common(last_action_aggregated["verb"])
        prev_nfind = get_most_common(last_action_aggregated["noun"]) 
    else:
        last_action = ann_torun[clip_uid][i + nprev -1]
        prev_vfind = vmap[last_action["verb"].split("_(")[0].replace('_',' ')]
        prev_nfind = nmap[last_action["noun"].split("_(")[0].replace('_',' ')]


    while True:
        if args.use_narration != "None":
            seperation_str = "Narrations: "
        else:
            seperation_str = "Past actions: "

        instruct = prompt_full[:prompt_full.find(seperation_str)]
        examples = [seperation_str[:-1] + x for x in prompt_full[prompt_full.find(seperation_str) : prompt_full.rfind(seperation_str)].split(seperation_str[:-1]) if x != ""]
        query = prompt_full[prompt_full.rfind(seperation_str):]
        
        for examples_this in zip(*[iter(examples)] * nexample):
            if args.wo_examples == "True":
                prompt_this = [instruct] + ["<Problem>\n" + query + '\nFuture actions:']
                prompt_this = "".join(prompt_this)     
            else:           
                examples = ""
                for ex in list(examples_this):
                    narations = ex[:ex.find('Past actions:')]
                    temp = ex[ex.find('Past actions:'):].split('),')
                    past_actions = '),'.join(temp[:8]) + '),\n'
                    future_actions = 'Future actions:' + '),'.join(temp[8:])             

                    examples += narations + past_actions + future_actions


                prompt_this = [instruct] + ["<Example>\n" + examples] + ["<Problem>\n" + query + '\nFuture actions:']
                prompt_this = "".join(prompt_this)
         
            number_of_tokens = len(tokenizer(prompt_this)['input_ids'])
            if number_of_tokens > 4000:
                print(f"Out of token! Current token length {number_of_tokens}")
                nfail += 1
                continue 
            if text_generation_model.startswith("api"): # api_httppath
                gen = generator({
                    "prompt" : prompt_this,
                    "max_new_tokens" : min(200, 4096 - number_of_tokens),
                    "num_return_sequences" : 1,
                    "top_k" : top_k,
                    "top_p" : top_p,
                    "logit_processor": "Falcon"
                })
            elif "gpt" in text_generation_model:
                gen = generator(prompt_this, 
                    max_new_tokens=min(200, 4096 - number_of_tokens),
                    num_return_sequences=1,
                    pad_token_id=50256,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True
                )
            else:
                gen = generator(prompt_this, 
                    max_new_tokens=min(200, 4096 - number_of_tokens), 
                    num_return_sequences=1,
                    pad_token_id=50256,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    eos_token_id=generator.tokenizer.eos_token_id
                ) 
            
            for x in gen:
                pred_this = []
                try:
                    #print(x["generated_text"][len(prompt_this):], flush=True)
                    # __________, (put, pot), (take, lid), (take, spoon), (turn on, switch), (adjust, pot), (cover, pot), (take, spoon), (hold, spatula), __________, (put, pot), (take, lid), (take, spoon), (turn on, switch), (adjust, pot), (cover, pot),
                    for vn in x["generated_text"][len(prompt_this):].split(".")[0].split("), "):
                        vfind, nfind = annotation_utils_new.get_parsed(vn, vmap, nmap)
                        if len(vfind) and len(nfind): # found both verb and noun
                            pred_this.append((vmap[vfind[0]], nmap[nfind[0]]))
                            prev_vfind = vmap[vfind[0]]
                            prev_nfind = nmap[nfind[0]]
                        elif len(vfind): # only found verb
                            pred_this.append((vmap[vfind[0]], prev_nfind))
                            prev_vfind = vmap[vfind[0]]
                        elif len(nfind): # only found noun
                            pred_this.append((prev_vfind, nmap[nfind[0]]))
                            prev_nfind = nmap[nfind[0]]

                    if len(pred_this) == 0:
                        raise Exception
                    if len(pred_this) < 20:
                        if args.padding == "last":
                            pred_this += [pred_this[-1]] * (20 - len(pred_this)) 
                        elif args.padding == "same":
                            while len(pred_this) < 20:
                                pred_this += pred_this
                except Exception as e:
                    continue 
                
                pred_this = pred_this[:20]

                cntt += 1
                pred["verb"].append([x[0] for x in pred_this][:20])
                pred["noun"].append([x[1] for x in pred_this][:20])
                
                if is_vis:
                    gen_list.append(x)

                if cntt == 5:
                    break 
            
            if cntt == 5:
                break 

        nfail += 1
        if nfail > 10 or cntt >= 5:
            break 

    if nfail > 10:
        if len(pred["verb"]):
            while len(pred["verb"]) < 5:
                pred["verb"].append(pred["verb"][-1])
                pred["noun"].append(pred["noun"][-1])
        else:
            print("Failed to generate using the following prompt: " + tr_prompt)
            return 

    pred_all[action_key] = pred
    if version == "v2":
        with open("./llm-log/" + logfile, "a") as f:
            for w in ["verb", "noun"]:
                for pd in pred[w]:
                    f.write(action_key + " " + w + " " + " ".join([str(x) for x in pd]) + "\n")


with tqdm(total=len(torun)) as pbar:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(generate_per_clip_uid_action_idx, arg): arg for arg in torun}
        for future in concurrent.futures.as_completed(futures):
            pbar.update(1)

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

with open("./llm-log/" + logfile, "r") as f:
    for lines in itertools.zip_longest(*[f] * 10):
        clip_uid, idx = lines[0].split(" ")[0].split("_")
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
