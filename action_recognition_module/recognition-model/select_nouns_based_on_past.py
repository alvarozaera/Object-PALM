import json
import torch
import random
import collections
from tqdm import tqdm

# Variables to select the POC case
split = "train"
annotation_folder = "ego4d_data/v2/annotations"
N_NOUNS_TO_SAMPLE = 9 #19
N_NOUNS_TO_SAMPLE_FULLY_RANDOM = 3 #10
GT_RATIO = 1.0

def get_dset(split):
    with open("{}/fho_lta_{}.json".format(annotation_folder, split), "r") as f:
        dset = json.load(f)

    annotations = collections.defaultdict(list)
    for entry in dset["clips"]:
        annotations[entry['clip_uid']].append(entry)

    # Sort windows by their PNR frame (windows can overlap, but PNR is distinct)
    annotations = {
        clip_uid: sorted(annotations[clip_uid], key=lambda x: x['action_idx'])
        for clip_uid in annotations
    }

    return annotations


with open("./ego4d_data/v2/annotations/fho_lta_taxonomy.json", "r") as f:
    lta_taxonomy = json.load(f)

n_nouns = len(lta_taxonomy["nouns"])

annots = get_dset(split)

json_out = {}

for clip_uid in tqdm(annots):
    nouns_clip = set()
    for ann in annots[clip_uid]:
        action_idx = ann['action_idx']
        noun_label = ann['noun_label']
        nouns_action = []
        # SELECT N_NOUNS RANDOM NOUNS FROM THE PAST OF THE VIDEO (IF THERE ARE NOT ENOUGH, SELECT RANDOMLY)
        # EXCLUDE GT NOUN TO AVOID REPETITION (ALL NOUNS MUST BE DIFFERENT)
        if len(nouns_clip) < N_NOUNS_TO_SAMPLE:
            to_sample = set(range(n_nouns))
            to_sample.remove(noun_label)
            nouns_action = random.sample(sorted(to_sample), N_NOUNS_TO_SAMPLE)
        else:
            to_sample = set(nouns_clip)
            if noun_label in to_sample:
                to_sample.remove(noun_label)
            nouns_action = random.sample(sorted(to_sample), N_NOUNS_TO_SAMPLE - N_NOUNS_TO_SAMPLE_FULLY_RANDOM)
            to_sample = set(range(n_nouns)) - set(nouns_action)
            to_sample.remove(noun_label)
            nouns_action += random.sample(sorted(to_sample), N_NOUNS_TO_SAMPLE_FULLY_RANDOM)

        nouns_clip.add(noun_label)

        # Add GT based on GT_RATIO
        if random.random() < GT_RATIO:
            nouns_action.append(noun_label)
        else:
            to_sample = set(range(n_nouns)) - set(nouns_action)
            to_sample.remove(noun_label)
            last_noun = random.sample(sorted(to_sample), 1)[0]
            nouns_action.append(last_noun)

        # shuffle to get correct noun in different position
        random.shuffle(nouns_action)

        #print(f"{clip_uid}_{action_idx}", *nouns_action)
        json_out[f"{clip_uid}_{action_idx}"] = nouns_action

with open(f"poc_{N_NOUNS_TO_SAMPLE+1}_nouns_{N_NOUNS_TO_SAMPLE - N_NOUNS_TO_SAMPLE_FULLY_RANDOM}_past_{N_NOUNS_TO_SAMPLE_FULLY_RANDOM}_rand_no_rep_{split}.json", "w") as f:
    json.dump(json_out, f)
