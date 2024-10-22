import os
import json
import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer

from model.model import FrozenInTime

# NOTE: Variables to change (input and output folders)
annotation_folder = "/cluster/scratch/azaera/ego4d/v2/annotations"
out_folder = "/cluster/project/cvg/students/azaera/noun_egovlp_distilbert_embs_cls"
egovlp_path = "/cluster/scratch/azaera/models/egovlp.pth"
os.makedirs(out_folder, exist_ok=True)

with open(f"{annotation_folder}/fho_lta_taxonomy.json", "r") as f:
    lta_taxonomy = json.load(f)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
video_params= {
                "model": "SpaceTimeTransformer",
                "arch_config": "base_patch16_224",
                "num_frames": 16,
                "pretrained": True,
                "time_init": "zeros"
            }
text_params = {
    "model": "distilbert-base-uncased",
    "pretrained": True,
    "input": "text"
}
model = FrozenInTime(video_params, text_params, load_checkpoint=egovlp_path)

model.to("cuda")
model.eval()

def noun_inference(noun_t, noun_idx):
    if noun_t == "sphygmomanometer":
        noun_t = "blood pressure monitor"
    if noun_t == "lubricant":
        noun_t = "grease"
    if noun_t == "derailleur":
        noun_t = "bike gear shifter"
    if noun_t == "puzzle or game piece":
        noun_t = "game board piece"
    noun = tokenizer(noun_t, return_tensors='pt', padding=True, truncation=True)
    noun = {key: val.cuda() for key, val in noun.items()}

    text_embed = model.compute_text_tokens(noun)[0]
    num_words = noun['attention_mask'][0].sum()
    #text_embed = text_embed[1 : num_words-1] # remove CLS and SEP tokens
    text_embed = text_embed[0] # CLS token

    torch.save(text_embed, os.path.join(out_folder, f'{noun_idx}.pt'))


for i, noun in enumerate(tqdm(lta_taxonomy['nouns'])):
    noun_inference(noun.split("_(")[0].replace('_', ' '), i)