import os 
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "8" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "8" # export NUMEXPR_NUM_THREADS=1
import numpy as np 
import json 
from PIL import Image 
import torch 
import transformers
transformers.set_seed(42) # does it really work under load balancing?
from transformers import pipeline, AutoTokenizer
from transformers import Blip2Processor, Blip2ForConditionalGeneration
#from lavis.models import load_model_and_preprocess
import asyncio 
from pydantic import BaseModel
from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute
from typing import Callable

import time 

import sample_utils

print("###################################")
print(os.environ["CUDA_VISIBLE_DEVICES"])

app = FastAPI()
# pip install "uvicorn[standard]" gunicorn transformers fastapi
# run with gunicorn api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8964 -t 600 -c ./config.py

class ImageQuery(BaseModel):
    image:str = None
    prefix:str = "A person is"
    max_new_tokens : int = 80
    ncaption : int = 1
    top_k : int = 5
    top_p : float = 0.75

class TextQuery(BaseModel):
    prompt:str = None
    max_new_tokens:int = 80
    num_return_sequences:int = 5
    pad_token_id:int = 50256
    top_k : int = 50
    top_p : float = 5
    logit_processor : str = "None"


app = FastAPI()
model_card = "gpt2-xl" # "instructblip" # "Salesforce/blip2-opt-2.7b" 
idxcard = int(os.environ["APP_WORKER_ID"]) - 1

worker_id_to_card = {
    0 : 0,
    1 : 0
}

idxcard = worker_id_to_card[idxcard]

if "blip2" in model_card:
    processor, image_to_text = (
        Blip2Processor.from_pretrained(model_card), 
        Blip2ForConditionalGeneration.from_pretrained(
            model_card, torch_dtype=torch.float16,
    ).to(idxcard))
elif "gpt-neo" in model_card:
    generator = pipeline('text-generation', model=model_card, device=idxcard) 
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    from optimum.onnxruntime import ORTModelForCausalLM
    ort_model = ORTModelForCausalLM.from_pretrained(
        model_card,
        export=True,
        provider="CUDAExecutionProvider",
        use_merged=True,
    )
    generator = pipeline('text-generation', model=ort_model, tokenizer=tokenizer, device=idxcard) 
    '''
elif "gpt2" in model_card:
    generator = pipeline('text-generation', model=model_card, device=idxcard) 
elif "instructblip" in model_card:
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5_instruct",
        model_type="flant5xl",
        is_eval=True,
        device=idxcard,
    )

# block the loop with async w/o await, s.t. one request per worker process
# no external thread pool used
@app.post("/caption/")
async def caption_image(query : ImageQuery):
    images = [Image.fromarray(np.array(json.loads(query.image), dtype=np.uint8))]
    na_this = image_to_text.generate(
        **(processor(
            images=images,
            text=query.prefix,
            return_tensors="pt",
        ).to(idxcard, torch.float16)), 
        max_new_tokens=query.max_new_tokens, 
        do_sample=True if query.ncaption > 1 else False,
        num_return_sequences=query.ncaption, 
        top_k=query.top_k, 
        top_p=query.top_p, 
    )
    decoded = [x.strip() for x in processor.batch_decode(
        na_this, 
        skip_special_tokens=True
    )]
    return decoded

@app.post("/caption_instructblip/")
async def caption_image(query: ImageQuery):
    use_nucleus_sampling = False 
    image = Image.fromarray(np.array(json.loads(query.image), dtype=np.uint8))
    image = vis_processors["eval"](image).unsqueeze(0).to(idxcard)

    samples = {
        "image": image,
        "prompt": query.prefix,
    }

    output = model.generate(
        samples,
        length_penalty=1.,
        repetition_penalty=1.,
        num_beams=5,
        max_length=250,
        min_length=1,
        top_p=0.9,
        use_nucleus_sampling=use_nucleus_sampling,
    )

    return output



@app.post("/generation/")
async def generation_text(query : TextQuery):
    if query.logit_processor == "None":
        return generator(query.prompt, 
            max_new_tokens=query.max_new_tokens, 
            num_return_sequences=query.num_return_sequences, 
            pad_token_id=query.pad_token_id,
            top_k=query.top_k,
            top_p=query.top_p,
            do_sample=True if query.num_return_sequences > 1 else False,
        )
    elif query.logit_processor == "Ego4D_all":
        return  generator(query.prompt, 
            max_new_tokens=query.max_new_tokens, 
            num_return_sequences=query.num_return_sequences, 
            pad_token_id=query.pad_token_id,
            top_k=query.top_k,
            top_p=query.top_p,
            do_sample=True if query.num_return_sequences > 1 else False,
            **{'logits_processor' : [sample_utils.logit_processor]},
        )