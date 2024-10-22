'''
!pip install rpyc > /dev/null
!pip install transformers==4.27.2 > /dev/null
!pip install accelerate > /dev/null
'''


import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
import torch
from rpyc.utils.server import ThreadedServer
import transformers
from transformers import pipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from PIL import Image
import numpy as np

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

model_cache = {} # "{}_{}".format(model_card, device)

ngpu = len(get_gpu_memory())

@rpyc.service
class AllService(rpyc.Service):
    def __init__(self):
        self.set = False
        self.set_seed = False

    def _set_pipeline_from_cached(self, model_card, model_cached):
        if "gpt" in model_card:
            self.generator = model_cache[model_cached]
        elif "instructblip2" in model_card:
            self.model, self.vis_processors, _ = model_cache[model_cached]
        elif "blip2" in model_card:
            self.processor, self.image_to_text = model_cache[model_cached]
        else:
            print("missing model card!")
            raise Exception 
        return 

    def _set_pipeline(self, model_card, idxcard):
        model_cached = "{}_{}".format(model_card, idxcard)
        if "gpt" in model_card:
            model_cache[model_cached] = pipeline('text-generation', model=model_card, device=idx_card)
            self.generator = model_cache[model_cached]
        elif "instructblip2" in model_card:
            model_cache[model_cached] = load_model_and_preprocess(
                name=model_card,
                model_type="flant5xxl",
                is_eval=True,
                device=idx_card,
            )
            self.model, self.vis_processors, _ = model_cache[model_cached]
        elif "blip2" in model_card:
            model_cache[model_cached] = (
                Blip2Processor.from_pretrained(model_card), 
                Blip2ForConditionalGeneration.from_pretrained(
                    model_card, torch_dtype=torch.float16,
            ).to(idxcard))
            self.processor, self.image_to_text = model_cache[model_cached]
        else:
            print("missing model card!")
            raise Exception 
        return 

    @rpyc.exposed
    def set_pipeline(self, model_card):
        self.model_card = model_card
        print("generator set to ", model_card)
        for cached in model_cache:
            if model_card in cached:
                self._set_pipeline_from_cached(model_card, model_cache[cached])
                self.device = int(cached.split("_")[-1])
                print("find in cached model:", cached)
                return 
        # find a empty card
        filled = [0 for i in range(ngpu)]
        for cached in model_cache:
            filled[int(cached.split("_")[-1])] = 1
        for idxcard, idxfilled in enumerate(filled):
            if idxfilled == 1:
                continue 
            self._set_pipeline(model_card, idxcard)
            self.device = idxcard
            return 
        # find a least occupied card
        freemem = get_gpu_memory()
        idxcard = np.argmax(freemem)
        self._set_pipeline(model_card, idxcard)
        self.device = idxcard
        return 

    def _generate_pipeline(self, input, max_new_tokens=80, num_return_sequences=5, pad_token_id=50256,
            top_k=10,
            top_p=0.75,):
        if not self.set_seed:
            transformers.set_seed(42)
            self.set_seed = True 
        return self.generator(input, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id,
            top_k=top_k,
            top_p=top_p,
            do_sample=True if num_return_sequences > 1 else False,
        )

    def _generate_instructblip(self, images, shape, text, min_len=1, max_len=80, beam_size=5, len_penalty=1, repetition_penalty=1, max_new_tokens=80, ncaption=1, top_k=5, top_p=0.9, decoding_method="Beam search"):
        prompt = text
        if type(images) != list:
            images = Image.fromarray(np.frombuffer(images, dtype=np.uint8).reshape(shape))
        else:
            images = [Image.fromarray(np.frombuffer(x, dtype=np.uint8).reshape(shape)) for x in images]
            
        use_nucleus_sampling = decoding_method == "Nucleus sampling"
        print(prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, use_nucleus_sampling)
        image = self.vis_processors["eval"](images).unsqueeze(0).to(self.device)

        samples = {
            "image": image,
            "prompt": prompt,
        }

        output = self.model.generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=use_nucleus_sampling,
        )
        print(output)

        return output

    def _generate_blip2(self, images, shape, text, max_new_tokens, ncaption, top_k=5, top_p=0.75):
        if not self.set_seed:
            transformers.set_seed(42)
            self.set_seed = True 
        if type(images) != list:
            images = Image.fromarray(np.frombuffer(images, dtype=np.uint8).reshape(shape))
        else:
            images = [Image.fromarray(np.frombuffer(x, dtype=np.uint8).reshape(shape)) for x in images]
        na_this = self.image_to_text.generate(
            **(self.processor(
                images=images,
                text=text,
                return_tensors="pt",
            ).to(self.device, torch.float16)), 
            max_new_tokens=max_new_tokens, 
            do_sample=True if ncaption > 1 else False,
            num_return_sequences=ncaption, 
            top_k=top_k, 
            top_p=top_p, 
        )
        decoded = [x.strip() for x in self.processor.batch_decode(
            na_this, 
            skip_special_tokens=True
        )]
        return decoded

    @rpyc.exposed
    def generate(*args, **kwargs):
        if "gpt" in self.model_card:
            return self._generate_pipeline(*args, **kwargs)
        if "instructblip2" in self.model_card:
            return self._generate_instructblip(*args, **kwargs)
        if "blip2" in self.model_card:
            return self._generate_blip2(*args, **kwargs)



@rpyc.service
class TestPipelineService(rpyc.Service):
    def __init__(self):
        self.set = False
        self.set_seed = False
    @rpyc.exposed
    def set_pipeline(self, text_generation_model):
        if self.set:
            return 
        print("generator set to ", text_generation_model)
        if text_generation_model not in model_cache:
            model_cache[text_generation_model] = pipeline('text-generation', model=text_generation_model, device=device)
        self.generator = model_cache[text_generation_model]
        self.set = True 

    @rpyc.exposed
    def generate(self, input, max_new_tokens=80, num_return_sequences=5, pad_token_id=50256,
            top_k=10,
            top_p=0.75,):
        if not self.set_seed:
            transformers.set_seed(42)
            self.set_seed = True 
        return self.generator(input, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id,
            top_k=top_k,
            top_p=top_p,
            do_sample=True if num_return_sequences > 1 else False,
        )

@rpyc.service
class TestBlipService(rpyc.Service):
    def __init__(self):
        self.set = False
        self.set_seed = False
    @rpyc.exposed
    def set_pipeline(self, text_generation_model, device):
        if self.set:
            return 
        print("generator set to ", text_generation_model)
        if text_generation_model not in model_cache:
            model_cache[text_generation_model] = (
                Blip2Processor.from_pretrained(text_generation_model), 
                Blip2ForConditionalGeneration.from_pretrained(
                    text_generation_model, torch_dtype=torch.float16,
            ).to(device))
        self.processor, self.image_to_text = model_cache[text_generation_model]
        self.device = device


    @rpyc.exposed
    def generate(self, images, shape, text, max_new_tokens, ncaption, top_k=5, top_p=0.75):
        if not self.set_seed:
            transformers.set_seed(42)
            self.set_seed = True 
        if type(images) != list:
            images = Image.fromarray(np.frombuffer(images, dtype=np.uint8).reshape(shape))
        else:
            images = [Image.fromarray(np.frombuffer(x, dtype=np.uint8).reshape(shape)) for x in images]
        na_this = self.image_to_text.generate(
            **(self.processor(
                images=images,
                text=text,
                return_tensors="pt",
            ).to(self.device, torch.float16)), 
            max_new_tokens=max_new_tokens, 
            do_sample=True if ncaption > 1 else False,
            num_return_sequences=ncaption, 
            top_k=top_k, 
            top_p=top_p, 
        )
        decoded = [x.strip() for x in self.processor.batch_decode(
            na_this, 
            skip_special_tokens=True
        )]
        return decoded

@rpyc.service
class TestInstructBlipService(rpyc.Service):
    def __init__(self):
        self.set = False
        self.set_seed = False
    @rpyc.exposed
    def set_pipeline(self, text_generation_model, device):
        if self.set:
            return 
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        print('Loading model...')
        if text_generation_model not in model_cache:
            model_cache[text_generation_model] = load_model_and_preprocess(
                name=text_generation_model,
                model_type="flant5xxl",
                is_eval=True,
                device=self.device,
            )
            
        self.model, self.vis_processors, _ = model_cache[text_generation_model]

        print('Loading model done!')



    @rpyc.exposed
    def generate(self, images, shape, text, min_len=1, max_len=80, beam_size=5, len_penalty=1, repetition_penalty=1, max_new_tokens=80, ncaption=1, top_k=5, top_p=0.9, decoding_method="Beam search"):
        prompt = text
        if type(images) != list:
            images = Image.fromarray(np.frombuffer(images, dtype=np.uint8).reshape(shape))
        else:
            images = [Image.fromarray(np.frombuffer(x, dtype=np.uint8).reshape(shape)) for x in images]
            
        use_nucleus_sampling = decoding_method == "Nucleus sampling"
        print(prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, use_nucleus_sampling)
        image = self.vis_processors["eval"](images).unsqueeze(0).to(self.device)

        samples = {
            "image": image,
            "prompt": prompt,
        }

        output = self.model.generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=use_nucleus_sampling,
        )
        print(output)

        return output

print('starting server')
server = ThreadedServer(AllService, port=8964)
server.start()