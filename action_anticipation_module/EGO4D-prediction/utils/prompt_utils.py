import random 
from utils import annotation_utils

import os
import torch
from transformers import pipeline, AutoTokenizer

def get_generator(text_generation_model, device):
    if text_generation_model == "chatgpt":
        nexample = 4
        import openai 
        openai.api_key = ""
        def chatgpt_interface(pack):
            gpt_prompt, max_new_tokens, num_return_sequences, top_k, top_p = pack
            ntry = 0
            while ntry < 10:
                try:
                    gen = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": gpt_prompt}
                        ],
                        n=num_return_sequences,
                        stop=".",
                        max_tokens=max_new_tokens,
                        request_timeout=20,
                        top_p=top_p
                    )
                    return [{
                        "generated_text" : gpt_prompt + x["message"]["content"]
                    } for x in gen["choices"]]
                except Exception as e:
                    print(e)
                    ntry += 1
                    continue 
        generator = chatgpt_interface
    elif text_generation_model == "hf_endpoints":
        import requests
        headers = {
            'Authorization': 'Bearer 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',
            'Content-Type': 'application/json',
        }
        def hf_endpoint_interface(gpt_prompt, max_new_tokens, num_return_sequences, pad_token_id, top_k, top_p):
            ntry = 0
            while True:
                try:
                    json_data = {
                        "inputs": gpt_prompt, 
                        "parameters": {
                            "max_new_tokens": max_new_tokens, 
                            "num_return_sequences": num_return_sequences, 
                            "pad_token_id": pad_token_id, 
                            "top_k": top_k, 
                            "top_p": top_p,
                            "do_sample": True if num_return_sequences > 1 else False,
                        }
                    }
                    response = requests.post(
                        'https://q2bl01apn41aamgv.us-east-1.aws.endpoints.huggingface.cloud',
                        headers=headers,
                        json=json_data,
                    )

                    return json.loads(response.text)
                except:
                    ntry += 1
                if ntry >= 5:
                    return [{
                        "generated_text" : gpt_prompt + "",
                    } for x in range(num_return_sequences)]
        generator = hf_endpoint_interface
    elif text_generation_model.startswith("rpc"): # rpc_model_ip_port
        import rpyc
        print("connecting to: ", text_generation_model.split("_")[2], int(text_generation_model.split("_")[3]))
        try:
            connection = rpyc.connect(text_generation_model.split("_")[2], int(text_generation_model.split("_")[3]), config={"sync_request_timeout": 300})
        except:
            connection = rpyc.ssh_connect()
        connection.root.set_pipeline(text_generation_model.split("_")[1], device="cuda:0")
        generator = connection.root.generate
    elif text_generation_model.startswith("api"): # api_httppath
        import json 
        import requests
        http_path = text_generation_model.split("_")[-1]
        def generator_api(pack):
            ntry = 0
            while True:
                try:
                    decoded = requests.post("http://{}/generation/".format(
                        http_path, 
                    ), json={
                        "prompt" : pack["prompt"],
                        "max_new_tokens" : pack["max_new_tokens"],
                        "num_return_sequences" : pack["num_return_sequences"],
                        "top_k" : pack["top_k"],
                        "top_p" : pack["top_p"],
                        "logit_processor" : pack.get("logit_processor", "None"),
                    }, timeout=1000)
                    return json.loads(decoded.text)
                except Exception as e:
                    print(e)
                    ntry += 1
                if ntry >= 5:
                    return [{
                        "generated_text" : pack["prompt"] + "",
                    } for x in range(pack["num_return_sequences"])]
        generator = generator_api
    elif text_generation_model == 'falcon':
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        generator = pipeline('text-generation', model="tiiuae/falcon-7b", tokenizer=tokenizer, device_map="auto", \
                            torch_dtype=torch.bfloat16, trust_remote_code=True)
    elif 'llama' in text_generation_model.lower():
        from transformers import LlamaForCausalLM, LlamaTokenizer
        #tokenizer = LlamaTokenizer.from_pretrained(f"/cluster/project/cvg/students/azaera/cache_hf/{text_generation_model}")
        tokenizer = AutoTokenizer.from_pretrained(f"/cluster/project/cvg/students/azaera/cache_hf/{text_generation_model}")
        #model = LlamaForCausalLM.from_pretrained(f"/cluster/project/cvg/students/azaera/cache_hf/{text_generation_model}", device_map='auto') #device_map="auto", offload_folder="offload", offload_state_dict=True)
        #model = LlamaModel.from_pretrained(f"/cluster/work/cvl/xiwang1/sankim/model/{text_generation_model}", device_map='auto')
        generator = pipeline('text-generation', model=f"/cluster/project/cvg/students/azaera/cache_hf/{text_generation_model}", tokenizer=tokenizer, device_map="auto",  model_kwargs={"torch_dtype": torch.bfloat16})
        return generator, tokenizer
    else:
        generator = pipeline('text-generation', model=text_generation_model, device_map="auto") 

    return generator 


def get_prompt(prompt_design, ann_for_prompt, nprev, nexample, fetch_k, lambda_mult, embedding_model,
    all_caption, all_question,
    clip2video, video2scenario, pred_action, tax, noun_list, is_scenario, is_past_action, use_narration, ncaption, use_narration_summary,
    device,
):

    prompt = "You are going to complete an action sequence, an action is one (verb, noun) pair. Given 8 past actions below, you are going to predict next 20 actions.{}{}".format(
             " You are also given a text description (Narrations) of the past actions for reference." if use_narration != "None" else "",
             " As an aid, next to the past actions, you will be given a list of objects that were present while the actions were being performed." if all_question["objects"] is not None else "")

    cnt = 0
    if prompt_design == "random":
        while cnt < nexample:
            clip_uid = random.choice(list(ann_for_prompt.keys()))
            l = len(ann_for_prompt[clip_uid])
            if(l <= 20 + nprev):
                continue 
            i_foridx = random.choice(range(l - 20 - nprev))
            i = ann_for_prompt[clip_uid][i_foridx]["action_idx"]
            allin = True
            for x in range(i, i + nprev):
                if clip_uid + "_" + str(x) not in all_caption:
                    allin = False
                if all_question !=None:
                    for _question in ["action", "location", "intention", "interaction", "prediction"]:
                        if len(all_question[_question]) and clip_uid + "_" + str(x) not in all_question[_question]:
                            allin = False 
                            break 
            if not allin:
                continue 
            tr_prompt = annotation_utils.get_tr_prompt(
                clip_uid=clip_uid, 
                i=i_foridx, 
                nprev=nprev, 
                annotations=ann_for_prompt,
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
                use_narration_summary=use_narration_summary
            )
            prompt += tr_prompt
            cnt += 1
    elif prompt_design in ["kmeans", "spectral"]:
        import json 
        while cnt < nexample:
            with open("/local/home/huangd/clip_baseline_LTA_Ego4d/output/{}_val_v2.json".format(prompt_design), "r") as f:
                kmeans_cluster = json.load(f)
            for cluster in range(nexample):
                for clip_uid_action_idx in kmeans_cluster["{}-sorted".format(cluster)]:
                    clip_uid, action_idx = clip_uid_action_idx.split("_")
                    i = int(action_idx)
                    i_foridx = i - ann_for_prompt[clip_uid][0]["action_idx"]

                    l = len(ann_for_prompt[clip_uid])
                    if(l <= 20 + nprev):
                        continue 

                    allin = True
                    for x in range(i, i + nprev):
                        if clip_uid + "_" + str(x) not in all_caption:
                            allin = False
                        if all_question !=None:    
                            for _question in ["action", "location", "intention", "interaction", "prediction"]:
                                if len(all_question[_question]) and clip_uid + "_" + str(x) not in all_question[_question]:
                                    allin = False 
                                    break 
                    if not allin:
                        continue 
                    tr_prompt = annotation_utils.get_tr_prompt(
                        clip_uid=clip_uid, 
                        i=i_foridx, 
                        nprev=nprev, 
                        annotations=ann_for_prompt,
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
                        use_narration_summary=use_narration_summary
                    )
                    prompt += tr_prompt
                    cnt += 1
                    break 
        print("NOTICE: using kmeans prompt")
    elif prompt_design in ["maxmargin", "semanticsim"]:
        print(len(ann_for_prompt))
        examples = []
        for clip_uid in ann_for_prompt:
            l = len(ann_for_prompt[clip_uid])
            if(l <= 20 + nprev):
                continue 
            for i_foridx in range(l - 20 - nprev):
                i = ann_for_prompt[clip_uid][i_foridx]["action_idx"]
                allin = True
                for x in range(i, i + nprev):
                    if clip_uid + "_" + str(x) not in all_caption:
                        print(clip_uid + "_" + str(x) + " not in all_caption")
                        allin = False
                    # if all_question !=None:
                    #     for _question in ["action", "location", "intention", "interaction", "prediction"]:
                    #         if len(all_question[_question]) and clip_uid + "_" + str(x) not in all_question[_question]:
                    #             allin = False 
                    #             break 
                if not allin:
                    continue 

                prompt_this = annotation_utils.get_tr_prompt(
                    clip_uid=clip_uid, 
                    i=i_foridx, 
                    nprev=nprev, 
                    annotations=ann_for_prompt,
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
                    use_narration_summary=use_narration_summary
                )
                captions_this, actions_this = prompt_this.split("Narrations: ")[int(use_narration != "None")].split("Past actions: ") #TODO
                examples.append({
                    "captions" : captions_this,
                    "actions" : actions_this,
                } if use_narration != "None" else {
                    "actions" : actions_this,
                })
                # captions_this, actions_this = prompt_this.split("Objects present during the 8 past actions: ")[int(all_question["objects"] is not None)].split("Past actions: ")
                # examples.append({
                #     "captions" : captions_this,
                #     "actions" : actions_this,
                # } if (all_question["objects"] is not None) else {
                #     "actions" : actions_this,
                # })

        import sys 
        #sys.path.append("/cluster/work/cvl/xiwang1/sankim/data/langchain00157")
        from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector, SemanticSimilarityExampleSelector
        from langchain.vectorstores import FAISS
        from langchain.prompts import FewShotPromptTemplate, PromptTemplate

        input_variables = ["captions", "actions"] if use_narration != "None" else ["actions"] #TODO
        template = "Narrations: {captions}Past actions: {actions}" if use_narration != "None" else "Past actions: {actions}"
        # input_variables = ["captions", "actions"] if all_question["objects"] is not None else ["actions"]
        # template = "Objects present during the 8 past actions: {captions}Past actions: {actions}" if all_question["objects"] is not None else "Past actions: {actions}"

        example_prompt = PromptTemplate(
            input_variables=input_variables,
            template=template,
        )

        if embedding_model == "mpnet":
            from langchain.embeddings import HuggingFaceEmbeddings
            model_name = "/cluster/project/cvg/students/azaera/cache_hf/langchain/sentence-transformers/all-mpnet-base-v2" #"sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': device}
            embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        elif embedding_model == "openai":
            from langchain.embeddings import OpenAIEmbeddings
            import os 
            os.environ["OPENAI_API_KEY"] = ""
            embeddings = OpenAIEmbeddings()

        if prompt_design == "maxmargin":
            print(len(examples))
            print(examples[0])
            example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                examples, 
                embeddings, 
                FAISS, 
                k=nexample,
                fetch_k=fetch_k,
                #lambda_mult=lambda_mult,
            )
        elif prompt_design == "semanticsim":
            example_selector = SemanticSimilarityExampleSelector.from_examples(
                examples, 
                embeddings, 
                FAISS, 
                k=nexample,
            )
        mmr_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=prompt,
            suffix=template, 
            input_variables=input_variables,
        )

        return mmr_prompt

    else:
        print("missing prompt design")
        exit()

    return prompt