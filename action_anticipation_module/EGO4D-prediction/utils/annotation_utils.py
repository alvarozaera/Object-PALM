import os 
import json 
import random
from collections import defaultdict

def get_clip2video(anno_path):

    clip2video = {}
    with open(anno_path, "r") as f:
        lta_train_json = json.load(f)
        
    for clip in lta_train_json["clips"]:
        clip2video.update({clip["clip_uid"] : clip["video_uid"]})

    return clip2video

def get_video2scenario():

    video2scenario = {}
    with open("/cluster/work/cvl/xiwang1/sankim/data/annotations/ego4d.json", "r") as f:
        ego4d = json.load(f)

    for video in ego4d["videos"]:
        video2scenario.update({video["video_uid"] : video["scenarios"]})

    return video2scenario

def get_vnmap(tax):

    vmap = {}
    for idxv, v in enumerate(tax["verbs"]):
        v = v.replace("_(", "*").replace("/", "*").replace(")", "").replace(",_", "*")
        for idxvv, vv in enumerate(v.split("*")):
            if vv == "":
                continue 
            vv = vv.replace("_", " ")
            if vv in vmap:
                if vmap[vv] != idxv and idxvv == 0:
                    pass
                else:
                    continue 
            vmap[vv] = idxv
        # break 

    nmap = {}
    for idxn, n in enumerate(tax["nouns"]): 
        #n = n.replace("_(", "*").replace("/", "*").replace(")", "").replace(",", "*")
        n = n.replace("_(", "*").replace("/", "*").replace(")", "").replace(",_", "*")
        for idxnn, nn in enumerate(n.split("*")):
            if nn == "":
                continue 
            nn= nn.replace("_", " ")
            #nn = nn.split('_')[0]
            if nn in nmap:
                if nmap[nn] != idxn and idxnn == 0:
                    pass
                else:
                    continue 
            nmap[nn] = idxn

    return vmap, nmap


def get_parsed(vn, vmap, nmap):
    vfind = []
    nfind = []
    vn = vn.replace('(','').replace(')','').replace('_,','').replace('_','').strip()

    vn_list = vn.split(',')
    if len(vn_list) ==1:
        found_verb = vn_list[0].strip()
        found_noun = ''
    if len(vn_list) ==2:
        found_verb = vn_list[0].strip()
        found_noun = vn_list[1].strip()
    elif len(vn_list) > 2:
        start_idx = 0
        for idx, vn in enumerate(vn_list):
            if vn == '':
                continue
            else:
                start_idx = idx
                break
        if start_idx+1 < len(vn_list):
            found_verb = vn_list[start_idx].strip()
            found_noun = vn_list[start_idx+1].strip()
        else:
            found_verb = vn_list[start_idx].strip()
            found_noun = ''

    for v in vmap:
        if v in found_verb:
            vfind.append(v)

    # "turn on " not "turn"
    vfind = [sorted(vfind, key=lambda x : -len(x))[0]] if len(vfind) else vfind

    for n in nmap:
        if n in found_noun:
            nfind.append(n)

    # "tape measure" not "tape"
    nfind = [sorted(nfind, key=lambda x : -len(x))[0]] if len(nfind) else nfind

    return vfind, nfind

from .detection_utils import unidet_mapping, unidet_classes

def get_tr_prompt(clip_uid, i, nprev, annotations, all_caption, clip2video, video2scenario,pred_action,tax,
    noun_list, is_scenario, is_test, is_pred_action, is_past_action, 
    use_narration, ncaption,
    all_question=None,
):
    tr_prompt = ""

    all_action = all_question.get("action", None)
    all_location = all_question.get("location", None)
    all_intention = all_question.get("intention", None)
    all_interaction = all_question.get("interaction", None)
    all_prediction = all_question.get("prediction", None)
    narrations = all_question.get("narration", None)
    ncaption = all_question.get("ncaption", 4)

    if use_narration == "GT":
        try:
            na_prompt = ""
            cnt = 0
            for narration in narrations[clip2video[clip_uid]]["narration_pass_1"]["narrations"]:
                if annotations[clip_uid][i]["action_clip_start_frame"] <= narration["timestamp_frame"] - annotations[clip_uid][i]["clip_parent_start_frame"] <= annotations[clip_uid][i + nprev - 1]["action_clip_end_frame"]:
                    cleaned = narration["narration_text"].replace("#C", "").replace("#c", "").replace("#O", "").replace("#unsure", "").replace("#Unsure", "").replace("\n", "").replace(". ", "")
                    cleaned = cleaned.strip()
                    #cleaned = cleaned.replace("C", "A person").strip()
                    na_prompt += cleaned + ". " if not cleaned.endswith(".") else ""
                    cnt += 1
                    if cnt > ncaption * 8:
                        break 
        except:
            na_prompt = "not available."
        finally:
            if na_prompt == "":
                na_prompt = "not available."

        if na_prompt == "not available.":
            na_prompt = []
                
            for idx in range(annotations[clip_uid][i]["action_idx"], annotations[clip_uid][i]["action_idx"] + nprev):
                def tsfm(tmp):
                    tmp = tmp.strip()
                    
                    tmp = tmp.split(".")[0]
                    if tmp.split(" ")[0].endswith("ing"):
                        tmp = "C is " + tmp
                    return "{}{}".format(tmp[:1].upper(), tmp[1:])
                na_list = [tsfm(x) for x in all_caption[clip_uid + "_" + str(idx)]]
                na_list_filtered = []
                for ii in range(len(na_list)):
                    add_filtered = True 
                    for jj in range(len(na_list)):
                        if jj == ii:
                            continue 
                        if na_list[ii] in na_list[jj]:
                            if len(na_list[ii]) == len(na_list[jj]) and ii < jj:
                                continue 
                            add_filtered = False
                            break 
                    if add_filtered:
                        na_list_filtered.append(na_list[ii])
                random.shuffle(na_list_filtered)
                na_prompt += na_list_filtered[:ncaption]      
            tr_prompt += "\nNarrations: " + ". ".join(na_prompt) + "."   
        else:
            tr_prompt += "\nNarrations: " + na_prompt

    elif use_narration != "None":
        na_prompt = []
            
        for idx in range(annotations[clip_uid][i]["action_idx"], annotations[clip_uid][i]["action_idx"] + nprev):
            if "imagecaption" in use_narration:
                def tsfm(tmp):
                    tmp = tmp.strip()
                    
                    tmp = tmp.split(".")[0]
                    if tmp.split(" ")[0].endswith("ing"):
                        tmp = "A person is " + tmp
                    return "{}{}".format(tmp[:1].upper(), tmp[1:])
                na_list = [tsfm(x) for x in all_caption[clip_uid + "_" + str(idx)]]
                na_list_filtered = []
                for ii in range(len(na_list)):
                    add_filtered = True 
                    for jj in range(len(na_list)):
                        if jj == ii:
                            continue 
                        if na_list[ii] in na_list[jj]:
                            if len(na_list[ii]) == len(na_list[jj]) and ii < jj:
                                continue 
                            add_filtered = False
                            break 
                    if add_filtered:
                        na_list_filtered.append(na_list[ii])
                random.shuffle(na_list_filtered)
                na_prompt += na_list_filtered[:ncaption]

            if "_action" in use_narration:
                action_list = all_action[clip_uid + "_" + str(idx - annotations[clip_uid][i]["action_idx"] + i)]
                action_max_freq = max(action_list, key=action_list.count)
                na_prompt += ["{}{}".format(action_max_freq[:1].upper(), action_max_freq[1:])]

            if "_location" in use_narration:
                location_list = all_location[clip_uid + "_" + str(idx - annotations[clip_uid][i]["action_idx"] + i)]
                location_max_freq = max(location_list, key=location_list.count)
                na_prompt += ["{}{}".format(location_max_freq[:1].upper(), location_max_freq[1:])]

            if "_interaction" in use_narration:
                interaction_list = all_interaction[clip_uid + "_" + str(idx - annotations[clip_uid][i]["action_idx"] + i)]
                interaction_max_freq = max(interaction_list, key=interaction_list.count)
                na_prompt += ["{}{}".format(interaction_max_freq[:1].upper(), interaction_max_freq[1:]) if interaction_max_freq.startswith("he") else "He is interacting with {}".format(interaction_max_freq)]

            if "_intention" in use_narration:
                intention_list = all_intention[clip_uid + "_" + str(idx - annotations[clip_uid][i]["action_idx"] + i)]
                intention_max_freq = max(intention_list, key=intention_list.count)
                na_prompt += ["{}{}".format(intention_max_freq[:1].upper(), intention_max_freq[1:])]

            if "_prediction" in use_narration:
                prediction_list = all_prediction[clip_uid + "_" + str(idx - annotations[clip_uid][i]["action_idx"] + i)]
                prediction_max_freq = max(prediction_list, key=prediction_list.count)
                na_prompt += ["{}{}".format(prediction_max_freq[:1].upper(), prediction_max_freq[1:])]
       

        tr_prompt += "\nNarrations: " + ". ".join(na_prompt) + "."

    if is_scenario:
        tr_prompt += "\nScenario: " + ", ".join(
            video2scenario[clip2video[clip_uid]]
        )

    if noun_list == "GT":
        tr_prompt += "\nNoun list: " + ", ".join([
            "{}".format(x) for x in list(set(
                [x["noun"].split("_(")[0].replace('_',' ') for x in annotations[clip_uid][(annotations[clip_uid][i]["action_idx"] + nprev):(annotations[clip_uid][i]["action_idx"] + 20 + nprev)]]
            ))
        ]) + "."

    pred_action_aggregated = None
    if noun_list == "Pred":
        pred_action_aggregated = defaultdict(lambda : {"verb": [], "noun": []})
        for idx in range(annotations[clip_uid][i]["action_idx"], (annotations[clip_uid][i]["action_idx"] + nprev)):
            key = clip_uid + "_" + str(idx)
            if not key in pred_action:
                continue
            for vpd, npd in zip(pred_action[key]["verb"], pred_action[key]["noun"]):
                offset = len(vpd) #4
                for j in range(offset):
                    pred_action_aggregated["{}_{}".format(clip_uid, idx + j - offset + 1)]["verb"].append(vpd[j])
                    pred_action_aggregated["{}_{}".format(clip_uid, idx + j - offset + 1)]["noun"].append(npd[j])

        for key in pred_action_aggregated:
            get_most_common = lambda x : max(set(x), key=x.count)
            pred_action_aggregated[key]["verb_agg"] = get_most_common(pred_action_aggregated[key]["verb"])
            pred_action_aggregated[key]["noun_agg"] = get_most_common(pred_action_aggregated[key]["noun"])

        prev_nouns = []
        for idx in range(annotations[clip_uid][i]["action_idx"], (annotations[clip_uid][i]["action_idx"] + nprev)):
            key = clip_uid + "_" + str(idx)
            prev_nouns.append(pred_action_aggregated[key]["noun_agg"])
        prev_nouns = list(set(prev_nouns))

        tr_prompt += "\nNoun list: " + ", ".join([
            tax["nouns"][noun].split("_(")[0].replace('_',' ') for noun in prev_nouns
        ]) + "."

    tr_prompt += "\nPast actions: "
    if is_past_action:
        if is_pred_action == "True":
            if not pred_action_aggregated:
                pred_action_aggregated = defaultdict(lambda : {"verb": [], "noun": []})
                for idx in range(annotations[clip_uid][i]["action_idx"], (annotations[clip_uid][i]["action_idx"] + nprev)):
                    key = clip_uid + "_" + str(idx)
                    if not key in pred_action:
                        continue
                    for vpd, npd in zip(pred_action[key]["verb"], pred_action[key]["noun"]):
                        offset = len(vpd) #4
                        for j in range(offset):
                            pred_action_aggregated["{}_{}".format(clip_uid, idx + j - offset + 1)]["verb"].append(vpd[j])
                            pred_action_aggregated["{}_{}".format(clip_uid, idx + j - offset + 1)]["noun"].append(npd[j])

                for key in pred_action_aggregated:
                    get_most_common = lambda x : max(set(x), key=x.count)
                    pred_action_aggregated[key]["verb_agg"] = get_most_common(pred_action_aggregated[key]["verb"])
                    pred_action_aggregated[key]["noun_agg"] = get_most_common(pred_action_aggregated[key]["noun"])

            tr_prompt += ", ".join(["({}, {})".format(
                tax["verbs"][pred_action_aggregated[clip_uid + "_" + str(idx)]["verb_agg"]].split("_(")[0].replace('_',' '), 
                tax["nouns"][pred_action_aggregated[clip_uid + "_" + str(idx)]["noun_agg"]].split("_(")[0].replace('_',' ')
            ) for idx in range(annotations[clip_uid][i]["action_idx"], (annotations[clip_uid][i]["action_idx"] + nprev))
            ]) + ", "

        elif "caption2vn" in is_pred_action:
            if not pred_action_aggregated:
                pred_action_aggregated = defaultdict(lambda : {"verb": [], "noun": []})
                for idx in range(annotations[clip_uid][i]["action_idx"], (annotations[clip_uid][i]["action_idx"] + nprev)):
                    key = clip_uid + "_" + str(idx)
                    if not key in pred_action:
                        continue
                    for vpd, npd in zip(pred_action[key]["verb"], pred_action[key]["noun"]):
                        offset = len(vpd) #4
                        for j in range(offset):
                            pred_action_aggregated["{}_{}".format(clip_uid, idx + j - offset + 1)]["verb"].append(vpd[j])
                            pred_action_aggregated["{}_{}".format(clip_uid, idx + j - offset + 1)]["noun"].append(npd[j])

                for key in pred_action_aggregated:
                    get_most_common = lambda x : max(set(x), key=x.count)
                    pred_action_aggregated[key]["verb_agg"] = get_most_common(pred_action_aggregated[key]["verb"])
                    pred_action_aggregated[key]["noun_agg"] = get_most_common(pred_action_aggregated[key]["noun"])

            tr_prompt += ", ".join(["({}, {})".format(
                tax["verbs"][pred_action_aggregated[clip_uid + "_" + str(idx)]["verb_agg"]].split("_(")[0].replace('_',' '), 
                tax["nouns"][pred_action_aggregated[clip_uid + "_" + str(idx)]["noun_agg"]].split("_(")[0].replace('_',' ')
            ) for idx in range(annotations[clip_uid][i]["action_idx"], (annotations[clip_uid][i]["action_idx"] + nprev))
            ]) + ", "

        else:
            tr_prompt += ", ".join(["({}, {})".format(x["verb"].split("_(")[0].replace('_',' '), x["noun"].split("_(")[0].replace('_',' ')) for x in annotations[clip_uid][i:(i + nprev)]]) + ", "
    if not is_test:
        tr_prompt += ", ".join(["({}, {})".format(x["verb"].split("_(")[0].replace('_',' '), x["noun"].split("_(")[0].replace('_',' ')) for x in annotations[clip_uid][(i + nprev):(i + 20 + nprev)]])
        tr_prompt += ".\n"
    return tr_prompt

def update_captions(origin, file_name, non_exist_ok=False):
    if not os.path.exists(file_name) and non_exist_ok:
        print("WARNING: file not exist", file_name)
        return origin 
    with open(file_name, "r") as f:
        raw = json.load(f)
        for k in raw:
            if len(raw[k]) == 0:
                continue 
            if type(raw[k][0]) == list:
                raw[k] = sum(raw[k], [])
            elif type(raw[k]) == str:
                raw[k] = [raw[k]]
            origin[k] += raw[k]

    return origin 
