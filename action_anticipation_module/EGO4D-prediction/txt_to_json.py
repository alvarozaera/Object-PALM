import json 
import collections


logfile = "llama7B_run.txt"

final_dict = collections.defaultdict(dict)
with open("./llm-log/" + logfile, "r") as f:
    for l in f:
        split_line = l.split(" ")
        clip_id = split_line[0]
        n_or_v = split_line[1]
        preds = [int(ele.strip()) for ele in split_line[2:]]

        if n_or_v in final_dict[clip_id].keys():
            final_dict[clip_id][n_or_v].append(preds)
        else:
            final_dict[clip_id][n_or_v] = [preds]


with open("./llm-log/" + logfile.replace('.txt', '.json'), 'w') as f:
    json.dump(final_dict, f)
