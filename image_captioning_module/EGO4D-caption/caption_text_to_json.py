import json


logfile = f"./captions/caption_file.txt"

with open(logfile, "r") as f:
	json_caption = {}
	for l in f:
		key = l.split(" ")[0]
		list_str = l.replace(key+" ", "")

		res = list_str.replace('\n','').replace("'","").strip('][').split(', ')

		json_caption[key] = [ele.replace('a person is ', '').replace('A person is ', '') for ele in res]

with open(logfile.replace(".txt", ".json"), "w") as outfile:
	json.dump(json_caption, outfile)
