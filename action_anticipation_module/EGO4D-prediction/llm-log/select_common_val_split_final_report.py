import json

output_intention = "llama8B_intention_final_report.txt"
output_intention_summary = "llama8B_intention_summary_final_report.txt"

intention_action_ids = set()
with open(output_intention, "r") as f:
    intention_lines = f.readlines()
    for line in intention_lines:
        intention_action_ids.add(line.split()[0])

intention_summary_action_ids = set()
with open(output_intention_summary, "r") as f:
    intention_summary_lines = f.readlines()
    for line in intention_summary_lines:
        intention_summary_action_ids.add(line.split()[0])

common_action_ids = intention_action_ids.intersection(intention_summary_action_ids)
print(f"Number of common action ids: {len(common_action_ids)}")

with open("llama8B_intention_final_report_common.json", "w") as f:
    json.dump(list(common_action_ids), f)