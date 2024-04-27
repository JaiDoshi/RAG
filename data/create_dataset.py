
import json
import bz2

domains = ['sports', 'open', 'music', 'movie', 'finance']
question_types = ['simple_w_condition', 'set', 'simple', 'aggregation', 'comparison', 'post-processing', 'multi-hop', 'false_premise']

dict = {}

final_json = []

for domain in domains:
    for question_type in question_types:
        dict[(domain, question_type)] = 0

           
with bz2.open('crag_task_1_v2.jsonl.bz2', "rt") as bz2_file:
    for line in bz2_file:
        if not dict:
            break
        data = json.loads(line)
        domain = data["domain"]
        question_type = data["question_type"]

        if (domain, question_type) not in dict:
            continue

        final_json.append(json.dumps(data) + '\n')
        dict[(domain, question_type)] += 1
        if dict[(domain, question_type)] == 2:
            del dict[(domain, question_type)]

with open('processed_data.jsonl', 'w') as f:
    f.writelines(final_json)
