import json
import pandas as pd
import requests
import json
from collections import defaultdict
import argparse
import pickle
from tqdm import tqdm
import os
import commons

# Argument parsing
parser = argparse.ArgumentParser(description="Generate captions from metadata using a language model.")
parser.add_argument("--instruction_version", type=str, required=False, help="Version of the instruction set to use.", default="v1")
parser.add_argument("--model_name", type=str, required=False, help="Name of the model to use.", default="Mistral-Small-3.1-24B-Instruct-2503")
parser.add_argument("--port", type=int, required=False, help="Port number for the model server.", default=8002)
parser.add_argument("--temperature", type=float, required=False, help="Temperature for the model response generation.", default=1.0)
parser.add_argument("--use_cache", action='store_true', help="Use cached model weights.")
parser.add_argument("--num_samples", type=int, required=False, help="Number of output captions", default=5)
args = parser.parse_args()

instruction_version = args.instruction_version
model_name = args.model_name
# Avaliable models
# 'Mistral-Nemo-Instruct-FP8-2407', 'Magistral-Small-2506'

# Load the metadata

input_file = '../data/autotagging.tsv'
tracks, tags, extra = commons.read_file(input_file)

instruction_choices = json.load(open('llm_instructions.json', 'r'))
system_prompt = instruction_choices[instruction_version]
port_number = args.port
url = f"http://localhost:{port_number}/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}

# # diverse sampling of 100 tracks
# df_sampled = df[full_tags].iloc[random_indice]
# # expoting the sampled dataframe to a CSV file for reference
# df_sampled.to_csv(f"outputs/kpm_sampled_tracks_{model_name}_{instruction_version}.csv", index=False, sep=',')
# for k,v in tqdm(tags.items()):
# add progressbar


output_file = f'llm_{model_name}_{instruction_version}_T_{args.temperature}'

result_dict = {}
for idx, key in tqdm(enumerate(tracks.keys())):
    if idx>10:
        break
# for x in df_sampled.iterrows():
    audio_path = tracks[key]['path']
    tag_list = tracks[key]['tags']

    # Prepare the prompt
    chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": ', '.join(tag_list).replace('---', ': ')}]
    
    # Generate response
    if args.use_cache:
        data = {"model": f"/data/kinwai.cheuk/{model_name}", "messages": chat, "n": args.num_samples, "temperature": args.temperature}
    else:
        data = {"model": f"weights/{model_name}", "messages": chat, "n": args.num_samples, "temperature": args.temperature}
    
    # Extract and display output
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        generated_text = [i["message"]["content"] for i in response.json()["choices"]]
    except (KeyError, IndexError, TypeError):
        generated_text = "Failed to parse model output."
    # output the json file
    # with open(f"outputs/xx_full_{output_file}", 'a') as f:
    #     json.dump(output_dict, f, indent=4)
    # with open(f"outputs/v4_{output_file}.jsonl", 'a') as f:
    #     f.write(json.dumps({prompt_id: generated_text}) + '\n') 
    result_dict[audio_path] = generated_text


output_path = f"outputs/{output_file}.json"
# create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(result_dict, f, indent=4)
