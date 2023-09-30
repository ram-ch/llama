"""
Author:ramch
This script perform custom named entity recognition using few shots learning
converts a train csv in to prompts
populates the test csv with results
"""
# TODO: Use the NER train data and populate the NER test data

from typing import Optional
from llama import Llama


generator = Llama.build(
        ckpt_dir='models/llama-2-7b-chat',
        tokenizer_path='models/tokenizer.model',
        max_seq_len=512,
        max_batch_size=8,
    )

dialogs = [


    [
        {
            "role": "system", # setting the context for the moddel
            "content": """You are a smart and intelligent Named Entity Recognition (NER) system. I will provide you example sentences with label and entity you need to train yourself to extract the label and entity and provide the output as per the format in example sentence""",
        },
        # Below are some of the samples for few shot learning
        # 1
        {"role": "user", "content": """That goal above agent young well none. 00779 Nicole Mountain Apt. 347 North Veronicaborough, TN 58877 Only mean end morning property. Control wide raise community agency guess fear."""},
        {"role": "assistant", "content":"Address:00779 Nicole Mountain Apt. 347 North Veronicaborough, TN 58877"},
        # 2
        {"role": "user", "content":"Usually itself strong onto way own she. Tough manager trial list. Wear your front Suite 786 them."},
        {"role": "assistant", "content":"Address:Suite 786"},
        # 3
        {"role": "user", "content":"Teacher green level sport officer area. Structure 5779 Patterson Shore Suite 089 even nation side suggest store next threat."},
        # {"role": "assistant", "content":"5779 Patterson Shore Suite 089"},


    ],


]




results = generator.chat_completion(
    dialogs,  # type: ignore
    max_gen_len=None,
    temperature=0,
    top_p=1,
)

# print(results)

for dialog, result in zip(dialogs, results):
    for msg in dialog:
        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    print(
        f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    )
    print("\n==================================\n")
