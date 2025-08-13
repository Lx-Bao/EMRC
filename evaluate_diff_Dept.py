import csv
import json
import argparse
import os
import torch
import random
import transformers
import time
import re

from tqdm import tqdm
import logging
import sys

import json
import random
import time
import asyncio


choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 2048

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.messages import BaseMessage


model_openthinker_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="openthinker:32b",
    url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240},
)

model_Qwen25_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen2.5:32b",
    url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048},
)

model_qwq_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwq",
    url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 20480},
)

model_phi4_14b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="phi4",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048},
)

model_qwen_7b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen2.5",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048},
)

model_exaone_7b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="exaone3.5",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048},
)

model_mistral_7b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="mistral",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048},
)

model_deepseek_7b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="deepseek-r1",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240},
)

model_deepseek_8b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="deepseek-r1:8b",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240},
)

model_llama_8b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="llama3.1",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240},
)

model_phi4_3b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="phi4-mini",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 4096},
)

model_glm4_9b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="glm4",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240},
)

model_gemma3_12b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="gemma3:12b",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240},
)

model_Qwen25_14b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen2.5:14b",
    url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048},
)

model_gemma3_12b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="gemma3:12b",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240},
)

model_exaone_deep_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="exaone-deep:32b",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240},
)

class DoctorTalker(ChatAgent):
    def __init__(self, 
        scenario=None, 
        max_infs=None,
        model = None,
        sys_msg = None,
        memory = None,
        message_window_size = None,
        system_message = None
    ):
        self.infs = 0
        self.MAX_INFS = max_infs
        self.scenario = scenario



        super().__init__(
            system_message=sys_msg,
            model=model,
            memory=memory,
            message_window_size=message_window_size,

        )

    def inference_doctor(self, question) -> str:
        q_prompt = question
        answer = self.step(q_prompt).msgs[0].content
        self.infs += 1
        return answer
    

def getFinalSystemPrompt(system_prompt, results):
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )


def run_llm(in_model, patient_response, prev_response=None):
    print("🪪 model_name:", in_model.model_type)
    for sleep_time in [1, 2, 4]:
        try:
            Doctor = DoctorTalker(model=in_model)
            result = Doctor.inference_doctor(question=patient_response)
            print("Final answer:", result)
            return result
        except Exception as e:
            print(e)
            time.sleep(sleep_time)  
            return ""  




Prompt_1 = "Please answer which medical department the following question belongs to and provide the complexity (difficulty) of the question. Then, please think step by step and finally give the correct letter choice of the question. The all answer needs to be provided in JSON format:"

Prompt_Dept = """
The department needs to select from the following list: 
1) Internal Medicine Dept.
2) Surgery Dept.
3) Obstetrics and Gynecology Dept.
4) Pediatrics Dept.
5) Neurology Dept.
6) Oncology Dept.
7) Otolaryngology Dept.
8) Emergency and Critical Care Dept. (Limited to acute conditions that cannot be categorized)
9) Psychiatry and Psychology Dept.
"""


Prompt_diff = """
Please indicate the difficulty/complexity of the medical query among below options:
1) low: a PCP or general physician can answer this simple medical knowledge checking question without relying heavily on consulting other specialists.
2) moderate: a PCP or general physician can answer this question in consultation with other specialist in a team.
3) high: Team of multi-departmental specialists can answer to the question which requires specialists consulting to another department (requires a lot of team effort to treat the case).
"""

Prompt_example = """
Aswer example：
{
"Analysis": ""
"medical_department": "Surgery Dept.",
"complexity": "moderate",
"Answer": "B"
}

{
"Analysis": ""
"medical_department": "Emergency and Critical Care Dept.",
"complexity": "low",
"Answer": "D"
}
"""

def remove_think_tags(string_list):
    pattern1 = r'<think>.*?</think>'
    pattern2 = r'## Thinking.*?## Final Response'
    cleaned_text = [re.sub(pattern1, '', s, flags=re.DOTALL) for s in string_list]
    cleaned_text = [re.sub(pattern2, '', s, flags=re.DOTALL) for s in cleaned_text]
    return cleaned_text

with open("./dev.jsonl", "r") as file:
    data = [json.loads(line.strip()) for line in file]

with open("Diff_Dept.txt", "w") as f:
    models = [model_exaone_deep_32b]
    for model in models:
        for item in data:
            question = item["question"]
            options = item["options"]
            Prompt = Prompt_1 + str(question) + str(options) + Prompt_Dept + Prompt_diff + Prompt_example
            outputs = run_llm(in_model=model, patient_response=Prompt)
            f.write(outputs + "\n")
        print("New model:::::")
        f.write("New model:::::" + "\n")


