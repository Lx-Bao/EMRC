import csv
import json
import argparse
import os
import torch
import random
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset

import json
import random
import time
import asyncio
# import camel


max_model_length = 4096
max_new_tokens = 2048

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.messages import BaseMessage

# from openai import OpenAI

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

model_openthinker_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="openthinker:32b",
    url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 20480,
                          "stream":True},
)

model_Qwen25_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen2.5:32b",
    url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048,
                          "stream":True},
)

model_qwq_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwq",
    url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240,
                          "stream":True},
)

model_phi4_14b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="phi4",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048,
                          "stream":True},
)

model_Qwen25_14b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen2.5:14b",
    url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048,
                          "stream":True},
)

model_Huatuo_8b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="ZimaBlueAI/HuatuoGPT-o1-8B",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 4096,
                          "stream":True},
)


model_med42_8b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="thewindmom/llama3-med42-8b",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048,
                          "stream":True},
)

model_yi_34b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="yi:34b",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048,
                          "stream":True},
)

model_mistral_24b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="mistral-small",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048,
                          "stream":True},
)

model_gemma_27b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="gemma3:27b",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048,
                          "stream":True},
)

model_Deepseekr1_14b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="deepseek-r1:14b",
    url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240,
                          "stream":True},
)

model_Deepseekr1_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="deepseek-r1:32b",
    url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240,
                          "stream":True},
)


model_qwen_7b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen2.5",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048,
                          "stream":True},
)

model_exaone_7b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="exaone3.5",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048,
                          "stream":True},
)

model_mistral_7b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="mistral",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048,
                          "stream":True},
)

model_deepseek_7b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="deepseek-r1",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240,
                          "stream":True},
)

model_deepseek_8b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="deepseek-r1:8b",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240,
                          "stream":True},
)

model_llama_8b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="llama3.1",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240,
                          "stream":True},
)

model_phi4_3b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="phi4-mini",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 4096,
                          "stream":True},
)

model_glm4_9b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="glm4",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240,
                          "stream":True},
)

model_gemma3_12b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="gemma3:12b",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240,
                          "stream":True},
)



model_exaone_deep_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="exaone-deep:32b",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240,
                          "stream":True},
)

model_mistral_small31_24b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="mistral-small3.1",
    #url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 10240,
                          "stream":True},
)




model_registry = {
    "model_Deepseekr1_32b": model_Deepseekr1_32b,
    "model_Deepseekr1_14b": model_Deepseekr1_14b,
    "model_gemma3_27b": model_gemma_27b,
    "model_mistral_24b": model_mistral_24b,
    "model_yi_34b": model_yi_34b,
    "model_med42_8b": model_med42_8b,
    "model_Huatuo_8b": model_Huatuo_8b,
    "model_Qwen25_14b": model_Qwen25_14b,
    "model_phi4_14b": model_phi4_14b,
    "model_qwq_32b": model_qwq_32b,
    "model_phi4_3b": model_phi4_3b,
    "model_mistral_7b": model_mistral_7b,
    "model_exaone_7b": model_exaone_7b,
    "model_deepseek_8b": model_deepseek_8b,
    "model_qwen_7b": model_qwen_7b,
    "model_gemma3_12b": model_gemma3_12b,
    "model_llama_8b": model_llama_8b,
    "model_QWen_32b": model_Qwen25_32b,
    "model_qwq_32b": model_qwq_32b,
    "model_DS_32b": model_Deepseekr1_32b,
    "model_opt_32b": model_openthinker_32b,
    "model_mistral_24b": model_mistral_24b,
    "model_Yi_34b": model_yi_34b,
}



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

import gc

async def run_llm( in_model, patient_response, prev_response=None, judge_promt = None, re_summer = None):
    # print("🪪 model_name:", in_model.model_type, file=file_out)
    print("🪪 model_name:", in_model.model_type)
    for sleep_time in [1, 2, 4]:
        try:
            if judge_promt:
                assistant_sys_msg =getFinalSystemPrompt(judge_promt, prev_response)
                # assistant_sys_msg = assistant_sys_msg + judge_Prompt_example
                Doctor = DoctorTalker(model=in_model, sys_msg= assistant_sys_msg)
                result = Doctor.inference_doctor(question=judg_question)
            elif prev_response:
                assistant_sys_msg =getFinalSystemPrompt(MOA_sys_msg, prev_response)
                Doctor = DoctorTalker(model=in_model, sys_msg=assistant_sys_msg)
                result = Doctor.inference_doctor(question=patient_response)
            elif re_summer:
                assistant_sys_msg =getFinalSystemPrompt(MOA_sys_msg2, prev_response)
                Doctor = DoctorTalker(model=in_model, sys_msg=assistant_sys_msg)
                result = Doctor.inference_doctor(question=patient_response)
            else:
                Doctor = DoctorTalker(model=in_model)
                result = Doctor.inference_doctor(question=patient_response)
            return result
        except Exception as e:

            error_str = str(e)

            print(e)
            # await asyncio.sleep(sleep_time)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if "CUDA error: out of memory" in error_str or "out of memory" in error_str:
                wait_time = 2 * (2 ** sleep_time)  
                await asyncio.sleep(wait_time)
            else:
                wait_time = 2 * (sleep_time + 1) 
                await asyncio.sleep(wait_time)
    print(f"All attempts failed for model {in_model.model_type}, returning empty result")
    return ""




MOA_sys_msg = """You have been provided with a set of responses and their summary from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response.  It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. You need to control the length of your output so that your answers are short, condensed and precise. Don't output content irrelevant to the question. The final answer has only one choice and needs to be provided in JSON format.

Responses from models:"""

MOA_sys_msg2 = """You have been provided with the summary of a set of responses from various open-source models to the latest user query. Your task is to refer the summary and generate a single, high-quality response.  It is crucial to critically evaluate the information provided in the summary, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. You need to control the length of your output so that your answers are short, condensed and precise. Don't output content irrelevant to the question. The final answer has only one choice and needs to be provided in JSON format.

Responses from models:"""



judg_question = """Please complete your task."""

class_Prompt_1 = "The following are multiple choice questions about medical knowledge. Please answer which medical department the following question belongs to and provide the complexity (difficulty) of the question. The all answer needs to be provided in JSON format:"

Answer_Prompt_1 = "Please think step by step and give the correct letter choice of the question. You need to control the length of your output so that your answers are short, condensed and precise. The all answer needs to be provided in JSON format:"



Prompt_Dept = """
The department needs to select from the following list: 
1) Internal Medicine Dept.
2) Surgery Dept.
3) Obstetrics and Gynecology Dept.
4) Pediatrics Dept.
5) Neurology Dept.
6) Oncology Dept.
7) Otolaryngology Dept.
8) Psychiatry and Psychology Dept.
9) Emergency and Critical Care Dept. (Limited to acute conditions that cannot be categorized)
"""


Prompt_diff = """
Please indicate the difficulty/complexity of the medical query among below options:
1) low: a PCP or general physician can answer this simple medical knowledge checking question without relying heavily on consulting other specialists.
2) moderate: a PCP or general physician can answer this question in consultation with other specialist in a team.
3) high: Team of multi-departmental specialists can answer to the question which requires specialists consulting to another department (requires a lot of team effort to treat the case).
"""

class_Prompt_example = """
Aswer example:
{
"medical_department": "Surgery Dept.",
"complexity": "low"
}

{
"medical_department": "Emergency and Critical Care Dept.",
"complexity": "moderate"
}

{
"medical_department": "Psychiatry and Psychology Dept.",
"complexity": "high"
}
"""

Answer_Prompt_example = """
Aswer format：
{
"Analysis": "",
"Answer": "B"
}


{
"Analysis": "",
"Answer": "D"
}
"""

agst_prompt = """You have received a set of responses from different open source models to the same user query. Your task is to identify and point out any errors present in the responses.You need to control the length of your output so that your answers are short, condensed and precise.

Responses from models:"""


sum_prompt = """You have received a set of responses from different open source models to the same user query. Your task is to summarize these responses, extract important, correct, and helpful information. You need to control the length of your output so that your answers are short, condensed and precise. 

Responses from models:"""



def clean_json(json_string):
    # Ensure that the JSON string uses double quotes
    # Replace single quotes with double quotes for property names and values
    json_string = json_string.replace("'", '"')
    
    # Handle edge case where there might be trailing commas or other minor issues
    # This can be extended with more cleanup logic if needed
    return json_string.strip()

def extract_json_from_txt(txt_content):
    # Regex to match 'model_name: deepseek-r1:14b' followed by JSON data (with or without ```json)
    json_pattern = r'[{][\s\S]*?[}]'
    try:
        matches = re.findall(json_pattern, str(txt_content), re.DOTALL)
    except:
        matches = [{
  "medical_department": "Emergency and Critical Care Dept.",
  "complexity": "moderate",
}]

    # If no matches found, return an empty dictionary
    if not matches:
        return [{
  "medical_department": "Emergency and Critical Care Dept.",
  "complexity": "moderate",
}]  
    print(matches)
    
    # Parse each cleaned JSON string and return a list of JSON objects
    for match in matches:
        try:
            match = match.replace('\\n', '\n').replace('\\', '')
            json_objects = json.loads(match)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            json_objects= {"medical_department": "Emergency and Critical Care Dept.", "complexity": "moderate",}  # If JSON is invalid, append an empty dictionary
    print(json_objects)
    return json_objects



import pandas as pd

# Load the dataset from the CSV file
# file_path = 'answer_accuracy_by_department.csv'
# csv_data = pd.read_csv(file_path)

# Function to get the top 3 models based on weighted scores
def get_top_models_for_input(department, difficulty, data, department_weight=0.5, difficulty_weight=0.5):
    """
    This function calculates the top 3 models based on weighted scores of a given department and difficulty level.
    department_weight and difficulty_weight are set to 0.5 by default.
    
    Parameters:
        department (str): The name of the department to evaluate.
        difficulty (str): The difficulty level to evaluate (Low, Moderate, High).
        data (DataFrame): The DataFrame containing the model performance data.
        department_weight (float): The weight of the department's score (default is 0.5).
        difficulty_weight (float): The weight of the difficulty's score (default is 0.5).
    
    Returns:
        DataFrame: A DataFrame containing the top 3 models based on the weighted score.
    """
    # Ensure department and difficulty columns are valid
    if department not in data.columns or difficulty not in data.columns:
        department = "Emergency and Critical Care Dept."
        difficulty = "moderate"


    # Extract department and difficulty scores
    department_scores = data[department]
    difficulty_scores = data[difficulty]
    # Calculate weighted scores
    weighted_scores = department_weight * department_scores + difficulty_weight * difficulty_scores

    # Add weighted scores to the dataframe for sorting
    data['Weighted Score'] = weighted_scores

    # Sort models by weighted score and select the top 3
    top_models = data[['Unnamed: 0', 'Weighted Score']].sort_values(by='Weighted Score', ascending=False).head(4)

    model_names = top_models['Unnamed: 0'].tolist()
    scores = top_models['Weighted Score'].tolist()

    return model_names, scores



def remove_think_tags(string_list):
    # 定义正则表达式去匹配以<think>开头</think>结尾的内容，包括换行符
    pattern1 = r'<think>.*?</think>'
    pattern2 = r'## Thinking.*?## Final Response'
    pattern3 = r'<think>.*?</user>'
    cleaned_text = [re.sub(pattern1, '', s, flags=re.DOTALL) for s in string_list]
    cleaned_text = [re.sub(pattern2, '', s, flags=re.DOTALL) for s in cleaned_text]
    cleaned_text = [re.sub(pattern3, '', s, flags=re.DOTALL) for s in cleaned_text]
    # 使用正则表达式替换掉所有匹配的部分
    return cleaned_text


async def batch_inference(llm, inf_prompt, question):
    layers = 3
    reference_models = llm

    aggregator_model = llm[0] 
    Antagonist_model = llm[0]
    summarizer = llm[0]

    results = await asyncio.gather(*[run_llm(in_model = model, patient_response = inf_prompt) for model in reference_models])
    results = remove_think_tags(results)
    # results = '\n'.join(results) if isinstance(results, list) else results
    summarize = await asyncio.gather(*[run_llm(in_model = summarizer, patient_response = sum_prompt, prev_response = results, judge_promt=sum_prompt)])
    summarize = remove_think_tags(summarize)
    
    results= results+summarize
    
    for _ in range(1, layers-1):
        print("🪜 Layer:", _)
        print("🪜 Layer:", _)

        results = await asyncio.gather(*[run_llm(in_model = model, patient_response = inf_prompt, prev_response = results) for model in reference_models])
        results = remove_think_tags(results)
        summarize = await asyncio.gather(*[run_llm(in_model = summarizer, patient_response = sum_prompt, prev_response = results, judge_promt=sum_prompt)])
        summarize = remove_think_tags(summarize)
        results= results + summarize


    # aggregator_model
    print("🪜 aggregator", _)
    results = await asyncio.gather(*[run_llm(in_model = aggregator_model, patient_response = question, prev_response = results)])
    results = remove_think_tags(results)
    

    return results



with open("./NEJMQA-655multi_questions.json", 'r', encoding='utf-8') as file:
    data = json.load(file)


async def main():
    all_out_json = []
    with open("diffNEJM.txt", "w") as f:
        class_models = model_gemma_27b
        for i, item in enumerate(data):
            question = item["question"]
            # options = item["options"]
            class_prompt = class_Prompt_1 + str(question) +  Prompt_Dept + Prompt_diff + class_Prompt_example
            Answer_Prompt = Answer_Prompt_1 + str(question)  + Answer_Prompt_example
            Question_Prompt = "Question:\n" + str(question)  + Answer_Prompt_example
            outputs = await asyncio.gather(run_llm(in_model=class_models, patient_response=class_prompt))
            out_json = extract_json_from_txt(outputs)
            all_out_json.append(out_json)  

            f.write(str(out_json) + "\n")
    with open("all_out_json.json", "w", encoding="utf-8") as json_file:
        json.dump(all_out_json, json_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())


