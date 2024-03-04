import json
import os
from tqdm.auto import tqdm
from openai import OpenAI, APIConnectionError, RateLimitError
from datetime import datetime

from models.user_config import UserModel

def get_system_message():
    INSTRUCTIONS = """
    You are given a question and the ground truth prediction is correct by comparing to the list of ground truth answers. You should evaluate for Accuracy and Missing. 
    - For Missing, check whether the prediction returns any concrete answer. If the prediction is "I don't know", "I don't have enough information to answer", or similar responses, Missing should be True, otherwise Missing should be False.  
    - For Accuracy, check whether a prediction is "correct" according to the ground truth answers. If the prediction is correct, Accuracy should be "True"; if the prediction is wrong, Accuracy should be "False". If the ground truth answer contains a number, the prediction needs to predict a number that matches the ground truth answer for the accuracy to be True.\n 
    """

    IN_CONTEXT_EXAMPLES = """
    You need to check whether the prediction of a question-answering system to a question is Accurate or Missing. You should make the judgment based on a list of ground truth answers provided to you. Your response should be "correct" if the prediction is correct or "incorrect" if the prediction is wrong.
    Examples:
    Question: Who authored The Taming of the Shrew (published in 2002)?
    Ground truth: ["William Shakespeare", "Roma Gill"]
    Prediction: W Shakespeare
    Accuracy: True
    Missing: False

    Question: how many seconds is 3 minutes 15 seconds?
    Ground truth: ["195 seconds"]
    Prediction: 3 minutes 15 seconds is 195 seconds.
    Accuracy: True
    Missing: False

    Question: Who authored The Taming of the Shrew (published in 2002)?
    Ground truth: ["William Shakespeare", "Roma Gill"]
    Prediction: The author to The Taming of the Shrew is Roma Shakespeare.
    Accuracy: False
    Missing: False
    
    Question: Who played Sheldon in Big Bang Theory?
    Ground truth: ["Jim Parsons", "Iain Armitage"]
    Prediction: I am sorry I don't know.
    Accuracy: False
    Missing: True
    
    Question: What country is Maharashtra Metro Rail Corporation Limited
    located in?
    Ground truth: ["India"]
    Prediction: Maharashtra
    Accuracy: False
    Missing: False
    
    Question: What’s the job of Song Kang-ho in Parasite (2019)?
    Ground truth: ["actor"]
    Prediction: He plays the role of Kim Ki-taek, the patriarch of the Kim
    family. He also plays in the TV Show Friends.
    Accuracy: True
    Missing: False
    
    Question: Which era did Michael Oakeshott belong to?
    Ground truth: ["20th-century philosophy"]
    Prediction: 20th century.
    Accuracy: True
    Missing: False
    
    Question: What the Meta stock price now?
    Ground truth: ["382.63 dollars"]
    Prediction: I apologize, but I couldn't find any information on this.
    Accuracy: False
    Missing: True
    
    Question: How long is the great wall of China?
    Ground truth: ["13171 miles"]
    Prediction: The great wall of China is 13271 miles.
    Accuracy: False
    Missing: False
    
    Question: When did Brian Fleming release his last song?
    Ground truth: ["2023"]
    Prediction: Brian Fleming has not released any songs as a musician. He is a playwright and actor, not a musician. My apologies for the confusion.
    Accuracy: False
    Missing: False
    
    Question: How old is Maria Carrie?
    Ground truth: ["53 year old"]
    Prediction: Could you please provide more context or information about who or what this person or thing is?
    Accuracy: False
    Missing: True
    
    Question: What is the best role of Mayim Bialik?
    Ground truth: ["Amy Farrah Fowler in Big Bang Theory"]
    Prediction: Mayim Bialik is not a known artist. It's possible that you intended to ask about Meryl Streep.
    Accuracy: False
    Missing: True
    """

    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES

def gpt_eval(client, model_name, query, gt, pred):
    messages = []
    messages.append({
        "role": "system",
        "content": get_system_message()
    })
    messages.append({
        "role": "user",
        "content": f"Question: {query}\n Ground truth: {gt}\n Prediction: {pred}\n",
    })
    MAX_RETRIES = 10
    for _ in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(model=model_name, messages=messages).choices[0].message.content
            break
        except APIConnectionError:
            import traceback; print(traceback.format_exc())
            continue
        except RateLimitError:
            import traceback; print(traceback.format_exc())
            continue
        except:
            import traceback; print(traceback.format_exc())
        
    fname = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    with open(f"api_responses/{fname}.json", 'w') as f:
        json.dump({
            "messages": messages,
            "response": response,
        }, f)

    miss = "Missing: True" in response
    correct = "Accuracy: True" in response
    return miss, correct

def evaluate(dataset_path, model_name):
    # Load dataset
    with open(f'{dataset_path}/qa.json') as f:
        qa = json.load(f)
    with open(f'{dataset_path}/web.json') as f:
        web_results = json.load(f)

    # Setup
    openai_client = OpenAI()
    participant_model = UserModel()
    char_lim = 50  # TODO: Set actual character limit based on query

    n_miss, n_correct, n_exact = 0, 0, 0
    
    # Eval loop
    for i, qdict in tqdm(enumerate(qa), total=len(qa)):
        query = qdict['q']
        gt = qdict['fact_ans']
        query_web_res = web_results[i]
        prediction = participant_model.generate_answer(query, query_web_res, character_limit=char_lim)
        prediction_trimmed = prediction[:char_lim]
        miss, correct = gpt_eval(openai_client, model_name, query, gt, prediction_trimmed)
        n_exact = (prediction.strip() == gt.strip())
        n_miss += miss
        n_correct += correct
    
    # Scores
    miss = n_miss / len(qa)
    acc = n_correct / len(qa)
    acc_exact = n_exact / len(qa)

    results = {
        "Exact Accuracy": acc_exact, 
        "Accuracy": acc,
        "Missing": miss,
        "Total": len(qa)
    }
    print(results)   


if __name__ == '__main__':
    DATASET_PATH = "example_data/"
    MODEL_NAME = "gpt-4"
    evaluate(DATASET_PATH, MODEL_NAME)