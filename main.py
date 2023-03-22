import openai
import pickle
import replicate
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np


app = FastAPI()


@app.post("/answer_question/")
async def answer_question_endpoint(question: str = Form(...), user_id: str = Form(...), image: UploadFile = File(...)):
    image_path = f"/storage/{user_id}.jpg"
    image_data = np.frombuffer(await image.read(), np.uint8)
    image_np = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    cv2.imwrite(image_path, image_np)
    answer_str = answer_question(question, image_path, user_id)
    return {"answer": answer_str}


def get_chat_log_path(user_id):
    return f"chat_logs/{user_id}_chat_log.pkl"


def image_related(chat_log_inline):
    """
    Determines if the question in chat log is related to the image
    Args:
        chat_log_inline (str): The chat-log in a single string.
    Returns:
        bool: True if the question is related to the image, False otherwise.
    """
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=20,
        messages=[
            {"role": "system", "content": "You are an AI assistant"},
            {"role": "user", "content": f"An AI assistant is designed to assist blind people by answering questions related to photos taken by the user. It utilizes the BLIP question-answering model to answer some of the questions. The following dialogue was provided:\n\n{chat_log_inline}\nDetermine whether the AI assistant requires visual context or information about the user's surroundings to reply to the user's last message. If the AI assistant requires any of these, reply with 'YES'. If none of these are required, reply with 'NO'. Please note that you can only write 'YES' or 'NO'. Begin."},
        ],
        temperature=0.3,
    )["choices"][0]["message"]["content"]

    return True if result.startswith("YES") else False


def correct_answer(question, blip_answer, chat_log_inline):
    """
    Corrects and returns the answer provided by BLIP
    Args:
        question (str): The question to be answered.
        blip_answer (str): The answer provided by the BLIP model.
        chat_log_inline (str): The chat-log in a single string.
    Returns:
        answer (str): The answer to the question.
    """
    QnA = f"User: {question} BLIP: {blip_answer}"
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=64,
        messages=[
            {"role": "system", "content": "You are an AI assistant"},
            {"role": "user", "content": f"We want to create an AI assistant to assist people with visual impairments by answering questions related to photos taken by the user. To accomplish this, we use the BLIP question-answering model. However, the model's responses are sometimes very brief, strange, and unpredictable. Your task is to write a proper reply to a user's prompt (User) based on BLIP's reply (BLIP) and nothing more. If you cannot provide a proper reply for any reason, just say 'I'm sorry. I cannot answer.' and nothing else. Do not mention the BLIP in your response. See the whole dialogue for reference:\n\n{chat_log_inline}\nBegin.\n\n" + QnA},
        ],
        temperature=0.5,
    )["choices"][0]["message"]["content"]

    return answer


def answer_question(question, image_path, user_id):
    """
    Answers the question using the BLIP model if the
    question is related to the image, otherwise uses GPT-3
    Args:
        question (str): The question to be answered.
        image_path (str): The path to the image.
        user_id (str): The user's ID.
    Returns:
        answer (str): The answer to the question.
    """
    chat_log_path = get_chat_log_path(user_id)
    try: # load previous chat-log from pickle file if it exists
        with open(chat_log_path, "rb") as f:
            chat_log = pickle.load(f)
    except: # otherwise, create a new chat-log
        chat_log = []

    # append the question to the chat-log
    chat_log_inline = " ".join([f"User: {U}\nAssistant: {A}\n" for U, A in chat_log])
    chat_log_inline += f"User: {question}\n"

    if (image_related(question)): # use the BLIP model to answer the question
        model = replicate.models.get("salesforce/blip-2")
        version = model.versions.get("4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608")
        inputs = {
            'image': open(image_path, "rb"),
            'caption': False,
            'question': question,
            'use_nucleus_sampling': False,
            'temperature': 0.7,
        }
        blip_answer = version.predict(**inputs)
        answer = correct_answer(question, blip_answer, chat_log_inline)

    else:
        answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=64,
        messages=[
            {"role": "system", "content": "You are an AI assistant who helps people with visual impairments."},
            {"role": "user", "content": f"You are Shigeo, an AI assistant designed to assist people with visual impairments. At times, you may require visual context, access to the user's camera, or information about the surroundings to answer some of the questions. Given the following dialogue with the user,\n\n{chat_log_inline}\nplease write reply for the last message. Write only the reply without quotation marks and nothing else. Begin."},
        ],
    )["choices"][0]["message"]["content"]

    # append the answer to the chat-log
    chat_log.append((question, answer))

    # if the chat-log is too long, remove the oldest question
    if len(chat_log) > 10:
        chat_log.pop(0)

    # save the chat-log to a pickle file
    with open(chat_log_path, "wb") as f:
        pickle.dump(chat_log, f)

    return answer

if __name__ == "__main__":
    while True:
        q = input("User: ")
        answer = answer_question(q, "./image.jpg", "assel")
        print(answer)