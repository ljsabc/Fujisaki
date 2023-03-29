# prompt_openai.py
# parsing with openai-guided prompts
from tkinter import E
import config
import random
import json
from prompt_util import checkResponse, original_post_prompt, findTopic
import openai
import os
from concurrent import futures
import collections

def process_single_original_qa(value):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                #{"role": "system", "content": "你是一个富有情感的善于分析他人发言的发帖助手。\
                #                                请提出一个问题，使用户输入的内容可以恰当回复你提出的问题。问题中禁止包含“这”这个字。\
                #                                如果难以提问，或者提出的问题更像是在追问用户的输入而不是让用户的输入解答问题，就请提出一个诸如“最近发生什么事？”之类的通用问题。"},
                
                {"role": "system", "content": "Based on the Chinese user input, create a question in SIMPLIFIED CHINESE that allows the user's input to serve as an appropriate response to your question.\
                                                If it's too difficult to come up with a question, or the user's input is too ambiguous, or you have to question base on the details of the user's input, \
                                                please give up and just output a general Chinese question which has the similar meaning of, '最近发生什么事了？'."},

                {"role": "user", "content": f"User input (in Chinese): {value}"},
            ],
        max_tokens=128,
        temperature=0.95
        )
    return response.choices[0]["message"]["content"].strip()

def process_single_original_continue(value):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                #{"role": "system", "content": "你是一个富有情感的善于分析他人发言的发帖助手。\
                #                               请试着在用户输入的内容前面补充一个不是疑问句的简短上文，使得用户的输入可以恰当地衔接你的上文。\
                #                               你的上文可以是用户输入之前发生的事情，也可以是一个短暂的前情提要，但不允许总结或重复用户输入。\
                #                               注意：是让用户输入跟随你的上文，而不是你的上文去跟随用户的输入。你的回答只需要包含上文。"},

                {"role": "system", "content": "Please create a EXTREMELY brief context IN SIMPLIFIED CHINESE that precedes the Chinese user input, \
                                                imagining a scenario where the user's input can naturally be right AFTER your context to form a complete story. \
                                                Your context should not cover the details of the user input, but rather set the stage for it. Note: \
                                                Your context should not contain the texts that are already in the user input. Your should output the Chinese context only, without any modifier."},
                {"role": "user", "content": f"User input (in Chinese): {value}"},
            ],
        max_tokens=128,
        temperature=0.95
        )
    return response.choices[0]["message"]["content"].strip()

def openai_process_original(item):
    tweet = item["text"]
    id = item["id"]
    res = []
    loadedCount = 0

    # check cache first
    loadedCount = 0
    if os.path.exists(f'openai_cache/{id}.txt'):
        try:
            with open(f'openai_cache/{id}.txt', 'r') as f:
                res = json.load(f)
                if len(res) >= config.OPENAI_MAX_SAMPLE:
                    # possibility that the cache is even larger than the max sample
                    return tweet, res
                else:
                    loadedCount = len(res)
        except Exception as e:
            print(e)
            pass

    if config.OPENAI_CACHE_ONLY:
        return None

    # now we need to invoke openai to generate the rest of the samples
    # sample several seed questions for each tweet
    for j in range(loadedCount, config.OPENAI_MAX_SAMPLE):
        # 75% Q&A, 25% completion
        if random.random() < 0.75:
            try:
                res.append(process_single_original_qa(tweet))
            except Exception as e:
                print(e)
                # broken
                return None
            
        else:
            try:
                res.append(process_single_original_continue(tweet))
            except Exception as e:
                print(e)
                # broken
                return None
            
    # save to cache
    with open(f'openai_cache/{id}.txt', 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    return tweet, res



def processOriginalTweet_openai(tweets, iteration_count):
    # the iteration count matters, as we will use it as the index to sample cached openai prompts
    openai.api_key = config.OPENAI_KEY

    # make the openai cache directory
    if not os.path.exists('openai_cache'):
        os.makedirs('openai_cache')

    # because it's sampling from openai, we do not need to do Q/A or completion
    # we better just do random sampling with a seed question for a small portion of samples

    final = []
    sample_threshold = 0.10 if iteration_count == 0 else 0      # sample 10% of the tweets

    for item in tweets:
    # sample a random float from 0-1 to decide the ways of generation
    # sample_range is a probablity accumulative list
    # [random post, completion, Q&A, rest (direct original post)]
        tweet = item["text"]
        rr = random.random()
        if rr < sample_threshold:
            # sample a random question, and concatenate
            instruction = f"{random.choice(original_post_prompt)}"
            user_input = f""
            if checkResponse(tweet):
                final.append({"instruction": instruction, "input": user_input, "output": tweet})
        elif rr < sample_threshold * 2:
            # no instructions, unconditional generation.
            final.append({"instruction": "", "input": "", "output": tweet})
        else:
            # do nothing, as we have made the sampling successfully
            pass

    # now proceed to the openai generation
    openai_process_list = []
    for item in tweets:
        tweet = item["text"]
        if checkResponse(tweet):
            openai_process_list.append(item)

    # now invoke a threadpool to process the openai generation
    with futures.ThreadPoolExecutor(max_workers=config.OPENAI_THREADS) as executor:
        results = executor.map(openai_process_original, openai_process_list)

    for r in results:
        if r:
            tweet, response = r
            if iteration_count < len(response):
                final.append({"instruction": response[iteration_count], "input": "", "output": tweet})
            else:
                # we do not have enough cached openai prompts
                pass

    return final