import json
import random
import re

import numpy as np
import collections

from scrape_twitter import process_tweet_ids
import config

from prompt_openai import processOriginalTweet_openai

from prompt_util import original_post_prompt, findTopic, cut_sent, checkResponse


def processOriginalTweet(tweets):

    # It is about the original tweet
    if config.PARSE_REPLIES:
        # if we parse the replies, we will have more data to sample from
        # we do not need to do the completion, and the Q&A part can be inferred from the in-reply-to
        # of the original posts, 40% are unconditional (with questions), 10% are completion, 20% are Q&A, 30% are unconditional (with no prompts)
        sample_range = [0.4, 0.5, 0.7, 1]
    else:
        sample_range = [0.35, 0.5, 0.95, 1]


    final = []
    for item in tweets:
    # sample a random float from 0-1 to decide the ways of generation
    # sample_range is a probablity accumulative list
    # [random post, completion, Q&A, rest (direct original post)]
        tweet = item["text"]
        rr = random.random()
        if rr < sample_range[0]:
            # sample a random question, and concatenate
            instruction = f"{random.choice(original_post_prompt)}"
            user_input = f""
            if checkResponse(tweet):
                final.append({"instruction": instruction, "input": user_input, "output": tweet})
        elif rr < sample_range[1]:
            # given a truncated tweet, ask for completion
            substring = cut_sent(tweet)
            if len(substring) > 1:
                user_input = f""
                rr = random.randint(1, len(substring)-1)
                instruction = "".join(substring[0:rr])
                if checkResponse(instruction):
                    final.append({"instruction": instruction, "input": user_input, "output": "".join(substring[rr:])})
            else:
                instruction = f"{random.choice(original_post_prompt)}"
                user_input = f""
                final.append({"instruction": instruction, "input": user_input, "output": tweet})
        elif rr < sample_range[2]:
            # QA like 
            # ask for a topic, the topic is mainly based on a substring of this tweet
            instruction = findTopic(tweet)
            if instruction is not None:
                user_input = f""
                if checkResponse(tweet):
                    final.append({"instruction": instruction, "input": user_input, "output": tweet})
            else:
                #if cannot find a topic
                instruction = f"{random.choice(original_post_prompt)}"
                user_input = f""
                if checkResponse(tweet):
                    final.append({"instruction": instruction, "input": user_input, "output": tweet})

        else:
            # no instructions, unconditional generation.
            final.append({"instruction": "", "input": "", "output": tweet})

    return final

def processReplyTweets(tweets):

    final = []
    context_count = []

    for index, t in enumerate(tweets):
        #print(index, t)
        tweet_id = t['id']
        tweet_text = t['text']
        context = t['context']

        # first, we need to check if the reply itself is interesting
        if not checkResponse(tweet_text):
            continue

        # then, we need to check if the context is blank
        if context is None:
            continue

        if len(context) == 0:
            continue
        
        # next, we do a rough check in if the context is interesting
        # if the context is too short, we will not use it
        if not checkResponse("".join(context)):
            continue

        # We believe the context is interesting
        # in this way, we want to sample a random context length based on the probability distribution of 1/x
        # the longer the context, the less likely it will be sampled
        # if the context's length is not long enough, we will sample again
        '''
        l = len(context)
        p = []
        for j in range(l):
            p.append(np.power(j+1, -config.REPLY_TEMP))

        # normalize the probability
        p = np.array([i/sum(p) for i in p], dtype=np.float32)

        while True:
            # sample a number based on p
            r = np.random.choice(l, 1, p=p)[0]
            context_text = "\n".join(context[-r:])
            # check if the context is interesting
            if checkResponse(context_text):
                r_count.append(r)
                break
        '''
        
        context_count.append(len(context))

        # now we have a context, and a reply
        # go give the prompt
        final.append({"instruction": "\n".join(context), "input": "", "output": tweet_text})

        # but we can do more, we can also augment a Q&A like discussion within the topic, if we want
        # give a small random chance to do this
        if random.random() < 0.05:
            instruction = findTopic(context[-1])
            if instruction is not None:
                final.append({"instruction": instruction, "input": "", "output": tweet_text})

    # TODO: any other sort of prompt engineering?
    # use a counter to see how many long contexts are used
    print(collections.Counter(context_count))
        
    return final

def write_json(md_path, final_md, lang):

    # construct a instruction dataset    
    final = []

    # construct a list of tweets to be downloaded to sample the contexts
    context_tweets = []

    # original tweets
    original_tweets = []

    # firstly classify the type of each tweet
    for id, md, in_reply_to, quote, retweet in final_md:

        # content filter goes here:
        if md.strip() == "(media)":
            continue

        if in_reply_to and quote:
            # todo: process replies and quotes
            pass
        elif in_reply_to:
            # save them into a list; we will download them later
            context_tweets.append({"id": id, "text": md})
        elif quote:
            # todo: process quotes
            pass
        elif retweet:
            # not my tweet, simply discard them
            pass
        else:
            # original tweets
            original_tweets.append({"id": id, "text": md})


    # process with the original tweets
    for l in range(config.AUGMENTATION_FACTOR_ORIGINAL):
        if config.ENABLE_OPENAI:
            res = processOriginalTweet_openai(original_tweets, l)
            print(f"Batch: {l+1}, {len(res)} original tweets are added with openai.")
            final.extend(res)
        else:
            res = processOriginalTweet(original_tweets)
            print(f"Batch: {l+1}, {len(res)} original tweets are added.")
            final.extend(res)

    
    # process with the replies
    if config.PARSE_REPLIES:
        # Now things get even more interesting, we will scrape the tweets from the context_tweet_ids
        parsed_tweets = process_tweet_ids(context_tweets)
        print(f"Processed {len(parsed_tweets)} tweets from the context tweets.")

        for l in range(config.AUGMENTATION_FACTOR_REPLIES):
            res = processReplyTweets(parsed_tweets)
            print(f"Batch: {l+1}, {len(res)} replies are added.")
            final.extend(res)

        

    with open(md_path, "w") as f:
        # shuffle the dataset
        random.shuffle(final)
        f.write(json.dumps(final, indent=4, ensure_ascii=False))
