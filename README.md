# Fujisaki
An ongoing (fast prototyping) project to create your own doppelgänger based on your Twitter archive and LoRA+Alpaca

致力于创造一个属于你的不二咲千寻。项目处于初期阶段。

## What is this?

Inspired by Fujisaki Chihiro, tbf. I thought it would be a fun project, as I really cannot predict my life in the future, and I would like to see how my doppelgänger would react to the world. On the other hand, thank to the super-strong LLM and LoRA, which makes finetuning of LLM with small dataset possible.

## Sample outputs:

    # results after a 2-epoch training, with ~50k tweets, including replies
    # to be improved by hyperparameter tuning, more training data, and more epochs
    # also, the model is not finetuned with replies, so the replies are random

    Chihiro: 呜呜呜~
    Chihiro: 摸摸摸摸…RT @sodawon: @ljsabc 感觉这东西许多人都不知道，很不爽……
    Chihiro: 别去啊，你就不行了
    Chihiro: 早安~RT @alexandergxm: 早安～
    Chihiro: RT @xiaoyu520: 咱见一个哪儿估计可以买巨蛋甩的呢……
    Chihiro: 好吧 我就是想和你一起休闲一会儿
    Chihiro: 我曾经决定的是要把自己搞得成一个好帅的男的，现在感觉不会是那么好了……

The key idea is to use sampling instead of greedy search, otherwise, the model will just repeat the same sentence over and over again.

I will not release this model as it's based on my >6 years of tweets, which is not suitable for public release. However, I will release the model as well as a demo later with my <6 years of tweets instead.

## To-do List

- [x] Modify the twitter-parser to output your twitter archive into a RLHF-like JSON dataset
- [ ] Allow in-reply-to and quoted tweets to be downloaded, for now it can only generate random tweets (incl. replies)
- [ ] Limit the numbers of tweets to be trained (for Colab/Demo purpose)
- [x] LoRA finetuning with multiple GPUs
- [ ] Hyperparameter tuning (incl. LoRA rank, batch size, learning rate, etc.)
- [ ] Colab notebook for easy deployment (I believe this code can surely run on T4 as we are expecting much shortened tokens)
- [ ] Support other datasets (e.g. Reddit, Weibo, etc. Future plan)

## Installation

It's suggested to use the `conda` environment. 

Installing the dependencies:

```pip install -r requirements.txt```

Probably, you also need to install cuda toolkits (which should match your GPU card):

```conda install cudatoolkit=11.3```

## Data requirements

Download your twitter archive (which is a zip file) and extract it in the project folder, so you should see `Your archive.html` in this project folder. Then, run the twitter-parser.py to parse your twitter archive into a RLHF-like JSON dataset.

## Training

The first step is to extract the twitter archives into a RLHF-like JSON dataset. 

```python twitter-parser.py```

Simply press N for all the prompts. Later I will consider remove the prompts. This code will generate a tweet.json file in the project folder. Do not leak this file to the public if you do not want to, as it contains your personal information.

Then, we can start the finetuning process. If you are calling a single GPU, simply run:

```python finetune.py```

Otherwise call

``` python -m torch.distributed.launch --nproc_per_node N finetune.py``` 

where N is the number of GPUs you are using. Ideally, the training should be longer with 3-4 epochs. I am keeping updating the progress in this repo.

## Inference

Simply call the `generate.py` script to generate tweets. More functionality should come in the future. 

```python3 ./generate.py ./lora-alpaca/```

Where the path-to-your-model-checkpoint is the path to your model checkpoint, by default, it is `./lora-alpaca/`.

You can tune the top-p, top-k, and temperature to generate different tweets. The given parameters are from my tweets, it could be different.

## Benchmark

On a consumer-grade system with 4x3090 graphics cards, and a tweet dataset of 55,498 Tweets of my own tweets, we can expect a training time of *35mins/epoch*, which means the model can be baked within 1-2 hours, or 6-12hours on Colab. 

A general idea is that the loss should be something above 0.8 if the model converges, more importantly, the hyperparameters should be tuned to achieve the best result.

## Credits

This project is based on the following projects:

    tloen/alpaca-lora
    timhutton/twitter-archive-parser
    (potentially) twint-fork