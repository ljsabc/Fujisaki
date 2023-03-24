# Fujisaki
An ongoing (fast prototyping) project to create your own doppelgänger based on your Twitter archive and LoRA+Alpaca-like dataset.

致力于创造一个属于你的不二咲千寻。项目处于初期阶段。

## What is this?

Inspired by Fujisaki Chihiro (i.e., Alter Ego). I thought it would be a fun project, as I really cannot predict my death in the future, and I would like to see how my doppelgänger would react to the world. Thank to the super-strong LLM and LoRA to make it happen by allowing fine-tuning on small corpora.

On the other hand, this can also be seen as a bootstrapper project to fine-tune a Chinese LLM with a Chinese dataset, as I have adapted some of the code to tailor for Chinese (and probably other languages) synthesis.

## How does it work?

By placing your twitter archive in the project root, we parse the archive into a instruction-like JSON dataset. Then, we finetune the LLM with the dataset, and generate tweets with the finetuned model.

For now it cannot generate replies, but it can generate random tweets. 

## Colab:

Training Colab is available at: https://colab.research.google.com/drive/1AGndDZLHwN_tj6vSz_0285TlQKpOo5Cj?usp=sharing

The inference Colab will be available soon (as I do not want to use my own old tweets).

## Sample outputs:

    # results after a 4-epoch training, with ~50k tweets, original post generation only

    Chihiro: 卧槽我要喝了！
    Chihiro: 又是这样了……
    Chihiro: 我是真的有点迷郭了，刚才看到这个就想起来我一直在用猫粮盆上写“维基百科”
    Chihiro: 我好想买个1080Ti了啊
    Chihiro: 我妈跟我说这个猫比饼盘猫多了，因为饼盘猫在垃圾桶里不会发现什么东西就听到他的声 音。所以我妈觉得这个猫比饼盘猫好，因为他会哭，那就可以知道他是否在家里。

    （我妈还说这个猫比饼盒猫更好，因为饼盒猫会哭，然后又吐）

    Chihiro: 我想做个人博客，但是我不知道我真的能开心着
    还得说自己的故事了，没有话就没有意义

The key idea is to use sampling instead of greedy search. By enabling repetition_penalty, the generated tweets should be at least very close to the original tweets. Then it's up to the transferred knowledge from the base model to "glue" the knowledge with the tweet context.

## To-do List

- [x] Modify the twitter-parser to output your twitter archive into a RLHF-like JSON dataset
- [x] Categorized in-reply-to and quoted tweets for better conditional generation
- [ ] Allow in-reply-to and quoted tweets to be downloaded, for now it can only generate random tweets/replies/quotes
- [x] LoRA finetuning with multiple GPUs
- [ ] Hyperparameter tuning (incl. LoRA rank, batch size, learning rate, etc.)
- [ ] Colab notebook for easy deployment (I believe this code can surely run on T4 as we are expecting much shortened tokens)
- [ ] Support other datasets (e.g. Reddit, Weibo, etc. Future plan)
- [ ] Pretrain a Chinese llama first (Confirmed with @RealJosephus that there will be a better based model)

## Installation

It's suggested to use the `conda` environment. 

Installing the dependencies:

```pip install -r requirements.txt```

Probably, you also need to install cuda toolkits (which should match your GPU card):

```conda install cudatoolkit=11.3```

## Data requirements

Download your twitter archive (which is a zip file) and extract it in the project folder, so you should see `Your archive.html` in this project folder. Then, run the twitter-parser.py to parse your twitter archive into a RLHF-like JSON dataset.

## Training

The model is currently based on [Luotuo](https://github.com/LC1332/Chinese-alpaca-lora), but could be easily adapted to other models.

The first step is to extract the twitter archives into a RLHF-like JSON dataset. 

```python twitter-parser.py <lang>```

where `<lang>` represents the language you use. By default it's English, but for now you can set it to `zh_hans` to support Simplified Chinese. i18n is welcomed for the prompt generation. 

This code will generate a tweet.json file in the project folder. Do not leak this file to the public if you do not want to, as it contains your personal information.

Next, convert the Luotuo weights to a HF checkpoint:

    mkdir luotuo
    cd luotuo 
    wget https://huggingface.co/silk-road/luotuo-lora-7b-0.3/raw/main/adapter_config.json
    wget https://huggingface.co/silk-road/luotuo-lora-7b-0.3/resolve/main/adapter_model.bin
    cd ..
    python ./export_hf_checkpoint.py ./luotuo
    mv ./hf_ckpt ./luotuo_ckpt

This will take a little while, and the luotuo checkpoint will be used later as the base model for LoRa. Then, we can start the finetuning process. If you are calling a single GPU, simply run:

```python finetune.py```

Otherwise call

```python -m torch.distributed.launch --nproc_per_node N finetune.py``` 

where N is the number of GPUs you are using. [An issue](https://github.com/tloen/alpaca-lora/issues/8) gives a detailed explanation of the distributed training.

The key to success is **overfitting with bigger models** and probably with a longer training time. The lower the training/eval loss you achieve, the better performance you will get. The training epoch is tested with 3, but you can make it shorter if you are running of computation budget. In that case, the learning rate could also help, but I did not test it. Send an issue for reporting your results, or a PR for improving the code.

## Inference

Simply call the `generate.py` script to generate tweets. More functionality should come in the future. 

```python3 ./generate.py ./lora-alpaca/```

Where the path-to-your-model-checkpoint is the path to your model checkpoint, by default, it is `./lora-alpaca/`.

You can tune the top-p, top-k, and temperature to generate different tweets. The given parameters are from my tweets, it could be different.

## Benchmark

On a consumer-grade system with 4x3090 graphics cards, and a tweet dataset of 55,498 Tweets of my own tweets, we can expect a training time of *40mins/epoch*, which means the model can be baked within 1-2 hours, or 6-12hours on Colab. 

A general idea is that the validation loss should be something around 0.7-0.8 if the model converges (with a good corpus).

## Credits

This project is based on the following projects:

    tloen/alpaca-lora
    timhutton/twitter-archive-parser
    LC1332/Chinese-alpaca-lora (I will consider a donation)
    HuggingFace: KBlueLeaf/guanaco-7B-lora-embed
    (potentially) twint-fork