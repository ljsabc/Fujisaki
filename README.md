# Fujisaki
An ongoing (fast prototyping) project to create your own doppelgänger based on your Twitter archive and LoRA+Alpaca-like dataset.

致力于创造一个属于你的不二咲千寻。项目处于初期阶段。

## What is this?

Inspired by Fujisaki Chihiro (i.e., Alter Ego). I thought it would be a fun project, as I really cannot predict my death in the future, and I would like to see how my doppelgänger would react to the world. Thank to the super-strong LLM and LoRA to make it happen by allowing fine-tuning on small corpora.

目前项目会以中文内容为主，也希望该项目能帮助到一些使用中文语料库进行清洗、prompt engineering和模型生成的人。

## How does it work?

将你的Twitter存档目录解压缩放置在项目根目录下，即可自动处理。相关的程序会根据推文的内容、时间、引用、回复等信息，生成一个instruction风格的JSON数据集。然后，我们将该数据集用于某种中文语言模型的调参，生成新的推文，或启用给予上下文的对话。

目前项目的文字生成能力很优秀，但是对话能力尚仍欠缺。

## Colab:

稍后为您带来，厨子正在炒菜。

## Sample outputs:

    # 训练的半成品，由~16k条推文生成

    >最近有没有什么突破性的成就？
    Chihiro:我其实喜欢把没听过的乐队名字念一遍,比如
    “我可来劲了” “干你娘哦啊!?”

    > [ENTER]
    Chihiro:卧槽,这个图好眼熟......
    原来我是在给《不能说的秘密》做PV的啊

    >这游戏真的有毒
    Chihiro:有大佬能教教我怎么快速获得16GB的内存吗?(

    >有哪些令人惊讶的方面？
    Chihiro:啊哈!
    我这么好看的女孩子居然被你们喜欢了(绝望

    >有哪些令人惊讶的方面？
    Chihiro:我是脑残粉



相对于索引、问答目的的项目，该项目会更大程度上利用Sampling，即相似的上下文也会生成非常不一样的回答。希望这样的生成方法能够提供更高的互动性。

## To-do List

- [x] Modify the twitter-parser to output your twitter archive into a instruction dataset
- [x] Categorized in-reply-to and quoted tweets for better conditional generation
- [ ] Allow in-reply-to and quoted tweets to be downloaded, for now it can only generate random tweets/replies/quotes
- [x] LoRA finetuning with multiple GPUs
- [ ] Hyperparameter tuning (incl. LoRA rank, batch size, learning rate, etc.)
- [ ] Colab notebook for easy deployment (I believe this code can surely run on T4 as we are expecting much shortened tokens)
- [ ] Support other datasets (e.g. Reddit, Weibo, etc. Future plan)

## Installation

推荐使用 `conda` 环境。安装依赖：

```pip install -r requirements.txt```

有些时候，你可能要安装cudatoolkit：

```conda install cudatoolkit=11.3```

## Data requirements

解压缩你的推文存档，放置在项目根目录下，即可自动处理。解压缩之后你应该能在项目根目录里面看到`Your archive.html`这个文件。然后，运行`twitter-parser.py`来解析你的推文存档，生成一个RLHF风格的JSON数据集。

同样的，你可以参考`tweets_sample.md`来生成你自己的数据集，或者等待项目更新。

## 训练

目前的模型基于[ChatGLM+LoRa](https://github.com/mymusise/ChatGLM-Tuning/)，与[Luotuo](https://github.com/LC1332/Chinese-alpaca-lora)的处理方式较为类似。

首先使用

    python twitter-parser.py

来处理推文存档，稍许等待之后，你会在项目根目录下看到一个`tweets.md`的文件。这个文件包含了你的推文存档中的所有推文，以及相关的信息。为了保护你的隐私，请不要公开该文件。

生成相应的数据之后，我们需要进一步调用ChatGLM的`tokenizer`来生成对应的tokenized数据集。这一步需要一些时间。

    python3 ./cover_alpaca2jsonl.py --data_path tweets.md --save_path tweets.jsonl
    python ./tokenize_dataset_rows.py --jsonl_path ./tweets.jsonl --save_path tweets.tokens --max_seq_length 128

（可选）使用128个token是因为我的大部份推文，连同instruction一起，也不会超过128个token。如果你的推文较长，可以在生成jsonl之后调用`python length.py`输出的数据适当增加`max_seq_length`的数值。


接下来便可调用`finetune.py`来进行模型训练。根据不同的GPU数量，你可以直接调用

    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    --master_port=1234 \
    finetune.py \
    --dataset_path tweets.tokens \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epoch 1 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate 6e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 20 \
    --output_dir output \
    --ddp_find_unused_parameters false \
    --warmup_steps 50

进行多卡训练，或者

    python finetune.py \
    --dataset_path tweets.tokens \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epoch 1 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 20 \
    --output_dir output \
    --warmup_steps 50 

项目的调参还在研究中，目前的参数和[ChatGLM+LoRa](https://github.com/mymusise/ChatGLM-Tuning/)很类似，不过可以根据GPU数量调节学习率。默认的学习率是`2e-5`，如果卡数有富余也许可以倍增。

## Inference

调用 `infer.py` 进行对话。你可以输入任何问题（但是目前没什么用），不过即便什么都不输入也可以生成一个很类似我的推文。

```python3 ./infer.py```

可以到文件中调节top-p，top-k和temerature，以便生成更多的样本。

## Benchmark

在4张3090的配置下面，训练一个16,990条推文的数据集，每一个epoch需要16分钟。训练大概需要2-3个epoch能够达成最佳状态。

## Credits

This project is based on the following projects:

    27182812/ChatGLM-chinese-insturct
    timhutton/twitter-archive-parser
    LC1332/Chinese-alpaca-lora (Donated❤️)

Inspired by the following projects:

    tloen/alpaca-lora
    HuggingFace: KBlueLeaf/guanaco-7B-lora-embed
    (potentially) twint-fork