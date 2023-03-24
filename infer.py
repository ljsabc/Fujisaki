
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch

from transformers import AutoTokenizer, GenerationConfig

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map='auto').cuda().half()
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)


from peft import get_peft_model, LoraConfig, TaskType, PeftModel

peft_path = "output/chatglm-lora.pt"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    r=8,
    lora_alpha=32, lora_dropout=0.1
)


model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
# TODO: check if full precision is necessary
torch.set_default_tensor_type(torch.cuda.FloatTensor)
model.eval()

generation_config = GenerationConfig(
        temperature=0.95,
        top_p=0.85,
        #top_k=top_k,
        repetition_penalty=1.2,
        num_beams=1,
        do_sample=True,
)

with torch.no_grad():
    while True:
        context = input(">")
        input_text = f"Context: {context}Answer: " 
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).cuda()
        out = model.generate(
            input_ids=input_ids,
            max_length=160,
            generation_config=generation_config
            
        )
        out_text = "Chihiro:" + tokenizer.decode(out[0]).split("Answer: ")[1]
        print(out_text)
