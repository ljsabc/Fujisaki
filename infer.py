
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch
import sys

from transformers import AutoTokenizer, GenerationConfig

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).cuda().half()
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)


from peft import get_peft_model, LoraConfig, TaskType, PeftModel

#peft_path = "output/chatglm-lora.pt"

#peft_config = LoraConfig(
#    task_type=TaskType.CAUSAL_LM, inference_mode=True,
#    r=12,
#    lora_alpha=32, lora_dropout=0.1
#)

peft_path = sys.argv[1] if len(sys.argv) > 1 else "output/" 
model = PeftModel.from_pretrained(
       model,
       peft_path,
       torch_dtype=torch.float16,
    )


#model = get_peft_model(model, peft_config)
#model.load_state_dict(torch.load(peft_path), strict=False)
# TODO: check if full precision is necessary
torch.set_default_tensor_type(torch.cuda.FloatTensor)
model.eval()

generation_config = GenerationConfig(
        temperature=1,
        top_p=0.999,
        #top_k=150,
        #repetition_penalty=1.1,
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
