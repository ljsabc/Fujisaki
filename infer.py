import torch
import sys

from transformers import AutoTokenizer, GenerationConfig, AutoModel

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision="658202d").cuda().half()
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision="658202d")

from peft import get_peft_model, LoraConfig, TaskType, PeftModel

peft_path = sys.argv[1] if len(sys.argv) > 1 else "output/" 
model = PeftModel.from_pretrained(
       model,
       peft_path,
       torch_dtype=torch.float16,
    )
print(model)

# TODO: check if full precision is necessary
torch.set_default_tensor_type(torch.cuda.FloatTensor)
model.eval()

generation_config = GenerationConfig(
        temperature=0.9,
        top_p=0.975,
        #top_k=150,
        #repetition_penalty=1.1,
        num_beams=1,
        do_sample=True,
)

with torch.no_grad():
    while True:
        context = input(">")
        input_text = f"Context: {context}Answer: " 
        ids = tokenizer([input_text], return_tensors="pt")
        inputs = ids.to("cuda")
        #input_ids = torch.LongTensor([ids]).cuda()
        out = model.generate(
            **inputs,
            max_length=224,
            generation_config=generation_config
            
        )
        out = out.tolist()[0]
        #print(out)
        decoder_output = tokenizer.decode(out)
        #print(decoder_output)
        out_text = "Chihiro:" + decoder_output.split("Answer: ")[1]
        print(out_text)
