import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# 微调后的模型路径
MODEL_PATH = "/data/ckpt_neko_9b/v2-20260421-133543/checkpoint-460"
DEVICE = "cuda:0"
# ====================================================


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  
    device_map=DEVICE,           
    # low_cpu_mem_usage=True
)


generation_config = GenerationConfig(
    max_new_tokens=2048,     
    temperature=1.1,        
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.05, # 重复惩罚
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


def chat(query: str):
    
    messages = [
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    print(f"\n用户：{query}")
    print(f"模型：{response}\n")

if __name__ == "__main__":
    # 替换成你的测试问题！
    chat("你是谁呀")
    chat("我打你一拳")