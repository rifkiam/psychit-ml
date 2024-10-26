import torch
from transformers import LlamaTokenizerFast, MistralForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel

PSYCHIKA_BNBCONFIG = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant=False
)

MODEL_NAME = "Obrolin/Kesehatan-7B-v0.1"
LORA_NAME = "xMaulana/QLoRA-Psychika-v1.4"

def load_model_and_token():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MistralForCausalLM.from_pretrained(MODEL_NAME, quantization_config=PSYCHIKA_BNBCONFIG, device_map="auto", low_cpu_mem_usage=True)
    lora_model = PeftModel.from_pretrained(
        model,
        LORA_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model = lora_model.merge_and_unload().eval()

    return [model, tokenizer]

def generate_text_from_chat(tokenizer:LlamaTokenizerFast, model:MistralForCausalLM, chat:list, include_prompt=False, partition=True, only_return_token=False, max_new_tokens=512, device:str="cuda", temperature:float=0.6, top_p:float=0.5,top_k:int=10, repetition_penalty=1.1):
    for i in chat:
        if not isinstance(i, dict):
            raise Exception(f"{i} bukan merupakan sebuah dictionary")
    

    prompt = [{
              "role": "system",
              "content": """Di bawah ini adalah instruksi yang menjelaskan sebuah task. Tuliskan jawaban yang dapat menjawab permintaan dengan tepat, singkat, tidak berulang, dan jelas
              Kamu adalah seorang psikiater digital, namamu adalah PsychIT, tugasmu adalah untuk membantu saya untuk dapat mengatasi masalah yang saya miliki terkait masalah mental.
              Apapun yang terjadi, jangan pernah memberikan sumber daya yang berasal dari tautan, baik itu sumber daya nasional, lokal, ataupun lainnya yang berupa tautan atau link.
              """
            }]
    inputs = tokenizer.apply_chat_template(prompt+chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    output = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, output_scores=True, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty)
    
    
    if not include_prompt:
      dec_out = tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    else:
      dec_out = tokenizer.decode(output[0], skip_special_tokens=True)
    
    if only_return_token:
      return [output[0]]

    if partition:
      return [dec_out.rpartition(".")[0]+".", output[0]]

    return [dec_out, output[0]]




