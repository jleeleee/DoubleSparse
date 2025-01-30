import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from topk_llama import LlamaTopKAttention

def load_model_and_tokenizer(path, model_name, token_budget):
    print(f"Loading model from path: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(
        path, 
        pad_token_id=tokenizer.pad_token_id,
        **kwargs
    )
    model = model.eval()

    # Convert model to use TopK attention
    config = AutoConfig.from_pretrained(path)
    model = LlamaTopKAttention.convert_llama_attention_to_top_k(model, config, top_k=token_budget)
    
    return model, tokenizer

def generate_text(model, tokenizer, message, history=[], max_input_length=4096, temperature=0.7, max_new_tokens=256):
    """Generates text based on the given prompt and conversation history."""
    # Build conversation format
    conversation = []
    for user_msg, assistant_msg in history:
        conversation.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ])
    conversation.append({"role": "user", "content": message})

    # Apply chat template
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    # Truncate if needed
    if input_ids.shape[1] > max_input_length:
        input_ids = input_ids[:, -max_input_length:]
        attention_mask = attention_mask[:, -max_input_length:]

    # Move to appropriate device
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # Set up generation parameters
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Generate response
    output = model.generate(**generate_kwargs)[0]
    
    # Extract and decode the new tokens
    new_tokens = output[input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat interface with TopK attention.')
    parser.add_argument('--token_budget', type=int, required=True, help='TopK attention budget')
    parser.add_argument('--model', type=str, default="llama3.1-8b-instruct-128k")
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum new tokens to generate')
    args = parser.parse_args()

    # Load model configuration
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    
    # Initialize model and tokenizer
    model_name = args.model
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, args.token_budget)
    max_length = model2maxlen[model_name]

    # Chat loop with history
    history = []
    while True:
        print("\nUser:", end=" ")
        message = input()
        if message.lower() == "quit":
            break
        
        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            message=message,
            history=history,
            max_input_length=max_length,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )
        
        print(f"\nAssistant: {response}")
        
        # Update history
        history.append((message, response))