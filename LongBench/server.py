import argparse
import json
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument('--token_budget', type=int, required=True)
parser.add_argument('--model', type=str, default="llama3.1-8b-instruct-128k")
parser.add_argument('--host', type=str, default="0.0.0.0")
parser.add_argument('--port', type=int, default=8000)

args = parser.parse_args()
# ----------------------------
# Your existing model code
# ----------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from topk_llama import LlamaTopKAttention


def load_model_and_tokenizer(path, model_name, token_budget):
    print(f"Loading model from path: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(
        path,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs
    )
    model.eval()

    config = AutoConfig.from_pretrained(path)
    model = LlamaTopKAttention.convert_llama_attention_to_top_k(
        model, config, top_k=token_budget
    )
    return model, tokenizer

def generate_text(
    model,
    tokenizer,
    message,
    history=[],
    max_input_length=4096,
    temperature=0.7,
    max_new_tokens=256
):
    """
    Minimal 'chat-style' generator. For an OpenAI-like completion,
    we treat 'message' as the user's prompt. For chat, we build a
    conversation (history) + message. The code here is your same
    function with slight modifications.
    """
    # Build conversation format
    conversation = []
    for user_msg, assistant_msg in history:
        conversation.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ])
    conversation.append({"role": "user", "content": message})

    # Convert conversation to input_ids
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    # Truncate if needed
    if input_ids.shape[1] > max_input_length:
        input_ids = input_ids[:, -max_input_length:]
        attention_mask = attention_mask[:, -max_input_length:]

    # Move to device
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # Generate
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )[0]

    # Extract newly generated tokens
    new_tokens = output[input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response

def compute_token_logprobs(
    model,
    tokenizer,
    prompt: str
):
    """
    A minimal example showing how to return per‑token logprobs,
    which the harness needs for loglikelihood tasks.

    For each token in the *continuation*, we compute P(next_token|prompt_so_far).
    In an *ideal* solution for perplexity or loglikelihood, we'd do some offset
    copying for each sub-token. This is a simplified approach that works for
    short completions.
    """
    # Tokenize
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # shape: [batch, seq_len, vocab_size]
        logits = outputs.logits[:, :-1, :]

    # We want the logprobs for each *observed* token from 1..end
    # E.g. the next-token after token[i-1].
    # Because we have local model access, we can do the full cross-entropy.
    # We'll do a naive approach for demonstration:
    vocab_logprobs = torch.log_softmax(logits, dim=-1)  # [1, seq_len-1, vocab_size]

    # The *observed* tokens from index=1..end
    observed_tokens = input_ids[:, 1:]  # shape [1, seq_len-1]
    seq_len_minus_1 = observed_tokens.shape[1]

    # Gather logprobs of observed tokens
    gathered = torch.gather(vocab_logprobs, 2, observed_tokens.unsqueeze(-1)).squeeze(-1)
    gathered = gathered[0].tolist()  # shape [seq_len-1]

    # The tokens themselves:
    tokens = input_ids[0].tolist()

    # We produce "token_logprobs" from index=1..end. If you want the
    # logprob of each token from the start, you can offset differently.
    token_strings = tokenizer.convert_ids_to_tokens(tokens)
    
    # Return two aligned lists: tokens, and their logprobs
    # matching the OpenAI "logprobs" shape
    return token_strings[1:], gathered


# ----------------------------
# Pydantic request models
# ----------------------------

class CompletionRequest(BaseModel):
    prompt: Union[str, List[str]] = ""
    max_tokens: int = 128
    temperature: float = 0.7
    logprobs: Optional[int] = None  # if set, harness may request token logprobs


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.7


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI()

@app.on_event("startup")
def load_model_on_startup():
    global model, tokenizer, max_length, args
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))

    model_name = args.model
    path = model2path[model_name]
    max_length = model2maxlen[model_name]

    model_, tokenizer_ = load_model_and_tokenizer(path, model_name, args.token_budget)
    model = model_
    tokenizer = tokenizer_
    print("Model loaded on startup.")


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    """
    Mimic the OpenAI Completions endpoint so that the harness
    (with --model local-completions) can call us directly.
    Optionally handle logprobs if requested.
    """
    # The harness typically sends one prompt at a time, but
    # it can also send an array if "prompt" is a list. We'll just
    # handle the single prompt for simplicity:
    if isinstance(req.prompt, list):
        prompt_text = req.prompt[0] if req.prompt else ""
    else:
        prompt_text = req.prompt

    # For tasks that want logprobs (loglikelihood tasks), harness sets `logprobs`.
    if req.logprobs is not None and req.logprobs > 0:
        # We'll do a single forward pass over the entire prompt to get token-level logprobs
        # Then we won't actually generate new tokens (max_tokens=0).
        tokens, token_logprobs = compute_token_logprobs(model, tokenizer, prompt_text)

        # The harness specifically wants the logprobs for the "continuation" portion,
        # i.e. from the last token of the prompt onward. That can be more complicated
        # if there's a separate "stop" string. For simplicity, we do the naive approach:
        # we return the entire sequence's logprobs (minus the first token).

        # Build an OpenAI-like "logprobs" structure:
        # "tokens" => the string tokens
        # "token_logprobs" => float values
        # "top_logprobs" => optional dict of top tokens (omitted here for brevity)
        # "text_offset" => optional array for char offsets

        # Then "text" is empty or just the last token? Typically harness sets max_tokens=0
        # for loglikelihood. We'll produce an empty "text" here, but some tasks
        # might do partial generation. Adjust to your needs.
        response = {
            "id": "cmpl-xxx",
            "object": "text_completion",
            "created": 1234567890,
            "model": args.model,
            "choices": [
                {
                    "text": "",  # no new text generated
                    "index": 0,
                    "finish_reason": "length",
                    "logprobs": {
                        "tokens": tokens,
                        "token_logprobs": token_logprobs,
                    },
                }
            ]
        }
        return response

    else:
        # Normal “generate text” flow
        output_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            message=prompt_text,
            history=[],  # no chat style history for completions
            max_input_length=max_length,
            temperature=req.temperature,
            max_new_tokens=req.max_tokens
        )

        return {
            "id": "cmpl-xxx",
            "object": "text_completion",
            "created": 1234567890,
            "model": args.model,
            "choices": [
                {
                    "text": output_text,
                    "index": 0,
                    "finish_reason": "stop"
                }
            ]
        }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """
    Mimic the OpenAI ChatCompletions endpoint so that the harness
    (with --model local-chat-completions) can call us.
    Note that loglikelihood-based tasks do not typically use chat
    endpoints. 
    """
    history = []
    current_user = None

    # Convert the chain of messages to a user->assistant format
    for m in req.messages:
        if m.role == "user":
            if current_user is not None:
                # user repeated messages consecutively? Just store an empty assistant response
                history.append((current_user, ""))
            current_user = m.content
        elif m.role == "assistant":
            if current_user is not None:
                history.append((current_user, m.content))
                current_user = None
        else:
            # 'system' or 'other' roles: incorporate as you like
            # For simplicity, we ignore them or treat them as user instructions:
            pass

    user_message = current_user if current_user else ""

    # Generate 
    output_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        message=user_message,
        history=history,
        max_input_length=max_length,
        temperature=req.temperature,
        max_new_tokens=req.max_tokens
    )

    return {
        "id": "chatcmpl-xxx",
        "object": "chat.completion",
        "created": 1234567890,
        "model": args.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text
                },
                "finish_reason": "stop"
            }
        ]
    }

def main():
    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
