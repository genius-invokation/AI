import os
import re
import asyncio

# from torch import FloatTensor

# import torch
import fcntl
from together import Together
from openai import OpenAI
# from tqdm.asyncio import tqdm_asyncio


class APIUsageTracker:
    def __init__(self, model, cost_file = 'total_cost.txt', max_temp_lines = 10):
        self.model = model
        self.cost_file = cost_file
        self.max_temp_lines = max_temp_lines
        self.temp_cost = 0
        self.temp_count = 0
        self._initialize_file()

    def _initialize_file(self):
        if not os.path.exists(self.cost_file):
            with open(self.cost_file, 'w') as f:
                f.write('0.0')

    def update(self, prompt_tokens, completion_tokens):
        if self.model == "gpt-4o-mini":
            prompt_cost = (prompt_tokens / 1_000_000) * 0.15
            completion_cost = (completion_tokens / 1_000_000) * 0.60
        elif self.model == "meta-llama/Meta-Llama-3-70B-Instruct-Turbo":
            prompt_cost = (prompt_tokens / 1_000_000) * 0.88
            completion_cost = (completion_tokens / 1_000_000) * 0.88
        elif self.model == "Qwen/Qwen2.5-7B-Instruct-Turbo":
            prompt_cost = (prompt_tokens / 1_000_000) * 0.3
            completion_cost = (completion_tokens / 1_000_000) * 0.3
        elif self.model == "Qwen/Qwen2.5-72B-Instruct-Turbo":
            prompt_cost = (prompt_tokens / 1_000_000) * 1.2
            completion_cost = (completion_tokens / 1_000_000) * 1.2
        else:
            prompt_cost = 0
            completion_cost = 0
            # TODO: Add a warning log
            # raise ValueError(f"Unsupported model: {model}")

        call_cost = prompt_cost + completion_cost

        self.temp_cost += call_cost
        self.temp_count += 1

        if self.temp_count >= self.max_temp_lines:
            self._update_total_cost()

        return call_cost

    def _update_total_cost(self):
        with open(self.cost_file, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                current_cost = float(f.read().strip() or '0.0')
                new_cost = current_cost + self.temp_cost
                f.seek(0)
                f.truncate()
                f.write(f'{new_cost:.6f}')
                self.temp_cost = 0
                self.temp_count = 0
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def get_total_cost(self):
        with open(self.cost_file, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return float(f.read().strip() or '0.0')
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

class LLMAPI:   
    def __init__(self, model, useapi, api_key, requests_per_minute):
        self.model = model
        self.useapi = useapi

        if self.useapi == "openai":
            self.client = OpenAI(api_key = api_key)
        elif self.useapi == "together":
            self.client = Together(
                api_key = api_key)  # public key
        elif self.useapi == "vllm":
            self.client = Together(
                api_key = "empty", base_url = "http://localhost:54323/v1")  # public key


        self.api_tracker = APIUsageTracker(self.model)

    def get_api_response(self, prompt):
        response = self.client.chat.completions.create(
            model = self.model,
            messages = [{"role": "user", "content": prompt}],
        )

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        call_cost = self.api_tracker.update(prompt_tokens, completion_tokens)

        return response.choices[0].message.content

    def get_api_responses(self, prompts):
        return [self.get_api_response(prompt) for prompt in prompts]

