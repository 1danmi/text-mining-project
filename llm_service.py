from llama_cpp import Llama


class llm_service:
    def __init__(self, model_path: str, num_workers: int = 2):
        self._llm = Llama(model_path=model_path, n_gpu_layers=-1)
        self._num_workers = num_workers

    def batch_generate(self, prompts: list[str], max_token: int = 128, stop: list[str] | None = None):
        return [self.generate(prompt, max_tokens=max_token) for prompt in prompts]

    def generate(self, prompt: str, max_tokens: int = 128, stop: list[str] | None = None):
        result = self._llm(prompt, max_tokens=max_tokens, stop=stop)
        return result["choices"][0]["text"]

    def __del__(self):
        del self._llm



