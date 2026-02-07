# modules/llm_manager.py
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class LLMManager:
    def __init__(self):
        self.models = {}

    def get_model(self, name):
        if name not in self.models:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if model.device.type == "cuda" else -1,
                return_full_text=False
            )
            self.models[name] = pipe
        return self.models[name]

    def generate(self, prompt, model_name="mistral"):
        pipe = self.get_model(model_name)
        outputs = pipe(prompt, max_new_tokens=200)
        return outputs[0]["generated_text"]
