from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

class TarkovBot:
    def __init__(self):
        model_id = "deepseek-ai/deepseek-llm-7b-chat"
        offload_path = "./model_offload"

        print("⏳ Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            offload_folder=offload_path
        )
        print("✅ Model loaded.")

    def ask(self, prompt: str, max_new_tokens=300):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()


if __name__ == "__main__":
    bot = TarkovBot()
    while True:
        prompt = input("\n📣 You: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        response = bot.ask(prompt)
        print(f"\n🤖 Bot: {response}")