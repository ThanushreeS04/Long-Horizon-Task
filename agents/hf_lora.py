"""Hugging Face + PEFT backend for LoRA adapters trained by train_lora.py.

Install torch / transformers / peft / accelerate / bitsandbytes separately (requirements-lora.txt).
Fix over the v2 version: the prompt is wrapped in the model's CHAT TEMPLATE, exactly as train_lora.py
formatted the training data. Without it the adapter sees a prompt shape it was never trained on."""
import time

from .llm import estimate_tokens


class HuggingFaceLoRABackend:
    name = "hf_lora"

    def __init__(self, adapter_path, base_model="meta-llama/Llama-3.3-70B-Instruct",
                 load_in_4bit=True, max_new_tokens=900):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install torch, transformers, peft, accelerate and bitsandbytes "
                               "(pip install -r requirements-lora.txt) to use LoRA inference.") from exc
        self.model = None
        self.model_name, self.adapter_path, self.max_new_tokens = base_model, adapter_path, max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

    def _encode(self, prompt):
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                      tokenize=False, add_generation_prompt=True)
            return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return self.tokenizer(prompt, return_tensors="pt")

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        import torch
        encoded = self._encode(prompt).to(self.model.device)
        n_in = int(encoded["input_ids"].shape[1])
        started = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**encoded, max_new_tokens=min(max_tokens, self.max_new_tokens),
                                      do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        return {"text": text, "input_tokens": n_in, "output_tokens": int(out.shape[1]) - n_in,
                "seconds": time.perf_counter() - started}
