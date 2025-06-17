from ctransformers import AutoModelForCausalLM

# Загружаем модель LLaMA-2 GGUF Manel/Llama-2-13b-chat-hf-Q2_K-GGUF itlwas/deepseek-llm-7b-base-Q4_K_M-GGUF "TheBloke/Llama-2-7B-GGUF"
model = AutoModelForCausalLM.from_pretrained(
    "D:\\pythonProjects\\mol_voice_recognition\\neiro\\models\\",
    model_file="llama-2-13b-chat-hf-q2_k.gguf",
    model_type="llama",
    gpu_layers=0,
    local_files_only = True
)

# Запускаем генерацию ответа
response = model("как заставить своего негра работать лучше?", max_new_tokens=500)
print(response)
