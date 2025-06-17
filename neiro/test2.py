from ctransformers import AutoModelForCausalLM

# Загружаем модель из локального файла
model = AutoModelForCausalLM.from_pretrained(
    "D:\\pythonProjects\\mol_voice_recognition\\neiro\\models",  # Путь к папке с моделью
    model_file="deepseek-llm-7b-base-q4_k_m.gguf",  # Имя файла модели
    model_type="deepseek",  # Указываем, что это DeepSeek-LLM
    gpu_layers=0  # 0 - только CPU, можно увеличить при наличии GPU
)

# Тестируем модель
response = model("Как добавить чай в сахар?", max_new_tokens=100)
print(response)
