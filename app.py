import gradio as gr
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import os
from huggingface_hub import hf_hub_download

# Указываю имя репозитория и название скачиваемой модели

model_name = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
model_file = "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

# model_name = "lmstudio-community/Mistral-Nemo-Instruct-2407-GGUF"
# model_file = "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"


# Загрузка с Hugging Face Hub
model_path = hf_hub_download(
    model_name,
    filename=model_file,
    local_dir='models/',  # Загрузку сделаем в папку "models" - опционально
    token="token"  #тут указываем ваш токен доступа с huggingface (Setting -> Access Toekns -> New token -> Generate Token)
)

print("My model path:", model_path)
# Путь до модели
# model_path = "/kaggle/working/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

# Инициализирую модель через LlamaCpp
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.5,
    max_tokens=500,
    n_ctx=4000,
    n_gpu_layers=-1,
)

def predict(user_input):
    # Создаём простой шаблон
    template = """
    <|start_header_id|>system<|end_header_id|>
    Вы личный ассистент по моде.
    Рассуждаешь логически перед тем как дать ответ
    Answer the question based only on the context
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Используй emoji и форматирование markdown текста чтобы иллюстрировать ответ
    Отвечай на вопрос по пунктам как для новичков
    Вопрос: {question}
    <|eom_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    # Используйте вашу модель для обработки запроса
    try:
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm

        output = chain.invoke({'question':user_input})
    except Exception as e:
        output = f"Ошибка: {str(e)}"
    
    return output

# Создание интерфейса Gradio
demo = gr.Interface(fn=predict,
             inputs="text",
             outputs="text")

demo.launch(debug=True)