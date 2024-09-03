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
# model_name = "second-state/Mistral-Nemo-Instruct-2407-GGUF"
model_name = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
model_file = "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"


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
    max_tokens=200,
    n_ctx=200,
    n_gpu_layers=-1,
)

def predict(user_input):
    prompt_template = """
    Вопрос: {question_info}
    """
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = prompt | llm
    output = llm_chain.invoke({'question_info': user_input})
    return output

# Создание интерфейса Gradio
demo = gr.Interface(fn=predict,
             inputs="text",
             outputs="text")

demo.launch(share=True)