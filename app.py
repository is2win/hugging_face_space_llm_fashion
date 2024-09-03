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
    max_tokens=4000,
    n_ctx=4000,
    n_gpu_layers=-1,
)

prompt_template = """
<|start_header_id|>system<|end_header_id|>
Вы личный ассистент по моде.
ЗАПРЕЩАЮ ВЫДУМЫВАТЬ и вредить людям
<|eot_id|>
<|start_header_id|>контекст<|end_header_id|>
{context_info}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Используй максимально emoji
Ответь на вопросы строго на основе предоставленного Контекста.
Если информация в контексте отсутствует, напиши сообщение "Ответа нет". 
Вопрос: {question_info}
<|eom_id|>
<|start_header_id|>assistant<|end_header_id|>

"""
prompt = PromptTemplate.from_template(prompt_template)
llm_chain = prompt | llm

context_info = """
Боб с боковым пробором Очень стильно выглядит стрижка мужской боб. Особенности прически: объем на макушке и в теменной зоне; боковые пряди спадают на виски; длинная челка, которую зачесывают на лоб или укладывают с пробором. В 2024 году стильную стрижку рекомендуют сочетать с боковым пробором. Такую прическу могут носить мужчины в любом возрасте. Боковой пробор невероятно популярен у современных бизнесменов. Подобный вариант укладки выглядит строго и презентабельно, подходит мужчинам, которые желают всегда выглядеть безупречно и собранно. 
Используем только средство для укладки - Barbara 100
"""

question_info= """Боб это?"""



output = llm_chain.invoke({"context_info": context_info, 'question_info':question_info}, config={"max_tokens": 5000})
# output = llm.invoke(prompt_template)
print(output)


def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()