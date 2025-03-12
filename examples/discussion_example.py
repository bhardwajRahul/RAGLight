from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv
import os

load_dotenv()
Settings.setup_logging()

model_name = 'llama3.2:3b'
system_prompt = Settings.DEFAULT_SYSTEM_PROMPT

llmOllama = Builder() \
.with_llm(Settings.OLLAMA, model_name=model_name, system_prompt=system_prompt) \
.build_llm()

# llmLMStudio = Builder() \
# .with_llm(Settings.LMStudio, model_name=model_name, system_prompt_file=system_prompt_directory) \
# .build_llm()

# llmLMStudio = Builder() \
# .with_llm(Settings.MISTRAL, model_name=model_name, system_prompt_file=system_prompt_directory) \
# .build_llm()

def chat():
    query = input(">>> ")
    if query == "quit" or query == "bye" : 
        print('🤖 : See you soon 👋')
        return
    response = llmOllama.generate({"question": query})
    # response = llmLMStudio.generate({"question": query})
    print('🤖 : ', response)
    return chat()

chat()
