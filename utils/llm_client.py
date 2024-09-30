import os
import anthropic
from openai import OpenAI
from reka.client import Reka
import google.generativeai as genai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


anthropic_client = anthropic.Anthropic(max_retries=3)
openai_client = OpenAI(max_retries=3)
deepinfra_client = OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=os.environ.get("DEEPINFRA_API_KEY"),
    max_retries=3
)
deepseek_client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPKSEEK_API_KEY"),
    max_retries=3
)
yi_client = OpenAI(
    base_url="https://api.01.ai/v1",
    api_key=os.environ.get("YI_API_KEY"),
    max_retries=3
)
reka_client = Reka(api_key=os.environ.get("REKA_API_KEY"))
genai.configure(api_key=os.environ.get("GOOGLEAI_API_KEY"))
gemini_1_5_pro_model_client = genai.GenerativeModel('gemini-1.5-pro',
                                                    generation_config=genai.GenerationConfig(temperature=0.0,
                                                                                             max_output_tokens=1024)
                                                    )
gemini_1_5_flash_model_client = genai.GenerativeModel('gemini-1.5-flash',
                                                      generation_config=genai.GenerationConfig(temperature=0.0,
                                                                                               max_output_tokens=1024)
                                                      )
mistral_client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))


CLIENT_MAP = {
    "meta-llama/Meta-Llama-3.1-405B-Instruct": deepinfra_client,
    "meta-llama/Meta-Llama-3.1-70B-Instruct": deepinfra_client,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": deepinfra_client,
    "meta-llama/Meta-Llama-3-70B-Instruct": deepinfra_client,
    "meta-llama/Meta-Llama-3-8B-Instruct": deepinfra_client,
    "google/gemma-2-27b-it": deepinfra_client,
    "google/gemma-2-9b-it": deepinfra_client,
    "Qwen/Qwen2-72B-Instruct": deepinfra_client,
    "Qwen/Qwen2-7B-Instruct": deepinfra_client,
    "microsoft/Phi-3-medium-4k-instruct": deepinfra_client,
    "microsoft/WizardLM-2-8x22B": deepinfra_client,
    "microsoft/WizardLM-2-7B": deepinfra_client,
    "mistralai/Mistral-7B-Instruct-v0.3": deepinfra_client,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": deepinfra_client,
    "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1": deepinfra_client,
    "databricks/dbrx-instruct": deepinfra_client,
    "gpt-3.5-turbo": openai_client,
    "gpt-4o-mini": openai_client,
    "gpt-4o": openai_client,
    "claude-3-5-sonnet-20240620": anthropic_client,
    "claude-3-haiku-20240307": anthropic_client,
    "deepseek-chat": deepseek_client,
    "deepseek-coder": deepseek_client,
    "yi-large": yi_client,
    "yi-large-turbo": yi_client,
    "reka-core-20240501": reka_client,
    "reka-flash-20240226": reka_client,
    "gemini-1.5-pro": gemini_1_5_pro_model_client,
    "gemini-1.5-flash": gemini_1_5_flash_model_client,
    "mistral-large-latest": mistral_client
}


def ask_llm(text, model_name):
    client = CLIENT_MAP[model_name]
    if isinstance(client, OpenAI):
        chat_completion = client.chat.completions.create(
                model=model_name,
                temperature=0.0,  # temperature set to 0.0 for greedy decoding and "deterministic" output
                messages=[
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                max_tokens=1024
        )
        return chat_completion.choices[0].message.content

    elif isinstance(client, anthropic.Anthropic):
        chat_completion = client.messages.create(
                model=model_name,
                temperature=0.0,  # temperature set to 0.0 for greedy decoding and "deterministic" output
                messages=[
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                max_tokens=1024
            )
        return chat_completion.content[0].text
    elif isinstance(client, Reka):
        chat_completion = client.chat.create(
                model=model_name,
                temperature=0.0,  # temperature set to 0.0 for greedy decoding and "deterministic" output
                messages=[
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                max_tokens=1024
            )
        return chat_completion.responses[0].message.content
    elif isinstance(client, genai.GenerativeModel):
        chat = client.start_chat(history=[])
        chat_completion = chat.send_message(text)
        return chat_completion.text
    elif isinstance(client, MistralClient):
        chat_response = client.chat(
            model=model_name,
            messages=[
                ChatMessage(role="user", content=text)
            ],
            temperature=0.0,
            max_tokens=1024
        )
        return chat_response.choices[0].message.content
