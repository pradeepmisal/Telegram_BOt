
from langchain_openai import ChatOpenAI  # Updated import
from langchain.schema import HumanMessage

# OpenRouter API Key
api_key = "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"

# Initialize LLaMA 3.1 model via OpenRouter
llm = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key
)

# Example Chat
messages = [HumanMessage(content="Tell me about AI-driven drug discovery.")]

# Get Response
response = llm.invoke(messages)
print(response.content)

