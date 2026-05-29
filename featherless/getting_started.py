import os
from dotenv import load_dotenv
from langchain_featherless_ai import ChatFeatherlessAi

load_dotenv()

FEATHERLESSAI_API_KEY = os.getenv("FEATHERLESS_API_KEY")
if not FEATHERLESSAI_API_KEY:
    raise RuntimeError("FEATHERLESS_API_KEY not set. Add it to a .env file or the environment.")

llm = ChatFeatherlessAi(
    api_key=f'{FEATHERLESSAI_API_KEY}',
    base_url="https://api.featherless.ai/v1",
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    (
        "human",
        "I love programming."
    ),
]



ai_msg = llm.invoke(messages)

