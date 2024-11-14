import os
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.core.base.llms.types import MessageRole, ChatMessage

dashscope_llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.environ["DASHSCOPE_API_KEY"]
)

ques=input("Do you have any questions?")
messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(role=MessageRole.USER, content=ques),
]
resp = dashscope_llm.chat(messages)
    print(resp)

for i in range(3):
    ques=input("Above is my answer .Do you have any other questions?")
    if ques.lower() in ["thanks","exit"]:
        break
    messages.append(
        ChatMessage(role=MessageRole.ASSISTANT, content=resp.message.content)
    )
    messages.append(
        ChatMessage(role=MessageRole.USER, content=ques)
    )
    resp = dashscope_llm.chat(messages)
    print(resp)

print("The chat is over")
