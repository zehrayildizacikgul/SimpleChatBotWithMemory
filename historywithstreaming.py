from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()


model = ChatOpenAI()


store = {} #for database

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the the best your abilities."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

config = {"configurable": {"session_id": "FirstChat"}}
with_message_history = RunnableWithMessageHistory(chain, get_session_history)



if __name__ == "__main__":
    while True:
        user_input = input("> ")
        for r in with_message_history.stream(
                [
                    HumanMessage(content=user_input)
                ],
                config=config,
        ):
            print(r.content, end=" ")

