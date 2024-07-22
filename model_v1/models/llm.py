from langchain_openai import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager

callback_manager = BaseCallbackManager([StdOutCallbackHandler()])
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True, callbacks=callback_manager)