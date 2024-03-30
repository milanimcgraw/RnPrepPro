# chainlit run app.py -w
# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)
# OpenAI Chat completion
from dotenv import load_dotenv
load_dotenv()
# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

import os
import openai
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
#from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.prompt import Prompt, PromptMessage 
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools



#from getpass import getpass

#openai.api_key = getpass("Please provide your OpenAI Key: ")
#os.environ["OPENAI_API_KEY"] = openai.api_key


#Loading data
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyMuPDFLoaderimport os
import openai
from getpass import getpass

#openai.api_key = getpass("Please provide your OpenAI Key: ")
#os.environ["OPENAI_API_KEY"] = openai.api_key

loader = PyMuPDFLoader(
    "https://www.studyguidezone.com/images/nclexrnteststudyguide.pdf",
)

documents = loader.load()

documents[0].metadata

#Split the data
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 50
)

docs = text_splitter.split_documents(documents)

len(docs)

#OpenAI Embeddings Model
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

#FAISS Vectore Store
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever()

#Creating a prompt template:
from langchain.prompts import ChatPromptTemplate

template = """You are a helpful assistant with expertise in Nurse bar exams. Use the following pieces of context from the provided paper to generate 5 question-answer pairs on the subject provided by the user.
If you do not find the subject or context in the paper do not try to make up the QA pairs.
ALWAYS return a "SOURCES" part in your response.
The "SOURCES" should be a reference to the source inside the document from which you got your answer.
Context:
{context}
Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

#Creating a RAG chain:
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

retrieval_augmented_qa_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
)

#We will be using the advanced Multiquery retriever provided by Langchain:
from langchain.retrievers import MultiQueryRetriever
advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=primary_qa_llm)

#We create a chain to stuff our documents into our prompt:
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
document_chain = create_stuff_documents_chain(primary_qa_llm, retrieval_qa_prompt)

#Create the new retrieval chain with advanced retriever:
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)

#And we create our chatbot functions:
user_template = """{input}
Think through your response step by step.
"""
@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        #"model": "gpt-4",
        "temperature": 1.0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
    print(message.content)

    prompt = Prompt(
        #provider=ChatOpenAI.id,
        provider="ChatOpenAI",
        messages=[
            PromptMessage(
                role="system",
                template=template,
                formatted=template,
            ),
            PromptMessage(
                role="user",
                template=user_template,
                formatted=user_template.format(input=message.content),
            ),
        ],
        inputs={"input": message.content},
        settings=settings,
    )

    print([m.to_openai() for m in prompt.messages])

    msg = cl.Message(content="")

    # Call OpenAI
    #async for stream_resp in await client.chat.completions.create(
    #    messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
    #):
        
    #    token = stream_resp.choices[0].delta.content
    #    if not token:
    #        token = ""
    #    await msg.stream_token(token)

    # Update the prompt object with the completion
    result = retrieval_chain.invoke({"input":message.content})
    msg.content = result["answer"]
    #print(temp)
    #prompt.completion = msg.content
    #prompt.completion = temp
    #msg.content = temp
    

    #prompt.completion = completion
    msg.prompt = prompt

    # Send and close the message stream
    await msg.send()
