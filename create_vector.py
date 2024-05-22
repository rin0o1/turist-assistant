from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import env
import os
from langchain_pinecone import PineconeVectorStore

os.environ["OPENAI_API_KEY"] = env.OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = env.PINECONE_API_KEY
os.environ["PINECONE_ENV"] = env.PINECONE_ENV

loader_dic = DirectoryLoader("data/")
data = loader_dic.load()

text_splitter = RecursiveCharacterTextSplitter()
splits = text_splitter.split_documents(data)

embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

index_name = ""
vectordb = PineconeVectorStore.from_documents(splits, embedding, index_name)
