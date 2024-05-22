from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SerpAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from langchain_community.document_loaders import DirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_openai_functions_agent, OpenAIFunctionsAgent, create_tool_calling_agent
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain import hub
import env
import os
import sys

os.environ["OPENAI_API_KEY"] = env.OPENAI_API_KEY

loader_dic = DirectoryLoader("data/")
data = loader_dic.load()

text_splitter = RecursiveCharacterTextSplitter()
splits = text_splitter.split_documents(data)

embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
vectordb = Chroma.from_documents(
    documents=splits, embedding=embedding, collection_name="general_info_tool"
)

llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])

general_info = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectordb.as_retriever()
)

general_info_tool = Tool(
    name="general_info_tool",
    func=general_info.run,
    description="usa questo strumento per rispondere a tutte le domande sull' Albergo dell'Orso Bo come per esempio soggiorno, arrivi, camere, animazione, parcheggio ed altro.",
)


def create_quote_tool_function(*args, **kwargs):
    return "https://www.orso-bo.it/quote/"    

create_quote_tool = Tool(
    name="create_quote_tool",
    func=create_quote_tool_function,    
    description="usa questo strumento per creare un preventivo per un soggiorno all'Albergo dell'Orso Bo o avere informazioni sulle disponibilita' delle camere."
)


memory_key = "chat_history"
memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)

# setting default prompt
system_message = SystemMessage(
    content=(
        "Tu sei l'assistente dell'Albergo dell'Orso Bo e dovrai rispondere alle domande che i clienti ti faranno riguardanti la struttura. \
        Rispondi solo se il contenuto della tua risposta viene dallo strumento 'general_info_tool' o 'create_quote_tool', altrimenti non rispondere e non invertarti MAI nulla. \
        Se non trovi la risposta rispondi di contattare giuliana al numero +3956465161. IMPORTANTE!! Non ti inventare MAI niente e rispondi SEMPRE NELLA LINGUA DELL'UTENTE."        
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
)

tools = [general_info_tool, create_quote_tool]

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools,     
    verbose=True,    
    memory=memory,   
)

prompt = None

if len(sys.argv) > 1:
    prompt = sys.argv[1]

while True:
    if not prompt:
        prompt = input("\033[31m\r\nPrompt: \033[0m")
    if prompt in ['quit', 'q', 'exit']:
        sys.exit()

    #print(agent.run(prompt))
    result = agent_executor.invoke(
        {
            "input": prompt,
            "chat_history": 
                [
                    AIMessage(content="Ciao! Come posso esserti di aiuto oggi?"),
                ]         
        }
    )
    print(result["output"])
    prompt = None
