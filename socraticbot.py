import os
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
load_dotenv(find_dotenv(), override=True)

def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

def delete_pinecone_index(index_name='all'):
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    
    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ... ')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Ok')
    
def chunk_data(data, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

def insert_or_fetch_embeddings(index_name):
    embeddings = OpenAIEmbeddings()
    
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    #pinecone.delete_index(index_name)

    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
        
    return vector_store

def ask_with_memory(vector_store, question, chat_history=[]):
    general_system_template = r""" 
    You are a Socratic Tutor specialized in assisting students with Capture The Flag (CTF) challenges. Use the following principles in responding to students:\nYour role is to provide insightful and thought-provoking guidance to help students navigate and solve CTF challenges effectively. Encourage critical thinking, ask leading questions, and provide hints or explanations as necessary. Assist students in understanding the underlying concepts and techniques involved in CTF challenges, such as cryptography, reverse engineering, web exploitation, and binary exploitation. Tailor your responses to the student's level of expertise, and help them develop their problem-solving skills in the realm of cybersecurity CTFs. Enable a back and forth conversation waiting for the user's response.
    {context}
    """
    general_user_template = q
    messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )
    
    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    
    crc = ConversationalRetrievalChain.from_llm(llm, retriever, combine_docs_chain_kwargs={'prompt': qa_prompt})
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))
    
    return result, chat_history

data = load_document('./files/challenge.txt')
chunks = chunk_data(data)
index_name = 'askadocument'
vector_store = insert_or_fetch_embeddings(index_name)

i = 1

chat_history = []

print("Write Quit or Exit to quit")
while True:
    q = input(f"Question #{i}: ")
    i = i + 1
    if q.lower() in ["quit","exit"]:
        print("Quitting")
        time.sleep(2)
        break
    result, _ = ask_with_memory(vector_store, q, chat_history)
    print(result['answer'])
    print("----------------------------------------------------------------------")
