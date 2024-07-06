import streamlit as st
import pinecone
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_community.vectorstores import Pinecone as LangChainPinecone

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'chunks' not in st.session_state:
    st.session_state['chunks'] = None
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None


def delete_pinecone_index(index_name='all'):
    pc = Pinecone()
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print(f"Deleting all indexes ...")
        for index in indexes:
            pc.delete_index(index)
        print("done")
    else:
        print(f"Deleting index {index_name} ...")
        pc.delete_index(index_name)
        print("done")


def chunk_data(data, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, length_function=len)
    chunks = text_splitter.split_documents(data)
    return chunks


def load_document(file):
    print("Loading PDF document: " + file)
    loader = PyPDFLoader(file)
    data = loader.load()
    return data


def insert_or_fetch_embeddings(index_name, chunks):
    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well

    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # creating the index and embedding the chunks into the index
        print(f'Creating index {index_name} and embeddings ...', end='')

        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        vector_store = LangChainPinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    results = vector_store.similarity_search_with_score(q)

    if not results or results[0][1] < 0.3:
        llm = ChatOpenAI(model='gpt-4', temperature=1)  # Updated model
        return llm.invoke(q).content
    else:
        llm = ChatOpenAI(model='gpt-4', temperature=0.6)  # Updated model
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        answer = chain.invoke(q)
        return answer['result']


def text_to_text_gpt(text):
    if st.session_state['data'] is None:
        delete_pinecone_index()
        st.session_state['data'] = load_document("about_us.pdf")
        st.session_state['chunks'] = chunk_data(st.session_state['data'], 256)
        st.session_state['vector_store'] = insert_or_fetch_embeddings("rag", st.session_state['chunks'])

    result = ask_and_get_answer(st.session_state['vector_store'], text)
    return result
