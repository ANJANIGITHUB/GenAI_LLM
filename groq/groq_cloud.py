import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import sys

load_dotenv()

# Reduce the size of the title
st.markdown("<h1 style='font-size: 25px;'>Document Q&A using GROQ & Google Gemini</h1>", unsafe_allow_html=True)


def get_text_from_pdfs(pdf_documents):
    text = ""
    for pdf in pdf_documents:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

groq_id             =st.sidebar.text_input("Enter Your Groq API ID")
gemini_api          =st.sidebar.text_input("Enter Your Google Gemini API ID")


llm=ChatGroq(groq_api_key=groq_id,
             model_name="Llama3-8b-8192")



prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


def vector_embedding():

    if "vectors" not in st.session_state:

        try:
            st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key=gemini_api)
            st.session_state.loader=PyPDFDirectoryLoader("./dir_pdfs") ## Data Ingestion
            st.session_state.docs=st.session_state.loader.load() ## Document Loading
            st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
            st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
            st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings

            # Get the list of file names from the directory
            file_names = [f for f in os.listdir("./dir_pdfs") if f.endswith('.pdf')]

            # Display the file names in Streamlit
            st.sidebar.write("List of PDF files in the directory:")
            for file in file_names:
                st.sidebar.write(file)
        except Exception as e:
            err_msg="Your Credentials were not provided"
            st.sidebar.write(":red[Your Credentials were not provided]")
            sys.exit()



if st.sidebar.button("Documents Embedding"):
    vector_embedding()
    st.sidebar.write(":green[Vector Store DB Is Ready]")

import time


prompt1=st.text_input("Enter Your Question From Documents")

if st.button("Submit"):
    if prompt1:
        if groq_id and gemini_api:
            document_chain=create_stuff_documents_chain(llm,prompt)
            retriever=st.session_state.vectors.as_retriever()
            retrieval_chain=create_retrieval_chain(retriever,document_chain)
            start=time.process_time()
            response=retrieval_chain.invoke({'input':prompt1})
            print("Response time :",time.process_time()-start)
            st.write(response['answer'])

            # With a streamlit expander
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")




