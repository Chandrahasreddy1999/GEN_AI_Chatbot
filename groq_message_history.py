## RAG Q&A conversation with pdf 
import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
## set up streamlight
st.title("Conversational RAG with PDF uploaded and chat history")
st.write("Upload pdf and chat with their content")

## Input the Groq API KEY
api_key=st.text_input("Enter your NVIDIA API key:",type="password")
## check if groq api key is provided
if api_key:
    llm = ChatNVIDIA(
    model="qwen/qwen2.5-coder-32b-instruct",
    api_key=api_key,
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)
    
    ## chat interface
    session_id=st.text_input("Session ID",value="default_session")

    ## state full manage chat history

    if "store" not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("choose a pdf file",type="pdf",accept_multiple_files=True)
    ## Process upload pdfs
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
        
        # split and create embeddings for the documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
        splits=text_splitter.split_documents(documents)
        vectorestore = Chroma.from_documents(documents=splits,embedding=embedding,persist_directory="./chroma_db"  # folder will be created
        )

        retriver=vectorestore.as_retriever()
        
        ## this prompt is for history
        contextulize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood,"
            "without the chat history. Do NOT answer the question,"
            "just reformulate it if needed and otherwise return it as is."
        )

        contextulize_q_prompt=ChatPromptTemplate.from_messages([
            ("system", contextulize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ])

        history_aware_retriver=create_history_aware_retriever(llm,retriver,contextulize_q_prompt)
        ## this prompt is for answer question
        system_prompt=(
            "You are an assistant for question-answering tasks"
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer, say that you"
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt=ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ])

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriver,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input=st.text_input("your question:")

        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke({"input":user_input},
                                                     config={"configurable":{"session_id":session_id}})
            st.write(st.session_state.store)
            st.write("Assistance:",response["answer"])
            st.write("Chat History:",session_history.messages)



