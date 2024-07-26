

import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import tempfile
from langchain_community.document_loaders import PyPDFLoader
import os

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. 환경 변수를 확인해주세요.")
    st.stop()

def main():
    st.title("Yeo's PDF챗팅앱")
    
    # 파일 업로더를 메인 영역으로 이동
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        process_uploaded_file(uploaded_file)
    else:
        st.write("PDF 파일을 업로드해주세요.")

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        
    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectors = FAISS.from_documents(data, embeddings)
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key),
        retriever=vectors.as_retriever()
    )
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["안녕하세요! " + uploaded_file.name + "에 관해 질문주세요."]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["안녕하세요!"]
        
    # 챗봇 대화 UI
    response_container = st.container()
    container = st.container()
    
    with container:
        with st.form(key='Conv_Question', clear_on_submit=True):
            user_input = st.text_input("질문:", placeholder="PDF파일에 대해 얘기해볼까요?", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input, chain)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], 
                        is_user=True, 
                        key=str(i) + '_user',
                        avatar_style="avataaars",
                        seed="Nala")
                message(st.session_state["generated"][i], 
                        key=str(i),
                        avatar_style="avataaars",
                        seed="Midnight")

def conversational_chat(query, chain):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

if __name__ == "__main__":
    main()    