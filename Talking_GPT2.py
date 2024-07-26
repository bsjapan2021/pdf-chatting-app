import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import tempfile
from langchain_community.document_loaders import PyPDFLoader
import os

# Streamlit Secrets에서 API 키 가져오기
try:
    openai_api_key = st.secrets["openai"]["api_key"]
except KeyError:
    openai_api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")

if not openai_api_key:
    st.error("OpenAI API 키가 필요합니다. Streamlit Secrets에서 설정하거나 위 필드에 입력해주세요.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

def main():
    st.title("Yeo's PDF챗팅앱")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    
    if uploaded_file:
        try:
            with st.spinner("PDF 처리 중..."):
                chain = process_uploaded_file(uploaded_file)
            if chain:
                st.success("PDF가 성공적으로 처리되었습니다. 이제 질문할 수 있습니다.")
            else:
                st.error("PDF 처리 중 오류가 발생했습니다.")
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            st.exception(e)
    
    st.subheader("채팅")
    user_input = st.text_input("질문을 입력하세요:", key="user_input")
    if st.button("전송"):
        if 'chain' in locals() and chain:
            try:
                response = conversational_chat(user_input, chain)
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", response))
            except Exception as e:
                st.error(f"채팅 처리 중 오류 발생: {str(e)}")
                st.exception(e)
        else:
            st.warning("먼저 PDF 파일을 업로드해주세요.")
    
    for i, (role, text) in enumerate(st.session_state.chat_history):
        message(text, is_user=(role == "user"), key=f"{i}_{role}")

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()
    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo'),
        retriever=vectors.as_retriever()
    )
    return chain

def conversational_chat(query, chain):
    result = chain({"question": query, "chat_history": [(msg[1], msg[2]) for msg in st.session_state.chat_history if msg[0] == "user"]})
    return result["answer"]

if __name__ == "__main__":
    main()
