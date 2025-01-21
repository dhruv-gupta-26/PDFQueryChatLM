import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from htmltemplates import css, bot_template, user_template
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def chonky(text):
    text_splitter= CharacterTextSplitter(separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():

    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n

    If the answer is not available in the context, just try to give an answer based on the internet data

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Append to the session state to maintain chat history
    st.session_state["messages"].append({"role": "user", "content": user_question})
    st.session_state["messages"].append({"role": "assistant", "content": response["output_text"]})

def main():
    st.set_page_config(page_title="PAQ Bot", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    st.header("🤖 PAQ Bot")

    # Display chat messages
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat input box
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        user_input(user_question)

    # Sidebar
    with st.sidebar:
        st.header("PAQ Bot")
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Pick a PDF file", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing"):
                # Get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # Get the text chunks
                text_chunks = chonky(raw_text)
                # Create the vector store
                vector_store = get_vectorstore(text_chunks)
                # Notify user
                st.session_state["file_processed"] = True  # Set a flag in session state
                st.success("Done")
        if not pdf_docs and "file_processed" not in st.session_state:
            st.info("Please upload a PDF file to start.")
        st.write("Made with ❤️ by PEC ACM")
        st.write("We call it PEC Bot or PAQ Bot, you can call it whatever you want")
        "[View the source code](https://github.com/Ya-Tin/PDFQueryChatLM.git)"

if __name__ == "__main__":
    main()
