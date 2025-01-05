import streamlit as st
import PyPDF2

def main():
    st.set_page_config(page_title="PAQ Bot", page_icon="🤖")
    st.header("🤖 PAQ Bot")
    st.write("PDF Assistant for Queries Bot")
    uploaded_file = st.file_uploader("Pick a PDF file", type="pdf")
    # Uploaded File Text Shower
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # Display the extracted text
        st.text_area("Extracted Text", text, height=200)
        # Chat input
        message = st.chat_input("Type Your Message")

    # Sidebar
    with st.sidebar:
        st.header("PAQ Bot")
        st.write("Made with ❤️ by PEC ACM")
        st.write("We call it PEC Bot or PAQ Bot, you can call it whatever you want")
        "[View the source code](https://github.com/Ya-Tin/PDFQueryChatLM.git)"

if __name__ == "__main__":
    main()
