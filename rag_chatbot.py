import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrived context tp answer this question. If you don't know the answer, just say that you don't know. Use 3 sentences maximum and keeo the answer concise.
Question: {question}
Context: {context}
Answer: 
"""

pdfs_directory = './pdfs/'

# Create vectors for text data
embeddings = OllamaEmbeddings(model = "deepseek-r1:7b")
# Stores the vectors in vector store
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model = "deepseek-r1:7b")

# Handle a file uploaded by the user 
def upload_pdf(file):
    with open(pdfs_directory + file.name, 'wb') as f:
        f.write(file.getbuffer())

# Load and extract all the text data from the pdf
# We use LangChain loaders and PDFPlumber to achieve this
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    return documents

# Each page would be a LangChain document which needs to be split 
# because LLM models work better with smaller length text
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )

    return text_splitter.split_documents(documents)


# Text needs to be indexed by converting it to vectors and stored in a VectorStore
def index_docs(documents):
    vector_store.add_documents(documents)

# Chat where users ask questions. We hit the vector store to answer these queries
def retrieve_docs(query):
    return vector_store.similarity_search(query)

# Ask the LLM to answer the user question with the context
def answer_questions(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})


uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documnets = retrieve_docs(question)
        answer = answer_questions(question, related_documnets)
        st.chat_message("assistant").write(answer)


