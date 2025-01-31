# RAG_Chatbot_Using_LangChain_And_DeepSeek
This is a practice project that uses RAG + Reasoning to read PDFs uploaded by the users and answer any questions about them. 
We make use of LangChains and DeepSeek R1 7B model to achieve this. The UI is generated using Streamlit.

Once the user uploads a PDF file, the program loads it and splits the text data. 
The data is converted to vectors.
When the user posts a question on the uploaded PDF, the model analyses the PDF and reasons before giving an answer. 

To run:
- Pull the DeepSeek R1 model using Ollama
    Reference - https://ollama.com/library/deepseek-r1

- Install all the dependencies from the requirements.txt file
    pip install -r requirements.txt

- Run the python file
    streamlit run rag_chatbot.py
