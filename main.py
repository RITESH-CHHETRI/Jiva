from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, render_template
from twilio.twiml.messaging_response import MessagingResponse
import requests

app = Flask(__name__)


def get_pdf_text(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(question, conversation_chain):
    response = conversation_chain({'question': question})
    chat_history = response['chat_history']
    return chat_history


load_dotenv()

pdf_file = "data.pdf"
raw_text = get_pdf_text(pdf_file)
text_chunks = get_chunk_text(raw_text)
vector_store = get_vector_store(text_chunks)
print("Vector store created.")
conversation_chain = get_conversation_chain(vector_store)

def answ(question):
    global chat_history, conversation_chain

    chat_history = handle_user_input(question, conversation_chain)
    print(chat_history)
    message = chat_history[-1]
    return message.content


@app.route('/bot', methods=['POST'])
def bot():
    global chat_history, conversation_chain
    print("response")
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    if incoming_msg == "hi":
        chat_history = []
    reply = answ(incoming_msg)
    msg.body(reply)
    return str(resp)

@app.route('/api', methods=['POST'])
def api():
    global chat_history, conversation_chain
    data = request.json
    question = data['question']
    response = handle_user_input(question, conversation_chain)
    return response

@app.route('/docs', methods=['GET'])
def docs():
    return render_template('api.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['GET','POST'])
def chat():
    global chat_history, conversation_chain
    if request.method == 'GET':
        chat_history = []
        return render_template('chat.html', chat_history=chat_history)
    question = request.form['question']
    chat_history = handle_user_input(question, conversation_chain)
    return render_template('chat.html', chat_history=chat_history)

if __name__ == "__main__":
    app.run()