import os
import sys
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import Replicate
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Twilio WhatsApp bot setup
app = Flask(__name__)
chat_history = []
conversation_chain = None

# Your API tokens and keys
os.environ['REPLICATE_API_TOKEN'] = ""

pineconekey = ""
pc = Pinecone(api_key=pineconekey)

# Loading and processing PDF document
loader = PyPDFLoader('data.pdf')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

index_name = "jiva"
"""
pc.create_index(
    name=index_name,
    dimension=768,
    metric='cosine',
    spec=ServerlessSpec(
        cloud='AWS',
        region='us-east-1'
    )
)
"""
index = pc.Index(index_name=index_name)
vectordb = LC_Pinecone.from_documents(texts, embeddings, index_name=index_name)

llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75, "max_length": 3000}
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)

def answ(incoming_msg):
    global chat_history, qa_chain
    result = qa_chain({'question': incoming_msg, 'chat_history': chat_history})
    answer = result['answer']
    chat_history.append((incoming_msg, answer))
    return answer

@app.route('/bot', methods=['POST'])
def bot():
    global chat_history, qa_chain
    print("response")
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    if incoming_msg == "hi":
        chat_history = []
    reply = answ(incoming_msg)
    msg.body(reply)
    return str(resp)

if __name__ == '__main__':
    app.run(debug=True)
