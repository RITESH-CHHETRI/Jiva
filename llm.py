import os
import sys
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import Replicate
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain


os.environ['REPLICATE_API_TOKEN'] = ""

pineconekey=""
pc = Pinecone(api_key=pineconekey)

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

chat_history = []
while True:
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))
