import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain.llms import OpenAI
from langchain_cohere import CohereEmbeddings
from langchain_cohere import CohereEmbeddings
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter



load_dotenv()

# llm = GoogleGenerativeAI(model="models/text-bison-001")
# llm = ChatCohere()
llm = OpenAI()

arabic = r"banque_AR.pdf"
french = r"Banque_FR.pdf"

# put all the urls of our website to scrape it
urls = [
    'https://fsciences.univ-setif.dz/main_page/french',
    'https://fsciences.univ-setif.dz/GP_FR/IngInfoTC.html',
    'https://fsciences.univ-setif.dz/GP_FR/Lincence_Informatique.html',
    'https://fsciences.univ-setif.dz/GP_FR/CS.html'
]

def pdf_text_extract(file):
    text = ""
    pdf_reader1 = PdfReader(file)
    for page in pdf_reader1.pages:
        text += page.extract_text()
    return text

def web_text_extract(urls):
  webText = ""
  loader = WebBaseLoader(urls)
  documents = loader.load() 
  for doc in documents:
      webText += str(doc)

  return webText

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore


st.set_page_config(page_title="Streaming bot", page_icon="🤖")
st.title("BuddyBot 🤖")

st.header("Chat with BuddyBot")
user_query = st.chat_input("Enter your question...")

# initialize the chat history with a hello from AI
if "conversation" not in st.session_state:
    st.session_state.conversation = [
        AIMessage(content="Hello, I am BuddyBot. Ask me anything about the faculty")
    ]

# create the conversation
for message in st.session_state.conversation:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

if user_query:
    # push the user input into the history
    st.session_state.conversation.append(HumanMessage(content=user_query))
    # show the humain input 
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        with st.spinner("Loading..."):

            # extract the text from pdf files
            arabicText = pdf_text_extract(arabic)
            frenchText = pdf_text_extract(french)
            # concatenate the two files
            document = frenchText + arabicText

            # extract the text from website
            webText = web_text_extract(urls)
            # concatinate the pdf text with web text
            document += webText

            # split text into chunks
            chunks = get_text_chunks(document)

            # get the vectore store
            vectore = vectorstore(chunks)

            # setup the prompt template (to make the user input better for the LLM)
            template = """
            You are a helpful assistant. Answer the following questions based on the given document:

            Document: {doc}

            User question: {user_question}
            """
            # create the prompt
            prompt = ChatPromptTemplate.from_template(template)

            # create a output parser
            output_parser = StrOutputParser()

            # combine the llm, promtpt and the output parser into a chain 
            chain = prompt | llm | output_parser

            # search the most ranked data for our question
            docs = vectore.similarity_search(query=user_query, k=1)

            # we give the user question and the doc to the chain
            answer = st.write_stream(chain.stream({
            "doc": docs,
            "user_question": user_query
            }))

            # push the response into the history
            st.session_state.conversation.append(AIMessage(content=answer))
