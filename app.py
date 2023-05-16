import openai
import os

from flask import Flask, jsonify, request

import PyPDF2
import requests
import uuid

from llama_index import SimpleDirectoryReader, StorageContext, load_index_from_storage, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from llama_index.node_parser import SimpleNodeParser

app = Flask(__name__)

# Helper method to build an index given a list of PDF url's

def build_index(listOfUrls, api_key, session_id):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    if (not listOfUrls or not session_id):
        return {}

    listOfFiles = []

    # Save all the files uploaded to disk so we can index them in the next step.
    for oneFileUrl in listOfUrls:
        url = 'https:' + oneFileUrl
        fileIdentifier = uuid.uuid4()
        fileName = str(fileIdentifier) + ".pdf"
        response = requests.get(url)

        with open(fileName, "wb") as file:
            file.write(response.content)
        
        listOfFiles.append(fileName)

    os.environ["OPENAI_API_KEY"] = api_key

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(input_files=listOfFiles).load_data()
    
    parser = SimpleNodeParser()

    nodes = parser.get_nodes_from_documents(documents)

    index = GPTVectorStoreIndex(nodes, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # Store each user's index on disk for querying later
    index.storage_context.persist(persist_dir='./storage/' + session_id)

    # Delete all the downloaded files now that we've indexed them
    for file in listOfFiles:
        os.remove(file)

    return index

@app.route('/construct_index/', methods=['GET', 'POST'])
def construct_index():
    api_key = request.args.get('api_key')
    listOfUrls = request.args.get('fileurls').split(", ")
    session_id = request.args.get('session_id')
    
    if (not listOfUrls or not session_id or not api_key):
        return {}
    
    build_index(listOfUrls, api_key, session_id)
    return {}

@app.route('/query_index/', methods=['GET', 'POST'])
def query_index():

    os.environ["OPENAI_API_KEY"] = request.args.get('api_key')
    session_id = request.args.get('session_id')
    question = request.args.get('question')

    if (not question or not session_id):
        return {}

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir='./storage/' + session_id)
    # load index
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()

    response = query_engine.query(question)

    return jsonify(message=response)


@app.route('/query/', methods=['GET', 'POST'])
def query():

    os.environ["OPENAI_API_KEY"] = request.args.get('api_key')

    question = request.args.get('question')

    listOfUrls = request.args.get('fileurls').split(", ")

    text = ''

    if (request.args.get('fileurls') == '[]'):
        return {}

    #Extract all the text from the PDF's and concatenate it into a string
    for oneFileUrl in listOfUrls:
        url = 'https:' + oneFileUrl

        response = requests.get(url)

        with open("file.pdf", "wb") as file:
            file.write(response.content)

        #Open the PDF file in read-binary mode
        with open("file.pdf", "rb") as file:
            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:
                text += page.extract_text()

        os.remove("file.pdf")
    
    prompt = question + "Only use the context given after the word 'CONTEXT:' to answer the question in the text before this sentence.: "
    prompt = prompt + 'CONTEXT: ' + text
 
    params = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    # Call the OpenAI API to generate a response
    response = openai.Completion.create(engine="text-davinci-003", **params)

    # Print the response text
    print(response.choices[0].text.strip())

    return jsonify(message=response)

@app.route('/')
def initial():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    app.run(debug=True)
