from flask import Flask, render_template, request

app = Flask(__name__)

# existing code for setup and functionality
import json
import os
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index import ServiceContext, set_global_service_context
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import GradientEmbedding
from llama_index.llms import GradientBaseModelLLM
from llama_index.vector_stores import CassandraVectorStore
from copy import deepcopy
from tempfile import NamedTemporaryFile

os.environ['GRADIENT_ACCESS_TOKEN'] = "sevG6Rqb0ztaquM4xjr83SBNSYj91cux"
os.environ['GRADIENT_WORKSPACE_ID'] = "4de36c1f-5ee6-41da-8f95-9d2fb1ded33a_workspace"

def create_datastax_connection():
    cloud_config = {
        'secure_connect_bundle': 'secure-connect-temp-db.zip'
    }

    with open("temp_db-token.json") as f:
        secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    astra_session = cluster.connect()
    return astra_session

llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)

embed_model = GradientEmbedding(
    gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
    gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
    gradient_model_slug="bge-large"
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=256
)

set_global_service_context(service_context)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

# Flask route for the chat functionality
@app.route('/')
def chat():
    return render_template('index.html')

@app.route('/process_chat', methods=['POST'])
def process_chat():
    prompt = request.form['prompt']

    if prompt.lower() == 'exit':
        return "Chat exited."

    pdf_response = query_engine.query(prompt)
    cleaned_response = pdf_response.response

    # Split the cleaned response into paragraphs
    paragraphs = cleaned_response.split('\n')

    # Pass paragraphs to the HTML template for rendering
    return render_template('assistant_response.html', paragraphs=paragraphs)


if __name__ == '__main__':
    app.run(debug=True)
