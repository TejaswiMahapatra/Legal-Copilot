import json
import os
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index import ServiceContext, set_global_service_context

os.environ['GRADIENT_ACCESS_TOKEN'] = ""
os.environ['GRADIENT_WORKSPACE_ID'] = ""


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


def main():
    astra_session = create_datastax_connection()

    row = astra_session.execute("select release_version from system.local").one()
    if row:
        print(row[0])
    else:
        print("An error occurred.")

    # Continue with the rest of your code
    from llama_index.llms import GradientBaseModelLLM
    from llama_index.embeddings import GradientEmbedding
    from llama_index.vector_stores import CassandraVectorStore
    from llama_index import SimpleDirectoryReader, VectorStoreIndex

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
    # as_chat_engine() is used for memory to chat with the documents
    query_engine = index.as_query_engine()

    while True:
        prompt = input("Ask your queries (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break

        pdf_response = query_engine.query(prompt)
        cleaned_response = pdf_response.response

        print(f"Assistant: {cleaned_response}")


if __name__ == '__main__':
    main()
