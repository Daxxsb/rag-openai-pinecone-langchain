import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter


def main():
    # Load environment variables
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Validate environment variables
    if not all([pinecone_api_key, pinecone_index_name, openai_api_key]):
        raise ValueError("Missing required environment variables.")

    # Validate data folder
    data_dir = "data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError("The 'data' folder does not exist.")

    # Load .txt documents
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())

    if not docs:
        raise ValueError("No .txt files found in the data folder.")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = []
    for doc in docs:
        texts.extend(splitter.split_text(doc))

    # Set up OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )

    # Store vectors in Pinecone
    PineconeVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        index_name=pinecone_index_name,
        pinecone_api_key=pinecone_api_key
    )

    print("======================================")
    print(f"Documents processed: {len(docs)}")
    print(f"Chunks stored: {len(texts)}")
    print(f"Index used: {pinecone_index_name}")
    print("Ingestion complete.")
    print("======================================")


if __name__ == "__main__":
    main()