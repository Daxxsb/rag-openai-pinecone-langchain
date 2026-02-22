import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA


def main():
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([pinecone_api_key, pinecone_index_name, openai_api_key]):
        raise ValueError("Missing required environment variables.")

    # Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )

    # Connect to existing index
    vectorstore = PineconeVectorStore(
        index_name=pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=openai_api_key,
        temperature=0
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Use the following context to answer the question.
        If you don't know the answer, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    print("RAG System Ready. Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ")

        if question.lower() == "exit":
            break

        try:
            response = qa_chain.run(question)
            print("\nAnswer:")
            print(response)
            print("\n" + "="*50 + "\n")

        except Exception as e:
            print("\nError occurred:")
            print(e)
            print("\n(This is expected if quota is exceeded.)\n")


if __name__ == "__main__":
    main()