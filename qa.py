# qa.py
import os

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_llm_retriever(text: str, image_texts: list[str] = None):
    """
    Build a retriever + QA pipeline using GPT‑5-mini, including image descriptions.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if image_texts:
        text += "\n\n" + "\n".join(image_texts)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=api_key)

    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableSequence

    prompt = PromptTemplate.from_template(
        "Use the following context (including image descriptions) to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    chain = RunnableSequence(prompt | llm | StrOutputParser())

    def qa_chain(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        return chain.invoke({"context": context, "question": question})

    return qa_chain