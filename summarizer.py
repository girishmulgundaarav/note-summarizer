import os
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

def llm_summarize_stream(text: str, image_texts: list[str] = None, model: str = "gpt-5-mini"):
    """
    Stream summary text chunk by chunk using the selected model.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, streaming=True)

    if image_texts:
        text += "\n\n" + "\n".join(image_texts)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = splitter.create_documents([text])

    prompt = PromptTemplate.from_template(
        "Summarize the following notes (including image descriptions) into concise bullet points:\n\n{input_text}"
    )

    chain = RunnableSequence(prompt | llm | StrOutputParser())

    # Return a generator that yields chunks
    def stream_summary():
        for d in docs:
            for chunk in chain.stream({"input_text": d.page_content}):
                yield chunk

    return stream_summary