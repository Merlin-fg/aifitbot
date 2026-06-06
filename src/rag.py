import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 1. 文档加载与切分
def load_and_split_documents(data_dir: str = "data"):
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " "]
    )
    return text_splitter.split_documents(documents)

# 2. 向量存储（持久化）
def get_vectorstore(embedding_model: str = "all-MiniLM-L6-v2"):
    persist_dir = "chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if os.path.exists(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        docs = load_and_split_documents()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    return vectorstore

# 3. 构建 RAG 链（底层组装，不依赖 langchain.chains）
def create_rag_chain(llm):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 系统提示模板（包含上下文）
    system_prompt = (
        "你是一位专业的私人健身教练与营养师。根据以下已知的健身知识片段回答用户问题。"
        "如果知识片段不足以回答，可以结合你自身的专业知识补充，但请明确指出。"
        "\n\n参考知识片段：{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 将检索到的文档列表格式化为一个上下文字符串
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 手动组装链：检索 → 格式化文档 → 填充 prompt → LLM → 提取文本
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain