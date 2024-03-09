import asyncio
import os
from langchain import hub
from langchain_elasticsearch import ElasticsearchStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings,HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

os.environ["OPENAI_API_KEY"] = "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"
path = '/rag-paper/data/txt'
text_loader_kwargs = {'autodetect_encoding': True}
loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()

#define text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=30,
    length_function=len,
)

doc_list = []
# for doc in docs:
#     tmp_docs = text_splitter.create_documents([doc.page_content])
#     doc_list += tmp_docs
#通过生成元数据区别用户权限
user1_doc = docs[0]
user1_doc_split_data = text_splitter.create_documents([user1_doc.page_content])
for i, doc in enumerate(user1_doc_split_data):
    doc.metadata["author"] = ["userid_001"]
    doc_list.append(doc)
user2_doc = docs[1]
user2_doc_split_data = text_splitter.create_documents([user2_doc.page_content])
for i, doc in enumerate(user2_doc_split_data):
    doc.metadata["author"] = ["userid_002"]
    doc_list.append(doc)

embedding_model = OpenAIEmbeddings(openai_api_key="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")
# embedding_model = HuggingFaceEmbeddings(
#     model_name="/Users/titus.w/Code/med-GLM/rag-paper/model/embedding_model/m3e-base",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": True}
# )
# embedding_model = HuggingFaceBgeEmbeddings(
#     model_name="/Users/titus.w/Code/med-GLM/rag-paper/model/embedding_model/bge-large-zh-v1.5",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": True}
# )
embedding_dim = len(embedding_model.embed_query('i love you'))
print(f"embedding模型的维度是：{embedding_dim}")


vector_database = ElasticsearchStore(
    es_url='http://localhost:9200',
    index_name='index_sd_1536_vectors',
    embedding=embedding_model,
    es_user='elastic',
    vector_query_field='query_vectors',
    #es_password='<PASSWORD>'
)
# vector_database.add_documents(doc_list)



#各种检索器
retriever = vector_database.as_retriever(search_type="similarity", search_kwargs={"k": 6})
#只检索关键字段
user_id = "userid_001"
filter_criteria = [{"term": {"metadata.author.keyword": user_id}}]
special_retriever = vector_database.as_retriever(search_kwargs={'filter': filter_criteria})



#各种prompt
#(context, question) rag_prompt
prompt = hub.pull('rlm/rag-prompt')
#(context, question) custom_rag_prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)
#多轮对话prompt，第一轮和后面几轮用的提示词不同
#(history, question) multi_qa_rag_prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
#(history, context, question) qa_frist_round_rag_prompt
multi_qa_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", multi_qa_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)




#各种大模型
llm = ChatOpenAI( model="gpt-3.5-turbo", max_tokens=1024)
# llm = HuggingFacePipeline.from_model_id(
#     model_id="/Users/titus.w/Code/model/Llama-2-7b-chat-hf",
#     task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 10},
# )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)




#乱七八糟各种方法
def stream_output(chain, query):
    #input: chain, query
    output = {}
    curr_key = None
    for chunk in chain.stream(query):
        for key in chunk:
            if key not in output:
                output[key] = chunk[key]
            else:
                output[key] += chunk[key]
            if key != curr_key:
                print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
            else:
                print(chunk[key], end="", flush=True)
            curr_key = key


async def print_chain(rag_chain, query_text, chat_history):
    ct = 0
    async for chunk in rag_chain.astream_log(
            {"question": query_text, "chat_history": chat_history},
            # include_tags=["contextualize_q_chain"],
    ):
        print(chunk)
        print("\n" + "-" * 30 + "\n")
        ct += 1
        if ct > 100: break






#各种链条
#<string> main_chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)


#<json> mian_chain (added sources)
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain_with_source = RunnableParallel(
    {"context": special_retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


#sub_chain (multi qa)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]
#<key-value> main_chain
multi_qa_rag_chain = (
    RunnablePassthrough.assign(
    # if history exists, mianchchain use subchain to rebuild the question
        context=contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
)






go_on = True
multi_qa = True
chat_history = []
while go_on:
    query_text = input("你的问题: ")
    if 'exit' in query_text:
        break
    print("AI需要回答的问题 [{}]\n".format(query_text))
    if multi_qa == True:
        ai_msg = multi_qa_rag_chain.invoke({"question": query_text, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=query_text), ai_msg])
        print(ai_msg)
        # asyncio.run(print_chain(multi_qa_rag_chain, query_text, chat_history))
    else:
        res = rag_chain_with_source.invoke(query_text)
        print(res)
        #stream_output(rag_chain_with_source, query_text)












pass
