import uuid
import base64
from IPython.display import HTML, display
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.messages import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from summary_generate_img_tab_text import texts, text_summaries, table_summaries, tables, image_summary_list, image_data_list


def plt_img_base64(img_base64):
    display(HTML(f''))

def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content

        if is_image_data(doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Adding texts to the messages
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    text_message = {
        "type": "text",
        "text": (
            "You are financial analyst.\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to answer the user question in the finance. \n"
            f"Question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


id_key = "doc_id"
# embeddings_model = QianfanEmbeddingsEndpoint(model='bge-base-zh-v1.5', endpoint='bge-base-zh-v1.5')
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings()),
    docstore=InMemoryStore(),
    id_key=id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts))) #MultiVectorRetriever
# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))
# Add images
img_ids = [str(uuid.uuid4()) for _ in image_data_list]
summary_images = [
    Document(page_content=s, metadata={id_key: img_ids[i]})
    for i, s in enumerate(image_summary_list)
]
retriever.vectorstore.add_documents(summary_images)
retriever.docstore.mset(list(zip(img_ids, image_data_list)))
# rag-paper pipeline
model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)
chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(img_prompt_func)
    | model
    | StrOutputParser()
)

query = "Which year had the highest holiday sales growth?"
print(chain.invoke(query))
docs = retriever.get_relevant_documents(query)
print(is_image_data(docs[1]))
print(plt_img_base64(docs[1]))
