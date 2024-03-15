import os
import io
import sys
import time
import torch
import base64
import chromadb
import argparse
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from chromadb.utils.data_loaders import ImageLoader
from langchain_elasticsearch import ElasticsearchStore
from langchain.schema.output_parser import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings,HuggingFaceBgeEmbeddings
sys.path.append("../../")
import utils


parser = argparse.ArgumentParser(description='image_rag')
parser.add_argument("--num_images_recall", type=int, default=1, help="num_images_recall")
parser.add_argument("--openai_api_key", type=str, default="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", help="openai_api_kay")
parser.add_argument("--query_image_path", type=str, default='/Users/titus.w/Downloads/WX20240304-163252.png', help="query_image_path")
parser.add_argument("--pair_data_path", type=str, default='/Users/titus.w/Code/med-RAG/rag-project/data/demo_quality_control_pair', help="pair_img_text_data_path")
parser.add_argument("--text_data_path", type=str, default='/Users/titus.w/Code/med-RAG/rag-project/data/demo_quality_control_text', help="only_text_data_path")
parser.add_argument("--embedding_model_path", type=str, default='/Users/titus.w/Code/med-RAG/model_embedding/bge-large-zh-v1.5', help="embedding_model_path")
parser.add_argument("--is_text_database", type=bool, default=True, help="is_text_database")


args = parser.parse_args()
os.environ["OPENAI_API_KEY"] = args.openai_api_key
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    imges_classes = os.listdir(args.pair_data_path)
    print(imges_classes)

    #knowlegde_image_text_pair
    image_paths = []
    text_paths = []
    texts_list = []
    for subdir, dirs, files in os.walk(args.pair_data_path):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(subdir, file))
            elif file.endswith('.txt'):
                text_paths.append(os.path.join(subdir, file))
    image_paths.sort()
    text_paths.sort()
    pair_index = [str(i) for i in range(len(image_paths))]
    from langchain.document_loaders import TextLoader
    for path in text_paths:
        texts_list.append(TextLoader(path, encoding='utf8').load())

    #knowlegde_only_text
    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader(args.text_data_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()  # load file
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=30,
        length_function=len,
    )
    doc_list = []  # text chunk
    for doc in docs:
        tmp_docs = text_splitter.create_documents([doc.page_content])
        doc_list += tmp_docs

    #logical routing
    timestamp = int(time.time())
    db_file = f"DB_{timestamp}"
    chroma_client = chromadb.PersistentClient(path=db_file)
    embedding_function = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()
    images_client = chroma_client.create_collection(
        name='multimodal_collection_images',
        embedding_function=embedding_function,
        data_loader=image_loader)
    images_client.add(ids=pair_index, uris=image_paths)

    embedding_model = HuggingFaceBgeEmbeddings(
        model_name= args.embedding_model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_database = ElasticsearchStore(
        es_url='http://localhost:9200',
        index_name='index_bge_1024_vectors',
        embedding=embedding_model,
        es_user='elastic',
        vector_query_field='query_vectors',
        # es_password='<PASSWORD>'
    )
    #vector_database.add_documents(doc_list) #ignore
    retriever = vector_database.as_retriever(search_type="similarity", search_kwargs={"k": 6})


    #img2img
    query_image = np.array(Image.open(args.query_image_path))
    print("Query Image")
    plt.imshow(query_image)
    plt.axis('off')
    plt.show()
    print("Results")
    retrieved = images_client.query(query_images=[query_image], include=['data'], n_results=args.num_images_recall)
    for img in retrieved['data'][0]:
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    text_from_image = []
    for i in range(args.num_images_recall):
        text_from_image.append(texts_list[int(retrieved['ids'][0][i])][0])
    combined_text = '\n'.join([doc.page_content for doc in text_from_image])

    if args.is_text_database == False:
        template = """使用以下上下文来回答最后的问题。
        您是医学影像方面的专家，尤其是在影像质控领域，能判断出片子是否拍的符合规格，如果不符合需要给出原因。
        你认真分析了用户上传的医学影像。
        如果你不知道答案，就说你不知道，不要试图编造答案。
        最多使用三句话，并尽可能保持答案简洁。
        使用以下检索到的上下文来回答问题，这些内容中包含了质控结果。\
        {context}
        问题: {question}
        有帮助的答案:""".format(question='{question}', context=combined_text)
        custom_rag_prompt = PromptTemplate.from_template(template)
        llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
        rag_chain = (
                {"question": RunnablePassthrough()}
                | custom_rag_prompt
                | llm
                | StrOutputParser()
        )


    base64_encoded_image = utils.image_to_base64(args.query_image_path)
    resized_base64_image = utils.resize_base64_image(base64_encoded_image)

    def split_image_text_types(docs):
        """Split numpy array images and texts"""
        images = []
        text = []
        images.append(
            utils.resize_base64_image(resized_base64_image)
        )
        for doc in docs:
            text.append(doc)
        return {"images": images, "texts": text}

    def prompt_func(data_dict):
        # Joining the context texts into a single string
        formatted_texts = "\n".join([doc.page_content for doc in data_dict["context"]["texts"]])
        messages = []

        # Adding image(s) to the messages if present
        if data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
                    #"url": f"data:image/jpeg;base64,{data_dict['context']['images']}"
                },
            }
            messages.append(image_message)

        # Adding the text message for analysis
        if data_dict["context"]["images"]:
            text_message = {
                "type": "text",
                "text": (
                    "您是医学影像方面的中国专家，尤其是在影像质控领域，能判断出片子是否拍的符合规格"
                    f"你认真分析了用户上传的医学影像,得出了以下这些内容：{combined_text}\n"
                    "上述内容中包含了图片的质控结果，你必须提取出该结果，并在回答的第一句话给出。"
                    "除了图像之外，您还被提供了相关文本以提供上下文。两者都将从基于“向量存储”中检索关于用户输入的关键字得到的。 "
                    "请利用您广泛的知识和分析技能来提取和上述质控结果有关内容的摘要，包括："
                    "- 导致该结果可能的原因.\n"
                    "- 尽可能通顺简洁.\n"
                    "- 你对影像理解后的评价.\n\n"
                    f"用户提出的问题: {data_dict['question']}\n\n"
                    "检索到的文本知识:\n"
                    f"{formatted_texts}"
                ),
            }
        else:
            text_message = {
                "type": "text",
                "text": (
                    "您是医学影像方面的中国专家，尤其是在影像质控领域。"
                    "您还被提供了相关文本以提供上下文。两者都将从基于“向量存储”中检索关于用户输入的关键字得到的。 "
                    "请利用您广泛的知识和分析技能来提取和上述质控结果有关内容的摘要，包括："
                    "- 导致该结果可能的原因.\n"
                    "- 尽可能通顺简洁.\n\n"
                    f"用户提出的问题: {data_dict['question']}\n\n"
                    "检索到的文本知识:\n"
                    f"{formatted_texts}"
                ),
            }
        messages.append(text_message)
        return [HumanMessage(content=messages)]

    #根据图片文本回答
    mllm = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)
    mian_chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(prompt_func)
        | mllm
        | StrOutputParser()
    )


    go_on = True
    while go_on:
        query_text = input("质控AI需要回答的问题: ")
        if 'exit' in query_text:
            break
        if args.is_text_database==False:
            res = rag_chain.invoke(query_text)
        else:
            res = mian_chain.invoke(query_text)
        print(res)
        # stream_output(rag_chain_with_source, query_text)


if __name__ == '__main__':
    main()



