import os
import base64
import open_clip
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from typing import List
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
import chromadb
import numpy as np
from langchain_community.vectorstores import Chroma



text_path = '/rag-paper/data/_mixed_data_project/X-ray/txt'
image_path = '/rag-paper/data/_mixed_data_project/X-ray/png'
os.environ["OPENAI_API_KEY"] = "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"


import base64
import io
from io import BytesIO

import numpy as np
from PIL import Image

from IPython.display import HTML, display


def plt_img_base64(img_base64):
    # 解码Base64字符串
    img_bytes = base64.b64decode(img_base64)

    # 将字节数据转换为PIL图像对象
    img = Image.open(io.BytesIO(img_bytes))

    # 显示图片
    img.show()
def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string.

    Args:
    base64_string (str): Base64 string of the original image.
    size (tuple): Desired size of the image as (width, height).

    Returns:
    str: Base64 string of the resized image.
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False
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
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            # Resize image to avoid OAI server error
            images.append(
                resize_base64_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            text.append(doc)
    return {"images": images, "texts": text}
# CLIP model

class ClipEmbedding():
    def __init__(self, clip_model, clip_tokenizer, clip_preprocess):
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.clip_preprocess = clip_preprocess

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text_features = []
        for text in texts:
            # Tokenize the text
            tokenized_text = self.clip_tokenizer(text)

            # Encode the text to get the embeddings
            embeddings_tensor = self.clip_model.encode_text(tokenized_text)

            # Normalize the embeddings
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)

            # Convert normalized tensor to list and add to the text_features list
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()
            text_features.append(embeddings_list)

        return text_features

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        try:
            from PIL import Image as _PILImage
        except ImportError:
            raise ImportError("Please install the PIL library: pip install pillow")

        # Open images directly as PIL images
        pil_images = [_PILImage.open(uri) for uri in uris]

        image_features = []
        for pil_image in pil_images:
            # Preprocess the image for the model
            preprocessed_image = self.clip_preprocess(pil_image).unsqueeze(0)

            # Encode the image to get the embeddings
            embeddings_tensor = self.clip_model.encode_image(preprocessed_image)

            # Normalize the embeddings tensor
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)

            # Convert tensor to list and add to the image_features list
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()

            image_features.append(embeddings_list)

        return image_features


clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                       cache_dir='/Users/titus.w/Code/model/CLIP-ViT'
                                                                                 '-B-16-DataComp.L-s1B-b8K')
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_embedding = ClipEmbedding(clip_model, clip_tokenizer, clip_preprocess)
print(f"embedding模型的维度是：{len(clip_embedding.embed_documents(['i love you'])[0])}")


text_loader_kwargs = {'autodetect_encoding': True}
loader = DirectoryLoader(text_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()
docs = sorted(docs, key=lambda x: int(x.metadata['source'].split('/')[-1].replace('.txt', '')))
doc_list = []
for doc in docs:
    tmp_docs = [doc.page_content]
    doc_list += tmp_docs
image_list = sorted(
    [
        os.path.join(image_path, image_name)
        for image_name in os.listdir(image_path)
        if image_name.endswith(".png")
    ]
)
image_list = sorted(
    image_list,
    key=lambda x: int(x.split('/')[-1].replace('.png', ''))
)
doc_vector_list = clip_embedding.embed_documents(doc_list)
image_vector_list = clip_embedding.embed_image(image_list)


# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# img_features_np = np.array(image_vector_list)
# text_features_np = np.array(doc_vector_list)
#
# # 设置支持中文的字体，这里以微软雅黑为例
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
# # Calculate similarity
# similarity = np.matmul(text_features_np, img_features_np.T)
#
# # Plot
# count = len(doc_list)
# plt.figure(figsize=(20, 14))
# plt.imshow(similarity, vmin=-0.1, vmax=0.1)
# # plt.colorbar()
# plt.yticks(range(count), doc_list, fontsize=18)
# plt.xticks([])
# original_images = []
# for i in range(len(image_list)):
#     image = Image.open(image_list[i]).convert("RGB")
#     original_images.append(image)
# for i, image in enumerate(original_images):
#     plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
# for x in range(similarity.shape[1]):
#     for y in range(similarity.shape[0]):
#         plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
#
# for side in ["left", "top", "right", "bottom"]:
#     plt.gca().spines[side].set_visible(False)
#
# plt.xlim([-0.5, count - 0.5])
# plt.ylim([count + 0.5, -2])
# plt.title("Cosine similarity between text and image features", size=20)
# plt.savefig('/Users/titus.w/Code/med-rag-paper/rag-project/data/runs/sim.png', dpi=300, bbox_inches='tight')




# vector_database = ElasticsearchStore(
#     es_url='http://localhost:9200',
#     index_name='openclip_512_vectors',
#     embedding=clip_embedding,
#     es_user='elastic',
#     vector_query_field='query_vectors',
#     # es_password='<PASSWORD>'
# )
# vector_database.add_documents(docs)
vector_database = Chroma(
    collection_name="mm_rag_clip_photos", embedding_function=clip_embedding
)
vector_database.add_images(uris=image_list)
vector_database.add_texts(texts=doc_list)

retriever = vector_database.as_retriever(search_kwargs={"k": 15})

def prompt_func(data_dict):
    # Joining the context texts into a single string
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                # "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
                "url": f"data:image/jpeg;base64,{data_dict['context']['images']}"
            },
        }
        messages.append(image_message)

    # Adding the text message for analysis
    text_message = {
        "type": "text",
        "text": (
            "As an expert art critic and historian, your task is to analyze and interpret images, "
            "considering their historical and cultural significance. Alongside the images, you will be "
            "provided with related text to offer context. Both will be retrieved from a vectorstore based "
            "on user-input keywords. Please use your extensive knowledge and analytical skills to provide a "
            "comprehensive summary that includes:\n"
            "- A detailed description of the visual elements in the image.\n"
            "- The historical and cultural context of the image.\n"
            "- An interpretation of the image's symbolism and meaning.\n"
            "- Connections between the image and the related text.\n\n"
            f"User-provided keywords: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)

    return [HumanMessage(content=messages)]

mllm = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)
chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(prompt_func)
    | mllm
    | StrOutputParser()
)

#text input, 发现召回前几个都是文本(如果添加一个id元数据绑定起来就可以了)，但是还是希望最好能直接召回图片
docs = retriever.get_relevant_documents("肺野太小了")
for doc in docs:
    if is_base64(doc.page_content):
        plt_img_base64(doc.page_content)
    else:
        print(doc.page_content)
print(chain.invoke("肺野太小了"))
# #2 输入是图片，langchai没有提供现成的api，只能直搜数据，当然召回的图片信息base64编码不要放在prompt里，只是召回显示看一下。
# docs = retriever.get_relevant_documents(image_list[0])



