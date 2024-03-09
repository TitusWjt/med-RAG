import wikipedia
from tqdm import tqdm
import shutil
import os
import chromadb
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from transformers import AutoModel, AutoProcessor
from transformers import AutoTokenizer
import torch
from huggingface_hub import hf_hub_download
from chromadb.config import Settings
#https://github.com/nadsoft-opensource/RAG-with-open-source-multi-modal/blob/main/rag-with-open-source-multi-modal.ipynb

images_pth = '/Users/titus.w/Code/med-rag-paper/rag-project/data/demo_data/png'
imges_classes = os.listdir(images_pth)
new_pth = '/Users/titus.w/Code/med-rag-paper/rag-project/data/demo_quality_control_pair'
if not os.path.exists(new_pth):
    os.mkdir(new_pth)
for cls in tqdm(imges_classes):
    cls_pth = os.path.join(images_pth, cls)
    new_cls_pth = os.path.join(new_pth, cls)
    if not os.path.exists(new_cls_pth):
        os.mkdir(new_cls_pth)
    for img in os.listdir(cls_pth)[:10]: #源数据前10张图片
        img_pth = os.path.join(cls_pth, img)
        new_img_pth = os.path.join(new_cls_pth, img)
        shutil.copy(img_pth, new_img_pth)
print(imges_classes)

# wiki_titles = { # the key is imgs class and the value is wiki title
#     '曝光不良': '曝光不良',
#     '椎间孔未显示': '椎间孔未显示',
#     '医源性异物': '医源性异物',
#     '中线偏移': '中线偏移',
#     '锁骨不对称': '锁骨不对称',
#     '肩胛骨未打开': '肩胛骨未打开',
#     '非医源性异物': '非医源性异物',
#     '吸气不足': '吸气不足',
#     '肩上未预留足够空间': '肩上未预留足够空间',
#     '肺野不全': '肺野不全',
# }
# # each class has 10 images and one text file content from the wiki page
# # each class has 10 images and one text file content from the wiki page
# for cls in tqdm(imges_classes):
#     cls_pth = os.path.join(new_pth, cls)
# #     page_content = wikipedia.page(wiki_titles[cls], auto_suggest=False).content
#     page_content = '先随便找点文本内容代替一下'
#
#     if not os.path.exists(cls_pth):
#         print('Creating {} folder'.format(cls))
#     else:
#         #save the text file
#         files_name= cls+'.txt'
#         with open(os.path.join(cls_pth, files_name), 'w') as f:
#             f.write(page_content)

client = chromadb.PersistentClient(path="DB")
embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()
collection_images = client.create_collection(
    name='multimodal_collection_images',
    embedding_function=embedding_function,
    data_loader=image_loader)

collection_text = client.create_collection(
    name='multimodal_collection_text',
    embedding_function=embedding_function,
    )

IMAGE_FOLDER = '/Users/titus.w/Code/med-rag-paper/rag-paper/data/_mixed_data_project/X-ray/png'
image_uris = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER) if not image_name.endswith('.txt')])   #获取所有图片路径
ids = [str(i) for i in range(len(image_uris))]  #为每条路径配备id
collection_images.add(ids=ids, uris=image_uris)
#此时只有图片存在向量数据库中
#文本召回图像，暂时不考虑
retrieved = collection_images.query(query_texts=["肺野不全"], include=['data'], n_results=3) #召回前3个
for img in retrieved['data'][0]:
    plt.imshow(img)
    plt.axis("off")
    plt.show()
#图像召回图像
query_image = np.array(Image.open(f"/Users/titus.w/Downloads/WX20240304-163252.png"))
print("Query Image")
plt.imshow(query_image)
plt.axis('off')
plt.show()
print("Results")
retrieved = collection_images.query(query_images=[query_image], include=['data'], n_results=10)
for img in retrieved['data'][0]:
    plt.imshow(img)
    plt.axis("off")
    plt.show()




from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()
TEXT_FOLDER = '/Users/titus.w/Code/med-rag-paper/rag-paper/data/_mixed_data_project/X-ray/txt'
text_pth = sorted([os.path.join(TEXT_FOLDER, image_name) for image_name in os.listdir(TEXT_FOLDER) if image_name.endswith('.txt')])
list_of_text = []
for text in text_pth:
    with open(text, 'r') as f:
        text = f.read()
        list_of_text.append(text)
ids_txt_list = ['id'+str(i) for i in range(len(list_of_text))]
collection_text.add(
    documents = list_of_text,
    ids =ids_txt_list
)
collection_text.add(
    documents = list_of_text,
    ids =ids_txt_list
)
results = collection_text.query(
    query_texts=["肺野太小了"],
    n_results=1
)

question = '这张图片拍的质量怎么样?'
query_image = '/Users/titus.w/Downloads/WX20240304-163252.png'
raw_image = Image.open(query_image)
doc = collection_text.query(   #图片搜出来的文本
    query_embeddings=embedding_function(query_image),

    n_results=1,

)['documents'][0][0]

plt.imshow(raw_image)
plt.show()
imgs = collection_images.query(query_uris=query_image, include=['data'], n_results=3) #图片搜出来的图片


prompt = """<|im_start|>system
好奇的人类和人工智能助手之间的聊天。
助手是医学影像质控中的佼佼者，对人类的问题给出有用、详细和礼貌的答案。
助手不会产生幻觉，并且非常注意细节，能给出片子拍的质量好坏的判断。<|im_end|>
<|im_start|>user
<image>
{question} 使用以下文章作为答案来源。不要写超出它的范围，除非你发现你的答案更好 {article}，如果你把你的答案变薄，最好把它加在文档后面。<|im_end|>
<|im_start|>assistant
""".format(question='question', article=doc)  #提示词中有图片召回的文本，和用户问题文本



model = AutoModel.from_pretrained("/Users/titus.w/Code/model/MC-LLaVA-3b", torch_dtype=torch.float16, trust_remote_code=True).to("cpu")
tokenizer = AutoTokenizer.from_pretrained("/Users/titus.w/Code/model/MC-LLaVA-3b")
image_processor = OpenCLIPImageProcessor(model.config.preprocess_config)
processor = LlavaProcessor(image_processor, tokenizer)
llava_script_path = '/rag-project/llava'
hf_hub_download(repo_id="visheratin/LLaVA-3b", filename="configuration_llava.py", local_dir=llava_script_path, force_download=True)
hf_hub_download(repo_id="visheratin/LLaVA-3b", filename="configuration_phi.py", local_dir=llava_script_path, force_download=True)
hf_hub_download(repo_id="visheratin/LLaVA-3b", filename="modeling_llava.py", local_dir=llava_script_path, force_download=True)
hf_hub_download(repo_id="visheratin/LLaVA-3b", filename="modeling_phi.py", local_dir=llava_script_path, force_download=True)
hf_hub_download(repo_id="visheratin/LLaVA-3b", filename="processing_llava.py", local_dir=llava_script_path, force_download=True)

inputs = processor(prompt, raw_image, model)
inputs['input_ids'] = inputs['input_ids'].to(model.device)
inputs['attention_mask'] = inputs['attention_mask'].to(model.device)
from transformers import TextStreamer
streamer = TextStreamer(tokenizer)
output = model.generate(**inputs, max_new_tokens=300, do_sample=True, top_p=0.5, temperature=0.2, eos_token_id=tokenizer.eos_token_id, streamer=streamer)
out = tokenizer.decode(output[0]).replace(prompt, "").replace("<|im_end|>", "")
print(out)
plt.imshow(raw_image)
plt.show()
imgs = collection_images.query(query_uris=query_image, include=['data'], n_results=3)
for img in imgs['data'][0][1:]:
    plt.imshow(img)
    plt.axis("off")
    plt.show()
print('answer is ==> '+out)



















pass




























