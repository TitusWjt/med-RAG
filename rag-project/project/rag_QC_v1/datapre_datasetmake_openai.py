import os
import base64
import requests
from typing import Any
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

os.environ["OPENAI_API_KEY"] = "sess-DLi7sV2qdgBV7nf5tRWxFi8kbyFQogoEOkkXZ8iQ"



class ImageSummarizer:

    def __init__(self, image_path, use_cloudflare=False) -> None:
        self.image_path = image_path
        self.use_cloudflare = use_cloudflare
        self.prompt = """
这里面有异物吗？
"""

    def base64_encode_image(self):
        with open(self.image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def summarize(self, prompt=None):
        base64_image_data = self.base64_encode_image()

        # 根据use_cloudflare参数选择API的URL
        if self.use_cloudflare:
            url = "https://gateway.ai.cloudflare.com/v1/80d06ad985e46750d65651b2272ad787/openai/openai/chat/completions"
        else:
            url = "https://api.openai.com/v1/chat/completions"

        data = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt if prompt else self.prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image_data}"}
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }

        response = requests.post(url, json=data, headers=headers)
        return base64_image_data, response.json()



# 假设这是你存放医学影像的路径
image_directory = '/Users/titus.w/Downloads/医学影像质控数据集(图文对)/异物'

# 获取路径下所有的PNG图片文件
png_images = []
for root, dirs, files in os.walk(image_directory):
    for file in files:
        if file.lower().endswith('.png'):
            png_images.append(os.path.join(root, file))
# 循环处理每个图片文件
for image_file in png_images:
    # 假设所有图片的质控结果是一样的，这里替换成实际的质控结果描述
    # 使用GPT-4生成质控报告
    summarizer = ImageSummarizer(image_file, use_cloudflare=False)
    data, summary = summarizer.summarize()
    print(summary)
    # 这里可以进一步将报告保存到文件或数据库中

