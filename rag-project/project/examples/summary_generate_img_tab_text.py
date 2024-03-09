import os
import base64
import requests
from typing import Any
from pydantic import BaseModel
from jpg_extract_form_pdf_yolo import raw_pdf_elements
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

os.environ["OPENAI_API_KEY"] = "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"



class ImageSummarizer:

    def __init__(self, image_path, use_cloudflare=False) -> None:
        self.image_path = image_path
        self.use_cloudflare = use_cloudflare
        self.prompt = """
You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Give a concise summary of the image that is well optimized for retrieval.
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

class Element(BaseModel):
    type: str
    text: Any

table_elements = []
text_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        table_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        text_elements.append(Element(type="text", text=str(element)))

image_path = '/rag-project/data/demo_data/jpg'
image_data_list = []
image_summary_list = []
for img_file in sorted(os.listdir(image_path)):
    if img_file.endswith(".jpg"):
        summarizer = ImageSummarizer(os.path.join(image_path, img_file), use_cloudflare=False)
        data, summary = summarizer.summarize()
        image_data_list.append(data)
        image_summary_list.append(summary)
image_summary_list = [choice['message']['content'] for item in image_summary_list for choice in item['choices']]
print(f"PDF contains {len(text_elements)} text data")
print(f"PDF contains {len(table_elements)} table data")
print(f"from dick produce {len(image_summary_list)} image abstracts")

#To summarize text and table by GPT3.5
prompt_text = """
  You are responsible for concisely summarizing table or text chunk:

  {element}
"""
prompt = ChatPromptTemplate.from_template(prompt_text)
summarize_chain = {"element": lambda x: x} | prompt | ChatOpenAI(temperature=0, model="gpt-3.5-turbo") | StrOutputParser()
tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
texts = [i.text for i in text_elements]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})