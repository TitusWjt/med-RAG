from IPython.display import Image
from unstructured.partition.pdf import partition_pdf

#extract jpg form yolox
#抽取pdf中的图片元素
images_path = "/rag-project/data/runs"
raw_pdf_elements = partition_pdf(
    filename="/rag-project/data/demo_data/weekly_market_recap.pdf",
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=images_path,
)
Image('/Users/titus.w/Code/med-rag-paper/rag-project/data/runs/figure-1-1.jpg')
