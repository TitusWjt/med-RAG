from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/bge-large-zh-v1.5', cache_dir='./')
print(model_dir)

