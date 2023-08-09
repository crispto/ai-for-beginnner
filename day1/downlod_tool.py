import time
from huggingface_hub import snapshot_download

repo_id = 'facebook/detr-resnet-50'
local_dir = './hub'
cache_dir = './cache'

while True:
    try:
        snapshot_download(repo_id,
                          local_dir=local_dir, 
                          cache_dir=cache_dir,
                          local_dir_use_symlinks=False,
                          resume_download=True,
                          allow_patterns=[".model", "*.json", "*.bin", "*.py", "*.md", "*.txt"],
                          ignore_patterns=["*.safetensors", "*.msgpack", "*.h5", "*.ot"])
    except Exception as e:
        print(e)
    else:
        print("download completed")
        break
        