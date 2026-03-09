import shutil, os
path = os.path.expanduser("~/.cache/huggingface/hub/datasets--librispeech_asr")
if os.path.exists(path):
    shutil.rmtree(path)
    print("Removed dataset cache")
else:
    print("No dataset cache found")
