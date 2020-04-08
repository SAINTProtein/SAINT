import os
from pathlib import Path
# if os.path.isdir('./outputs/'):
#     print('path already exists')
Path("./outputs").mkdir(parents=False, exist_ok=True)

inputlist = 'list_test'
inputdir = 'Test'


predict_batch_size = 64