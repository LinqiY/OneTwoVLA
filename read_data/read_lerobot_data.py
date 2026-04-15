import pandas as pd

# onetwovla-dataset
# path = "/inspire/hdd/global_user/gongjingjing-25039/lqyin/datasets/onetwovla-dataset/wild_move_to/data/chunk-006/episode_006000.parquet"

# coco
# path = "/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/coco/data/train-00000-of-00040-67e35002d152155c.parquet"

# latex_ocr
# path = "/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/LaTeX_OCR/full/test-00000-of-00001.parquet"
# path = "/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/LaTeX_OCR/human_handwrite/test-00000-of-00001.parquet"
path = "/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/LaTeX_OCR/small/test-00000-of-00001.parquet"

# aokvqa
path = "/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/A-OKVQA/data/test-00000-of-00001-d306bf3ad53b6618.parquet"

df = pd.read_parquet(path)

print(df.shape)
print(df.columns)
print(df.head())