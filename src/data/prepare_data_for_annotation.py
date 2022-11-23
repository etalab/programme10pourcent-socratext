from src.data.annotation_utils import AnnotationJsonCreator
from src.data.doctr_utils import DoctrTransformer
from pathlib import Path
import os
from pdf2image import convert_from_path
#import pandas as pd
#import xlrd
import cv2


img_folder_path = "./data/ggdrive"
output_path = "./data/tickets_ggdrive.json"


def main():
    list_img_path = [os.path.join(img_folder_path, x) for x in os.listdir(img_folder_path)]
    # convert pdf to img and save them
    for path in list_img_path:
        if path.endswith(('.pdf', '.PDF')):
            pages = convert_from_path(path, 500)
            path_no_suffix = path[:-4]
            for page_index, page in enumerate(pages):
                page.save(path_no_suffix + "_page_{}".format(page_index) + '.jpg', 'JPEG')

    list_img_path = [Path(os.path.join(img_folder_path, x)) for x in os.listdir(img_folder_path) if x.endswith(('.jpg', '.jpeg', ".png"))]
    list_doctr_docs = DoctrTransformer().transform(list_img_path)
    annotations = AnnotationJsonCreator(list_img_path, output_path).transform(list_doctr_docs, upload=False)

if __name__ == '__main__':
    main()