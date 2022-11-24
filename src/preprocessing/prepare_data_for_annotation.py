from src.data.annotation_utils import AnnotationJsonCreator
from src.data.doctr_utils import DoctrTransformer
from pathlib import Path
import os
from pdf2image import convert_from_path
#import pandas as pd
#import xlrd
import cv2
from src.data.image_preprocessing import rotate_image, get_angle


img_folder_path = "./data/to_straighten"
output_path = "./data/tickets_ggdrive_str.json"


def main():
    list_img_path = [os.path.join(img_folder_path, x) for x in os.listdir(img_folder_path)]
    # convert pdf to img and save them
    for path in list_img_path:
        if path.endswith(('.pdf', '.PDF')):
            pages = convert_from_path(path, 500)
            path_no_suffix = path[:-4]
            for page_index, page in enumerate(pages):
                page.save(path_no_suffix + "_page_{}".format(page_index) + '.jpg', 'JPEG')

        # rotate image
        original_image = cv2.imread(path)
        skew_angle_in_degrees = get_angle(original_image)
        rotated_image = rotate_image(original_image, -skew_angle_in_degrees)
        cv2.imwrite(path[:-4] + "_rotated" + '.jpg', rotated_image)

    list_img_path = [Path(os.path.join(img_folder_path, x)) for x in os.listdir(img_folder_path) if x.endswith(('.jpg', '.jpeg', ".png"))]
    list_doctr_docs = DoctrTransformer().transform(list_img_path)
    annotations = AnnotationJsonCreator(list_img_path, output_path).transform(list_doctr_docs, upload=False)

if __name__ == '__main__':
    main()