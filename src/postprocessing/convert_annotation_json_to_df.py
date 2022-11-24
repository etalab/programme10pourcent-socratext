from pathlib import Path

from src.data.annotation_utils import LabelStudioConvertor


annotation_json = Path("data/annotated_json/project-1-at-2022-11-23-16-22-ef6202f7.json")
output_path = Path("data/annotated_df/sample_1.csv")


df_annotation = LabelStudioConvertor(annotation_json, output_path, True).transform()
df_annotation.to_csv(output_path, index=False, sep= "\t")