import datasets
import os
import json
from PIL import Image

class ReceiptDataset(datasets.GeneratorBasedBuilder):

    def _info(self):
        features = datasets.Features(
            {
                "file_id": datasets.Value("string"),
                "input_image": datasets.Image(),
                "company": datasets.Value("string"),
                "date": datasets.Value("string"),
                "address": datasets.Value("string"),
                "total": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'data_dir': '../data'}
            )
        ]

    def _generate_examples(self, data_dir=None):
        key = -1
        for file in os.listdir(data_dir):
            if file.endswith(".txt"):
                file_id = file.split('.')[0]
                file_path = os.path.join(data_dir, file)
                with open(file_path, 'r') as f:
                    element_data = f.read()
                element_features = json.loads(element_data)
                image_path = os.path.join(data_dir, file_id) + '.jpg'
                if os.path.exists(image_path) and len(element_features.keys()) == 4:
                    key += 1
                    yield key, {
                        "file_id": file_id,
                        "input_image": Image.open(image_path).convert("RGB"),
                        "company": element_features["company"],
                        "date": element_features["date"],
                        "address": element_features["address"],
                        "total": element_features["total"],
                    }
