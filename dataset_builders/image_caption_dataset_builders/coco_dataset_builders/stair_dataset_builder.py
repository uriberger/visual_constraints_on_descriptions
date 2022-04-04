import os
import json
from dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.image_caption_dataset_builders.coco_dataset_builders.coco_dataset_builder import \
    CocoDatasetBuilder


class StairDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the STAIR-caption dataset, described in the paper 'STAIR Captions:
        Constructing a Large-Scale Japanese Image Caption Dataset' by Yoshikawa et al.
        This dataset is based on the COCO dataset.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(StairDatasetBuilder, self).__init__(root_dir_path, 'stair', data_split_str, struct_property,
                                                  CocoDatasetBuilder, 'COCO', indent)

        captions_file_prefix = 'stair_captions_v1.2'
        train_captions_file_name = f'{captions_file_prefix}_train.json'
        val_captions_file_name = f'{captions_file_prefix}_val.json'

        self.train_captions_file_path = os.path.join(root_dir_path, train_captions_file_name)
        self.val_captions_file_path = os.path.join(root_dir_path, val_captions_file_name)

    def get_caption_data(self):
        if self.data_split_str == 'train':
            external_caption_file_path = self.train_captions_file_path
        elif self.data_split_str == 'val':
            external_caption_file_path = self.val_captions_file_path
        with open(external_caption_file_path, 'r', encoding='utf8') as caption_fp:
            caption_data = json.load(caption_fp)['annotations']
        return caption_data
