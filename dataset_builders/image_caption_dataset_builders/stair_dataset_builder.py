import os
import json
from dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_caption_dataset_builders.coco_dataset_builder import CocoDatasetBuilder


class StairDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the dataset builder class for the STAIR-caption dataset, described in the paper 'STAIR Captions:
        Constructing a Large-Scale Japanese Image Caption Dataset' by Yoshikawa et al.
        This dataset is based on the COCO dataset.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(StairDatasetBuilder, self).__init__(root_dir_path, 'stair', data_split_str, struct_property,
                                                  indent)

        captions_dir_name = 'stair_captions_v1.2'
        captions_dir_path = os.path.join(root_dir_path, captions_dir_name)
        train_captions_file_name = f'{captions_dir_name}_train.json'
        val_captions_file_name = f'{captions_dir_name}_val.json'

        self.train_captions_file_path = os.path.join(captions_dir_path, train_captions_file_name)
        self.val_captions_file_path = os.path.join(captions_dir_path, val_captions_file_name)

    def get_caption_data(self):
        if self.data_split_str == 'train':
            external_caption_file_path = self.train_captions_file_path
        elif self.data_split_str == 'val':
            external_caption_file_path = self.val_captions_file_path
        with open(external_caption_file_path, 'r', encoding='utf8') as caption_fp:
            caption_data = json.load(caption_fp)['annotations']
        return caption_data

    def create_image_path_finder(self):
        # This dataset doesn't contain the images themselves- the images are in the COCO dataset
        coco_path = os.path.join(self.root_dir_path, '..', 'COCO')
        coco_builder = CocoDatasetBuilder(coco_path, self.data_split_str, self.struct_property, self.indent + 1)
        return coco_builder.create_image_path_finder()
