import os
import json
from dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.image_caption_dataset_builders.coco_dataset_builders.coco_dataset_builder import \
    CocoDatasetBuilder


class YJCaptionsDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the YJ-captions dataset, described in the paper 'Cross-Lingual Image
        Caption Generation' by Miyazaki and Shimizu.
        This dataset is based on the COCO dataset.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(YJCaptionsDatasetBuilder, self).__init__(root_dir_path, 'YJCaptions', data_split_str, struct_property,
                                                       CocoDatasetBuilder, 'COCO', indent)

        caption_file_name = 'yjcaptions26k_clean.json'
        self.caption_file_path = os.path.join(root_dir_path, caption_file_name)

        # We need to override a parent class behavior: no matter what data split is used in this dataset, we want the
        # base dataset builder (COCO builder) to use the train split, since all the images in this dataset are from the
        # COCO train split
        self.base_dataset_builder = CocoDatasetBuilder(os.path.join(root_dir_path, '..', 'COCO'),
                                                       'train', self.struct_property, self.indent + 1)

    def get_caption_data_internal(self):
        with open(self.caption_file_path, 'r', encoding='utf8') as caption_fp:
            caption_data = json.load(caption_fp)['annotations']
        return caption_data

    def get_all_image_ids(self):
        caption_data = self.get_caption_data_internal()
        return list(set([x['image_id'] for x in caption_data]))

    def get_caption_data(self):
        caption_data = self.get_caption_data_internal()
        data_split_image_ids = self.get_image_ids_for_split()
        data_split_image_ids_dict = {x: True for x in data_split_image_ids}
        return [x for x in caption_data if x['image_id'] in data_split_image_ids_dict]
