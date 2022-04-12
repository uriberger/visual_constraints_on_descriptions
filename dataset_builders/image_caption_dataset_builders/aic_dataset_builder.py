import os
import json
from dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_path_finder import ImagePathFinder


class AICImagePathFinder(ImagePathFinder):

    def __init__(self, images_dir_path):
        super(AICImagePathFinder, self).__init__()

        self.images_dir_path = images_dir_path

    def get_image_path(self, image_id):
        image_file_name = f'{hex(image_id)[2:]}.jpg'
        image_path = os.path.join(self.images_dir_path, image_file_name)

        return image_path


class AIChallengerDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the builder for the AI-Challenger dataset, described in the paper "AI Challenger : A Large-scale Dataset
        for Going Deeper in Image Understanding" by Wu et al.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(AIChallengerDatasetBuilder, self).__init__(root_dir_path, 'ai_challenger', data_split_str,
                                                         struct_property, indent)

        train_dir_name = 'ai_challenger_caption_train_20170902'
        val_dir_name = 'ai_challenger_caption_validation_20170910'
        train_captions_file_name = 'caption_train_annotations_20170902.json'
        val_captions_file_name = 'caption_validation_annotations_20170910.json'
        train_images_dir_name = 'caption_train_images_20170902'
        val_images_dir_name = 'caption_validation_images_20170910'

        self.train_captions_file_path = os.path.join(root_dir_path, train_dir_name, train_captions_file_name)
        self.val_captions_file_path = os.path.join(root_dir_path, val_dir_name, val_captions_file_name)
        self.train_images_dir_path = os.path.join(root_dir_path, train_dir_name, train_images_dir_name)
        self.val_images_dir_path = os.path.join(root_dir_path, val_dir_name, val_images_dir_name)

    @staticmethod
    def file_name_to_image_id(file_name):
        return int(file_name.split('.jpg')[0], 16)

    def load_from_json(self):
        if self.data_split_str == 'train':
            external_caption_file_path = self.train_captions_file_path
        elif self.data_split_str == 'val':
            external_caption_file_path = self.val_captions_file_path
        with open(external_caption_file_path, 'r') as caption_fp:
            loaded_data = json.load(caption_fp)
        return loaded_data

    def get_all_image_ids(self):
        loaded_data = self.load_from_json()
        return [self.file_name_to_image_id(x['image_id']) for x in loaded_data]

    def get_caption_data(self):
        loaded_data = self.load_from_json()
        caption_data = [x for outer in
                        [[{'image_id': self.file_name_to_image_id(y['image_id']), 'caption': z} for z in y['caption']]
                         for y in loaded_data]
                        for x in outer]
        return caption_data

    def get_gt_classes_data_internal(self):
        return None

    def get_gt_bboxes_data_internal(self):
        return None

    def get_class_mapping(self):
        return None

    def create_image_path_finder(self):
        if self.data_split_str == 'train':
            images_dir_path = self.train_images_dir_path
        elif self.data_split_str == 'val':
            images_dir_path = self.val_images_dir_path
        return AICImagePathFinder(images_dir_path)
