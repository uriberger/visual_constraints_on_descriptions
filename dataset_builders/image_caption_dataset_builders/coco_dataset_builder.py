import os
import json
from dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_path_finder import ImagePathFinder


class CocoImagePathFinder(ImagePathFinder):

    def __init__(self, data_split_str, train_images_dir_path, val_images_dir_path):
        super(CocoImagePathFinder, self).__init__()

        self.data_split_str = data_split_str
        self.train_images_dir_path = train_images_dir_path
        self.val_images_dir_path = val_images_dir_path

    def get_image_path(self, image_id):
        image_file_name = 'COCO_' + self.data_split_str + '2014_000000' + '{0:06d}'.format(image_id) + '.jpg'
        if self.data_split_str == 'train':
            images_dir_path = self.train_images_dir_path
        elif self.data_split_str == 'val':
            images_dir_path = self.val_images_dir_path
        image_path = os.path.join(images_dir_path, image_file_name)

        return image_path


class CocoDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the dataset builder class for the MSCOCO dataset, described in the paper
        'Microsoft COCO: Common Objects in Context' by Lin et al.
        Something weird about COCO: They published 3 splits: train, val, test, but they didn't provide labels for the
        test split. So we're going to ignore the test set.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(CocoDatasetBuilder, self).__init__(root_dir_path, 'coco', data_split_str, struct_property, indent)

        self.train_val_annotations_dir = 'train_val_annotations2014'

        train_captions_file_path_suffix = os.path.join(self.train_val_annotations_dir, 'captions_train2014.json')
        self.train_captions_file_path = os.path.join(root_dir_path, train_captions_file_path_suffix)
        val_captions_file_path_suffix = os.path.join(self.train_val_annotations_dir, 'captions_val2014.json')
        self.val_captions_file_path = os.path.join(root_dir_path, val_captions_file_path_suffix)

        self.train_images_dir_path = os.path.join(root_dir_path, 'train2014')
        self.val_images_dir_path = os.path.join(root_dir_path, 'val2014')

    def get_caption_data(self):
        if self.data_split_str == 'train':
            external_caption_file_path = self.train_captions_file_path
        elif self.data_split_str == 'val':
            external_caption_file_path = self.val_captions_file_path
        with open(external_caption_file_path, 'r') as caption_fp:
            caption_data = json.load(caption_fp)['annotations']
        return caption_data

    def create_image_path_finder(self):
        return CocoImagePathFinder(self.data_split_str, self.train_images_dir_path, self.val_images_dir_path)
