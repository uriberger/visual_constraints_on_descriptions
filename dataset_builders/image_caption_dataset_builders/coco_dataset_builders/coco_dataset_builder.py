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
        super(CocoDatasetBuilder, self).__init__(root_dir_path, 'COCO', data_split_str, struct_property, indent)

        train_val_annotations_dir = 'train_val_annotations2014'

        train_captions_file_path_suffix = os.path.join(train_val_annotations_dir, 'captions_train2014.json')
        self.train_captions_file_path = os.path.join(root_dir_path, train_captions_file_path_suffix)
        val_captions_file_path_suffix = os.path.join(train_val_annotations_dir, 'captions_val2014.json')
        self.val_captions_file_path = os.path.join(root_dir_path, val_captions_file_path_suffix)

        self.train_bboxes_file_name = 'instances_train2014.json'
        self.train_bboxes_file_path = os.path.join(root_dir_path, train_val_annotations_dir,
                                                   self.train_bboxes_file_name)
        self.val_bboxes_file_name = 'instances_val2014.json'
        self.val_bboxes_file_path = os.path.join(root_dir_path, train_val_annotations_dir,
                                                 self.val_bboxes_file_name)

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

    def get_gt_classes_data_internal(self):
        if self.data_split_str == 'train':
            external_bboxes_filepath = self.train_bboxes_file_path
        elif self.data_split_str == 'test':
            external_bboxes_filepath = self.val_bboxes_file_path
        bboxes_fp = open(external_bboxes_filepath, 'r')
        bboxes_data = json.load(bboxes_fp)

        category_id_to_class_id = {bboxes_data[u'categories'][x][u'id']: x for x in
                                   range(len(bboxes_data[u'categories']))}

        img_classes_dataset = {}
        # Go over all the object annotations
        for bbox_annotation in bboxes_data[u'annotations']:
            image_id = bbox_annotation[u'image_id']
            if image_id not in img_classes_dataset:
                img_classes_dataset[image_id] = []

            category_id = bbox_annotation[u'category_id']
            class_id = category_id_to_class_id[category_id]

            img_classes_dataset[image_id].append(class_id)

        return img_classes_dataset

    def get_class_mapping(self):
        bbox_fp = open(self.train_bboxes_file_path, 'r')
        bbox_data = json.load(bbox_fp)

        category_id_to_class_id = {bbox_data[u'categories'][x][u'id']: x for x in range(len(bbox_data[u'categories']))}
        category_id_to_name = {x[u'id']: x[u'name'] for x in bbox_data[u'categories']}
        class_mapping = {category_id_to_class_id[x]: category_id_to_name[x] for x in category_id_to_class_id.keys()}

        return class_mapping

    def create_image_path_finder(self):
        return CocoImagePathFinder(self.data_split_str, self.train_images_dir_path, self.val_images_dir_path)
