import os
import abc
from dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_caption_dataset_builders.coco_dataset_builders.coco_dataset_builder import CocoDatasetBuilder


class CocoBasedDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the base class for dataset based on COCO, i.e., the same images with captions in a different language.
    """

    def __init__(self, root_dir_path, name, data_split_str, struct_property, indent):
        super(CocoBasedDatasetBuilder, self).__init__(root_dir_path, name, data_split_str, struct_property,
                                                      indent)

        # This dataset doesn't contain the images themselves- the images are in the COCO dataset
        coco_path = os.path.join(self.root_dir_path, '..', 'COCO')
        self.coco_builder = CocoDatasetBuilder(coco_path, self.data_split_str, self.struct_property, self.indent + 1)

    @abc.abstractmethod
    def get_caption_data(self):
        return

    def get_gt_classes_data_internal(self):
        return self.coco_builder.get_gt_classes_data()

    def get_class_mapping(self):
        return self.coco_builder.get_class_mapping()

    def create_image_path_finder(self):
        return self.coco_builder.create_image_path_finder()
