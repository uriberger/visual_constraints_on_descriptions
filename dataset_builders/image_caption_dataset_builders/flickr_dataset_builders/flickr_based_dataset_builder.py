import os
import abc
from dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import \
    ImageCaptionDatasetBuilder
from dataset_builders.image_caption_dataset_builders.flickr_dataset_builders.flickr30k_dataset_builder import \
    Flickr30kDatasetBuilder


class FlickrBasedDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the base class for dataset based on Flickr30k, i.e., the same images with captions in a different
        language.
    """

    def __init__(self, root_dir_path, name, data_split_str, struct_property, indent):
        super(FlickrBasedDatasetBuilder, self).__init__(root_dir_path, name, data_split_str, struct_property,
                                                        indent)

        # This dataset doesn't contain the images themselves- the images are in the COCO dataset
        flickr30k_path = os.path.join(self.root_dir_path, '..', 'flickr30')
        self.flickr30k_builder = Flickr30kDatasetBuilder(flickr30k_path, self.struct_property, self.indent + 1)

    @abc.abstractmethod
    def get_caption_data(self):
        return

    def get_gt_classes_data_internal(self):
        return self.flickr30k_builder.get_gt_classes_data()

    def get_gt_bboxes_data_internal(self):
        return self.flickr30k_builder.get_gt_bboxes_data()

    def get_class_mapping(self):
        return self.flickr30k_builder.get_class_mapping()

    def create_image_path_finder(self):
        return self.flickr30k_builder.create_image_path_finder()
