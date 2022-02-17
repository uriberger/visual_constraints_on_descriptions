import os
from dataset_builders.dataset_builder import DatasetBuilder

from dataset_builders.image_caption_dataset_builders.coco_dataset_builder import CocoDatasetBuilder
from dataset_builders.image_caption_dataset_builders.flickr30k_dataset_builder import Flickr30kDatasetBuilder
from dataset_builders.image_caption_dataset_builders.multi30k_dataset_builder import Multi30kDatasetBuilder
from dataset_builders.imsitu_dataset_builder import ImSituDatasetBuilder


""" This utility creates a dataset builder given the dataset name.
    A dataset builder is an object that builds a torch.utils.data.Data object (the actual dataset), given the external
    files of the dataset.
    The utility assumes the name of the dataset is the name of its root directory.
"""


def create_dataset_builder(dataset_name, data_split_str, struct_property):
    root_dir = os.path.join(DatasetBuilder.get_datasets_dir(), dataset_name)
    if dataset_name == 'COCO':
        dataset_builder = CocoDatasetBuilder(root_dir, data_split_str, struct_property, 1)
    elif dataset_name == 'flickr30':
        dataset_builder = Flickr30kDatasetBuilder(root_dir, struct_property, 1)
    elif dataset_name == 'multi30k-dataset':
        dataset_builder = Multi30kDatasetBuilder(root_dir, data_split_str, struct_property, 1)
    elif dataset_name == 'imSitu':
        dataset_builder = ImSituDatasetBuilder(root_dir, data_split_str, struct_property, 1)
    else:
        assert False

    return dataset_builder
