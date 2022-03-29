import os
from dataset_builders.dataset_builder import DatasetBuilder

from dataset_builders.image_caption_dataset_builders.coco_dataset_builders.coco_dataset_builder import \
    CocoDatasetBuilder
from dataset_builders.image_caption_dataset_builders.flickr_dataset_builders.flickr30k_dataset_builder import \
    Flickr30kDatasetBuilder
from dataset_builders.image_caption_dataset_builders.flickr_dataset_builders.multi30k_dataset_builder import \
    Multi30kDatasetBuilder
from dataset_builders.image_caption_dataset_builders.coco_dataset_builders.stair_dataset_builder import \
    StairDatasetBuilder
from dataset_builders.image_caption_dataset_builders.flickr_dataset_builders.flickr8k_cn_dataset_builder import \
    Flickr8kCNDatasetBuilder
from dataset_builders.image_caption_dataset_builders.coco_dataset_builders.coco_cn_dataset_builder import \
    CocoCNDatasetBuilder
from dataset_builders.image_caption_dataset_builders.iapr_tc12_builder import IAPRTC12DatasetBuilder
from dataset_builders.imsitu_dataset_builder import \
    ImSituDatasetBuilder


""" This utility creates a dataset builder given the dataset name.
    A dataset builder is an object that builds a torch.utils.data.Data object (the actual dataset), given the external
    files of the dataset.
    The utility assumes the name of the dataset is the name of its root directory.
"""


def create_dataset_builder(dataset_name, data_split_str, struct_property, translated):
    root_dir = os.path.join(DatasetBuilder.get_datasets_dir(), dataset_name)
    if dataset_name == 'COCO':
        dataset_builder = CocoDatasetBuilder(root_dir, data_split_str, struct_property, 1)
    elif dataset_name == 'flickr30':
        dataset_builder = Flickr30kDatasetBuilder(root_dir, struct_property, 1)
    elif dataset_name == 'multi30k':
        dataset_builder = Multi30kDatasetBuilder(root_dir, data_split_str, struct_property, translated, 1)
    elif dataset_name == 'imSitu':
        dataset_builder = ImSituDatasetBuilder(root_dir, data_split_str, struct_property, 1)
    elif dataset_name == 'STAIR-captions':
        dataset_builder = StairDatasetBuilder(root_dir, data_split_str, struct_property, 1)
    elif dataset_name == 'flickr8kcn':
        dataset_builder = Flickr8kCNDatasetBuilder(root_dir, data_split_str, struct_property, 1)
    elif dataset_name == 'coco-cn':
        dataset_builder = CocoCNDatasetBuilder(root_dir, data_split_str, struct_property, translated, 1)
    elif dataset_name == 'iaprtc12':
        dataset_builder = IAPRTC12DatasetBuilder(root_dir, struct_property, 1)
    else:
        assert False

    return dataset_builder
