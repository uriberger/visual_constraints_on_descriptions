import os
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.flickr_dataset_builders.flickr30k_dataset_builder import \
    Flickr30kDatasetBuilder


class Flickr8kCNDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the flickr8k-cn dataset, described in the paper 'Adding Chinese Captions
        to Images' by Li et al.
        This dataset is based on the Flickr8k dataset.
    """

    def __init__(self, root_dir_path, struct_property, indent):
        super(Flickr8kCNDatasetBuilder, self).__init__(
            root_dir_path, 'flickr8kcn', 'Chinese',
            struct_property, Flickr30kDatasetBuilder, 'flickr30', indent
        )

        data_dir_name = 'data'
        caption_file_name = 'flickr8kzhc.caption.txt'
        self.caption_file_path = os.path.join(root_dir_path, data_dir_name, caption_file_name)

    def get_caption_data(self):
        image_id_captions_pairs = []
        with open(self.caption_file_path, 'r', encoding='utf8') as fp:
            for line in fp:
                striped_line = line.strip()
                if len(striped_line) == 0:
                    continue
                line_parts = striped_line.split()
                assert len(line_parts) == 2
                image_id = int(line_parts[0].split('_')[0])
                caption = line_parts[1]
                image_id_captions_pairs.append({'image_id': image_id, 'caption': caption})

        return image_id_captions_pairs
