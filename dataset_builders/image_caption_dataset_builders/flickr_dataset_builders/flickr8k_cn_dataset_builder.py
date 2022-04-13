import os
from dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.image_caption_dataset_builders.flickr_dataset_builders.flickr30k_dataset_builder import \
    Flickr30kDatasetBuilder


class Flickr8kCNDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the flickr8k-cn dataset, described in the paper 'Adding Chinese Captions
        to Images' by Li et al.
        This dataset is based on the Flickr8k dataset.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(Flickr8kCNDatasetBuilder, self).__init__(root_dir_path, 'flickr8kcn', data_split_str, struct_property,
                                                       Flickr30kDatasetBuilder, 'flickr30', indent)

        data_dir_name = 'data'
        caption_file_name = 'flickr8kzhc.caption.txt'
        self.caption_file_path = os.path.join(root_dir_path, data_dir_name, caption_file_name)

    def get_caption_data_internal(self):
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

    def get_all_image_ids(self):
        all_caption_data = self.get_caption_data_internal()
        return list(set([x['image_id'] for x in all_caption_data]))

    def get_caption_data(self):
        data_split_image_ids = self.get_image_ids_for_split()
        data_split_image_ids_dict = {x: True for x in data_split_image_ids}
        image_id_captions_pairs = []
        with open(self.caption_file_path, 'r', encoding='utf8') as fp:
            for line in fp:
                striped_line = line.strip()
                if len(striped_line) == 0:
                    continue
                line_parts = striped_line.split()
                assert len(line_parts) == 2
                image_id = int(line_parts[0].split('_')[0])
                if image_id in data_split_image_ids_dict:
                    caption = line_parts[1]
                    image_id_captions_pairs.append({'image_id': image_id, 'caption': caption})

        return image_id_captions_pairs
