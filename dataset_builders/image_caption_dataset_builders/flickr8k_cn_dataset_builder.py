import os
from dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_caption_dataset_builders.flickr30k_dataset_builder import Flickr30kDatasetBuilder


class Flickr8kCNDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the dataset builder class for the flickr8k-cn dataset, described in the paper 'Adding Chinese Captions
        to Images' by Li et al.
        This dataset is based on the Flickr8k dataset.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(Flickr8kCNDatasetBuilder, self).__init__(root_dir_path, 'multi30k', data_split_str, struct_property,
                                                     indent)

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

    def create_image_path_finder(self):
        # This dataset doesn't contain the images themselves- the images are in the flickr30k dataset
        flickr30k_path = os.path.join(self.root_dir_path, '..', 'flickr30')
        flickr30k_builder = Flickr30kDatasetBuilder(flickr30k_path, self.struct_property, self.indent + 1)
        return flickr30k_builder.create_image_path_finder()
