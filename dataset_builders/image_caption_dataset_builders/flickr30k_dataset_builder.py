import os
from dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_path_finder import ImagePathFinder


class Flickr30kImagePathFinder(ImagePathFinder):

    def __init__(self, images_dir_path):
        super(Flickr30kImagePathFinder, self).__init__()

        self.images_dir_path = images_dir_path

    def get_image_path(self, image_id):
        image_file_name = f'{image_id}.jpg'
        image_path = os.path.join(self.images_dir_path, image_file_name)

        return image_path


class Flickr30kDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the dataset builder class for the flickr30k dataset, described in the paper 'From image descriptions to
        visual denotations: New similarity metrics for semantic inference over event descriptions' by Young et al.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(Flickr30kDatasetBuilder, self).__init__(root_dir_path, 'flickr30k', data_split_str, struct_property,
                                                      indent)

        tokens_dir_name = 'tokens'
        tokens_file_name = 'results_20130124.token'
        self.tokens_file_path = os.path.join(self.root_dir_path, tokens_dir_name, tokens_file_name)

        images_dir_name = 'images'
        self.images_dir_path = os.path.join(self.root_dir_path, images_dir_name)

    def get_caption_data(self):
        fp = open(self.tokens_file_path, encoding='utf-8')
        image_id_captions_pairs = []
        for line in fp:
            split_line = line.strip().split('#')
            img_file_name = split_line[0]
            image_id = self.image_filename_to_id(img_file_name)
            caption = split_line[1].split('\t')[1]  # The first token is caption number

            image_id_captions_pairs.append({'image_id': image_id, 'caption': caption})

        return image_id_captions_pairs

    def create_image_path_finder(self):
        return Flickr30kImagePathFinder(self.images_dir_path)

    @staticmethod
    def image_file_name_to_id(image_file_name):
        return int(image_file_name.split('.')[0])
