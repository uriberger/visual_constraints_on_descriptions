import os
import gzip
from dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_caption_dataset_builders.flickr30k_dataset_builder import Flickr30kDatasetBuilder


class Multi30kDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the dataset builder class for the multi30k dataset, described in the paper 'Multi30K: Multilingual
        English-German Image Descriptions' by Elliott et al.
        This dataset is based on the Flickr30k dataset.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(Multi30kDatasetBuilder, self).__init__(root_dir_path, 'multi30k', data_split_str, struct_property,
                                                     indent)

        data_dir_name = 'data'
        task2_dir_name = 'task2'
        raw_dir_name = 'raw'
        image_splits_dir_name = 'image_splits'

        train_file_name_prefix = 'train'
        val_file_name_prefix = 'val'
        test_file_name_prefix = 'test_2016'
        caption_file_name_suffix = 'de.gz'
        image_file_name_suffix = 'images.txt'

        self.caption_inds = range(1, 6)

        train_caption_file_names = [f'{train_file_name_prefix}.{i}.{caption_file_name_suffix}' for i in self.caption_inds]
        val_caption_file_names = [f'{val_file_name_prefix}.{i}.{caption_file_name_suffix}' for i in self.caption_inds]
        test_caption_file_names = [f'{test_file_name_prefix}.{i}.{caption_file_name_suffix}' for i in self.caption_inds]

        train_image_file_name = f'{train_file_name_prefix}_{image_file_name_suffix}'
        val_image_file_name = f'{val_file_name_prefix}_{image_file_name_suffix}'
        test_image_file_name = f'{test_file_name_prefix}_{image_file_name_suffix}'

        task2_dir_path = os.path.join(root_dir_path, data_dir_name, task2_dir_name)

        caption_dir_path = os.path.join(task2_dir_path, raw_dir_name)
        self.train_caption_file_paths = [os.path.join(caption_dir_path, train_caption_file_names[i])
                                         for i in range(len(self.caption_inds))]
        self.val_caption_file_paths = [os.path.join(caption_dir_path, val_caption_file_names[i])
                                       for i in range(len(self.caption_inds))]
        self.test_caption_file_paths = [os.path.join(caption_dir_path, test_caption_file_names[i])
                                        for i in range(len(self.caption_inds))]

        image_dir_path = os.path.join(task2_dir_path, image_splits_dir_name)
        self.train_image_file_path = os.path.join(image_dir_path, train_image_file_name)
        self.val_image_file_path = os.path.join(image_dir_path, val_image_file_name)
        self.test_image_file_path = os.path.join(image_dir_path, test_image_file_name)

    def get_caption_data(self):
        line_to_image_id = []
        if self.data_split_str == 'train':
            image_file_path = self.train_image_file_path
        elif self.data_split_str == 'val':
            image_file_path = self.val_image_file_path
        elif self.data_split_str == 'test':
            image_file_path = self.test_image_file_path
        with open(image_file_path, 'r') as fp:
            for image_file_name in fp:
                image_id = int(image_file_name.split('.')[0])
                line_to_image_id.append(image_id)

        image_id_captions_pairs = []
        for i in range(len(self.caption_inds)):
            if self.data_split_str == 'train':
                caption_file_path = self.train_caption_file_paths[i]
            elif self.data_split_str == 'val':
                caption_file_path = self.val_caption_file_paths[i]
            elif self.data_split_str == 'test':
                caption_file_path = self.test_caption_file_paths[i]
            # with open(caption_file_path, 'r', encoding='utf-8') as fp:
            with gzip.open(caption_file_path, 'r') as fp:
                line_ind = 0
                for line in fp:
                    caption = line.strip().decode('utf-8')
                    image_id_captions_pairs.append({'image_id': line_to_image_id[line_ind], 'caption': caption})
                    line_ind += 1

        return image_id_captions_pairs

    def create_image_path_finder(self):
        # This dataset doesn't contain the images themselves- the images are in the flickr30k dataset
        flickr30k_path = os.path.join(self.root_dir_path, '..', 'flickr30')
        flickr30k_builder = Flickr30kDatasetBuilder(flickr30k_path, self.struct_property, self.indent + 1)
        return flickr30k_builder.create_image_path_finder()
