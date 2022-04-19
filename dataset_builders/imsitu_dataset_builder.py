from dataset_builders.image_path_finder import ImagePathFinder
from dataset_builders.dataset_builder import DatasetBuilder
from utils.general_utils import generate_dataset
import os
import json

""" ImSitu is the situation recognition dataset described in the paper "Situation recognition: Visual semantic role
    labeling for image understanding" by Yatskar et al. Each image is annotated by a verb and frames: Each frame is
    related to a specific verb, and specifies it's slots (e.g., for the verb "throwing", the slots might be "agent",
    "object", "target"...). The frames are taken from FrameNet and are filled by human annotators. 
    The purpose of this file is to create a dataset that maps, for each image and each human-annotated frame filling,
    how many slots are missing.
    In the original dataset, each image name is the name of it's verb, followed by the serial number of the image within
    this verb (e.g., 'sitting_1.jpg', 'sitting_2.jpg', 'staring_1.jpg', ...). Each verb will be specified by a specific
    index, and the image id will be the verb index and the serial number of the image concatenated.
"""


MULT_FACT = 1000


class ImSituImagePathFinder(ImagePathFinder):

    def __init__(self, images_dir_path, verb_to_ind):
        super(ImSituImagePathFinder, self).__init__()
        
        self.images_dir_path = images_dir_path
        self.ind_to_verb = [x[0] for x in sorted(list(verb_to_ind.items()), key=lambda x:x[1])]
        ''' There are less than 1000 images for each verb. So if we'll multiply the verb index by 1000, there will be 3
        digits left for the serial number of the image. For example, if the verb index is 35 and the serial number is
        456, the image id will be 35 * 1000 + 456 = 35,456. '''
        self.mult_fact = MULT_FACT

    def get_image_path(self, image_id):
        image_serial_num = image_id % self.mult_fact
        verb_ind = image_id // self.mult_fact
        image_file_name = self.ind_to_verb[verb_ind] + '_' + str(image_serial_num) + '.jpg'
        image_file_path = os.path.join(self.images_dir_path, image_file_name)
        return image_file_path


class ImSituDatasetBuilder(DatasetBuilder):
    """ This class builds the image->missing slots number dataset. """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(ImSituDatasetBuilder, self).__init__('imsitu', data_split_str, struct_property, indent)
        self.root_dir_path = root_dir_path
        self.images_dir_path = os.path.join(root_dir_path, 'resized_256')

        self.verb_to_ind_file_path = os.path.join(self.cached_dataset_files_dir, 'imsitu_verb_to_ind')

    def get_struct_data_internal(self):
        if self.struct_property == 'empty_frame_slots_num':
            with open(os.path.join(self.root_dir_path, self.data_split_str + '.json')) as fp:
                data = json.load(fp)

            verb_to_ind = self.generate_verb_to_ind_mapping()

            # For each image annotation, we count how many of the frame slots were left empty
            struct_data_lists = [
                [
                    (self.image_name_to_ind(x[0], verb_to_ind),
                     len([z for z in y.values() if z == ''])) for y in x[1]['frames']
                ]
                for x in data.items()
            ]
            struct_data = [x for outer in struct_data_lists for x in outer]

        return struct_data

    def generate_verb_to_ind_mapping(self):
        return generate_dataset(self.verb_to_ind_file_path, self.generate_verb_to_ind_mapping_internal)

    def generate_verb_to_ind_mapping_internal(self):
        with open(os.path.join(self.root_dir_path, self.data_split_str + '.json')) as fp:
            data = json.load(fp)

        all_image_names = list(data.keys())
        all_verbs = list(set([image_name.split('_')[0] for image_name in all_image_names]))
        return {all_verbs[i]: i for i in range(len(all_verbs))}

    def create_image_path_finder(self):
        verb_to_ind = self.generate_verb_to_ind_mapping()
        image_path_finder = ImSituImagePathFinder(self.images_dir_path, verb_to_ind)
        return image_path_finder

    @staticmethod
    def image_name_to_ind(image_name, verb_to_ind):
        name_parts = image_name.split('_')
        verb_name = name_parts[0]
        image_serial_num = name_parts[1].split('.')[0]
        return verb_to_ind[verb_name] * MULT_FACT + int(image_serial_num)
