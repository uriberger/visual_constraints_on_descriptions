from dataset_builders.image_path_finder import ImagePathFinder
from dataset_builders.single_dataset_builders.external_dataset_builders.external_dataset_builder \
    import ExternalDatasetBuilder
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


VERB_MULT_FACT = 1000


class ImSituImagePathFinder(ImagePathFinder):

    def __init__(self, images_dir_path, verb_to_ind):
        super(ImSituImagePathFinder, self).__init__()
        
        self.images_dir_path = images_dir_path
        self.ind_to_verb = [x[0] for x in sorted(list(verb_to_ind.items()), key=lambda x:x[1])]
        ''' There are less than 1000 images for each verb. So if we'll multiply the verb index by 1000, there will be 3
        digits left for the serial number of the image. For example, if the verb index is 35 and the serial number is
        456, the image id will be 35 * 1000 + 456 = 35,456. '''
        self.verb_mult_fact = VERB_MULT_FACT

    def get_image_path(self, image_id):
        image_serial_num = image_id % self.verb_mult_fact
        verb_ind = image_id // self.verb_mult_fact
        image_file_name = self.ind_to_verb[verb_ind] + '_' + str(image_serial_num) + '.jpg'
        image_file_path = os.path.join(self.images_dir_path, image_file_name)
        return image_file_path


class ImSituDatasetBuilder(ExternalDatasetBuilder):
    """ This class builds the image->missing slots number dataset. """

    def __init__(self, root_dir_path, struct_property, indent):
        super(ImSituDatasetBuilder, self).__init__('imsitu', 'English', struct_property, indent)
        self.root_dir_path = root_dir_path
        self.images_dir_path = os.path.join(root_dir_path, 'resized_256')

        self.verb_to_ind_file_path = os.path.join(self.cached_dataset_files_dir, 'imsitu_verb_to_ind')

        self.data_split_strs = ['train', 'dev', 'test']

    def get_struct_data_internal(self):
        verb_to_ind = self.generate_verb_to_ind_mapping()

        # Find the name of the empty slot we want
        property_parts = self.struct_property.split('_')
        assert len(property_parts) == 3
        slot_name = property_parts[1]
        if slot_name == 'agent':
            slot_list = ['agent', 'agents']
        elif slot_name == 'place':
            slot_list = ['place']
        else:
            assert False

        struct_data = []
        for data_split_str in self.data_split_strs:
            with open(os.path.join(self.root_dir_path, data_split_str + '.json')) as fp:
                data = json.load(fp)

            # For each image annotation, we count how many of the frame slots were left empty
            for image_name, annotation in data.items():
                image_id = self.image_name_to_ind(image_name, verb_to_ind)
                for frame in annotation['frames']:
                    relevant_slots = list(set(slot_list).intersection(frame.keys()))
                    empty_slots = int(
                        len(relevant_slots) > 0 and
                        len([x for x in relevant_slots if frame[x] != '']) == 0
                    )
                    struct_data.append((image_id, empty_slots))

        return struct_data

    def generate_verb_to_ind_mapping(self):
        return generate_dataset(self.verb_to_ind_file_path, self.generate_verb_to_ind_mapping_internal)

    def generate_verb_to_ind_mapping_internal(self):
        all_verbs = []
        for data_split_str in self.data_split_strs:
            with open(os.path.join(self.root_dir_path, data_split_str + '.json')) as fp:
                data = json.load(fp)

            image_names = list(data.keys())
            all_verbs += list(set([image_name.split('_')[0] for image_name in image_names]))
        all_verbs = list(set(all_verbs))
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
        return verb_to_ind[verb_name] * VERB_MULT_FACT + int(image_serial_num)
