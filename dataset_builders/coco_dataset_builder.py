from spacy.matcher import Matcher
import os
import json
from dataset_builders.dataset_builder import DatasetBuilder
from dataset_builders.image_path_finder import ImagePathFinder
from utils.general_utils import generate_dataset, for_loop_with_reports
from utils.text_utils import nlp, is_transitive_sentence


class CocoImagePathFinder(ImagePathFinder):

    def __init__(self, data_split_str, train_images_dir_path, val_images_dir_path):
        super(CocoImagePathFinder, self).__init__()

        self.data_split_str = data_split_str
        self.train_images_dir_path = train_images_dir_path
        self.val_images_dir_path = val_images_dir_path

    def get_image_path(self, image_id):
        image_file_name = 'COCO_' + self.data_split_str + '2014_000000' + '{0:06d}'.format(image_id) + '.jpg'
        if self.data_split_str == 'train':
            images_dir_path = self.train_images_dir_path
        elif self.data_split_str == 'val':
            images_dir_path = self.val_images_dir_path
        image_path = os.path.join(images_dir_path, image_file_name)

        return image_path


class CocoDatasetBuilder(DatasetBuilder):
    """ This is the dataset builder class for the MSCOCO dataset, described in the paper
        'Microsoft COCO: Common Objects in Context' by Lin et al.
        Something weird about COCO: They published 3 splits: train, val, test, but they didn't provide labels for the
        test split. So we're going to ignore the test set.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(CocoDatasetBuilder, self).__init__('coco', data_split_str, struct_property, indent)
        self.root_dir_path = root_dir_path

        self.train_val_annotations_dir = 'train_val_annotations2014'

        train_captions_file_path_suffix = os.path.join(self.train_val_annotations_dir, 'captions_train2014.json')
        self.train_captions_file_path = os.path.join(root_dir_path, train_captions_file_path_suffix)
        val_captions_file_path_suffix = os.path.join(self.train_val_annotations_dir, 'captions_val2014.json')
        self.val_captions_file_path = os.path.join(root_dir_path, val_captions_file_path_suffix)

        self.train_images_dir_path = os.path.join(root_dir_path, 'train2014')
        self.val_images_dir_path = os.path.join(root_dir_path, 'val2014')

        self.passive_images_file_path = os.path.join(self.cached_dataset_files_dir,
                                                     'coco_passive_images_' + self.data_split_str)
        self.nlp_data_file_path = os.path.join(self.cached_dataset_files_dir,
                                               'coco_nlp_data_' + self.data_split_str)
        self.transitivity_file_path = os.path.join(self.cached_dataset_files_dir,
                                                   'coco_transitivity_' + self.data_split_str)

        self.nlp_data = None

    def get_caption_data(self):
        if self.data_split_str == 'train':
            external_caption_file_path = self.train_captions_file_path
        elif self.data_split_str == 'val':
            external_caption_file_path = self.val_captions_file_path
        caption_fp = open(external_caption_file_path, 'r')
        caption_data = json.load(caption_fp)['annotations']
        return caption_data

    """ NLP data: the nlp data (spaCy analysis of each caption) is expensive to generate. So we'll do it once and cache
        it for future uses.
    """

    def generate_nlp_data(self):
        if self.nlp_data is None:
            self.nlp_data = generate_dataset(self.nlp_data_file_path, self.generate_nlp_data_internal)

    def generate_nlp_data_internal(self):
        self.log_print('Generating nlp data...')
        caption_data = self.get_caption_data()
        self.nlp_data = []

        self.increment_indent()
        for_loop_with_reports(caption_data, len(caption_data), 10000, self.collect_nlp_data_from_caption,
                              self.caption_report)
        self.decrement_indent()

        self.log_print('Finished generating nlp data')
        return self.nlp_data

    def collect_nlp_data_from_caption(self, index, sample, should_print):
        caption = sample['caption']
        self.nlp_data.append(nlp(caption))

    def caption_report(self, index, iterable_size, time_from_prev_checkpoint):
        self.log_print('Starting caption ' + str(index) +
                       ' out of ' + str(iterable_size) +
                       ', time from previous checkpoint ' + str(time_from_prev_checkpoint))

    """ Passive dataset: maps image ids to list of boolean stating whether each caption is passive. """

    def generate_passive_dataset(self):
        matcher = Matcher(nlp.vocab)
        passive_rule = [
            {'DEP': 'nsubjpass'},
            {'DEP': 'aux', 'OP': '*'},
            {'DEP': 'auxpass'},
            {'TAG': 'VBN'}
        ]
        matcher.add('Passive', [passive_rule])

        caption_data = self.get_caption_data()
        self.generate_nlp_data()
        passive_dataset = []

        for i in range(len(caption_data)):
            image_id = caption_data[i]['image_id']
            nlp_data = self.nlp_data[i]
            passive_dataset.append((image_id, len(matcher(nlp_data)) > 0))

        return passive_dataset

    """ Transitivity dataset: maps image ids to list of boolean stating whether the main verb in each caption is
        passive.
    """

    def generate_transitivity_dataset(self):
        caption_data = self.get_caption_data()
        self.generate_nlp_data()
        transitivity_dataset = []

        for i in range(len(caption_data)):
            image_id = caption_data[i]['image_id']
            nlp_data = self.nlp_data[i]
            roots = [token for token in nlp_data if token.dep_ == 'ROOT']
            if len(roots) != 1:
                # We don't know how to deal with zero or multiple roots, for now
                return

            root = roots[0]
            if root.pos_ != 'VERB':
                # We're not interested in non-verb roots
                return

            transitivity_dataset.append(image_id, is_transitive_sentence(nlp_data))

        return transitivity_dataset

    def create_struct_data_internal(self):
        if self.struct_property == 'passive':
            self.log_print('Generating passive dataset...')
            self.increment_indent()
            struct_data = self.generate_passive_dataset()
            self.decrement_indent()
        elif self.struct_property == 'transitivity':
            self.log_print('Generating transitivity dataset...')
            self.increment_indent()
            struct_data = self.generate_transitivity_dataset()
            self.decrement_indent()

        return struct_data

    def create_image_path_finder(self):
        return CocoImagePathFinder(self.data_split_str, self.train_images_dir_path, self.val_images_dir_path)
