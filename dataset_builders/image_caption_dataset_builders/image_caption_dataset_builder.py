from spacy.matcher import Matcher
from recognizers_number import recognize_number, Culture

import os
import abc

from dataset_builders.dataset_builder import DatasetBuilder
from utils.general_utils import generate_dataset, for_loop_with_reports
from utils.text_utils import nlp, is_transitive_sentence, tokenize_and_clean


class ImageCaptionDatasetBuilder(DatasetBuilder):
    """ This is the dataset builder class for all datasets of image, caption pairs. """

    def __init__(self, root_dir_path, name, data_split_str, struct_property, indent):
        super(ImageCaptionDatasetBuilder, self).__init__(name, data_split_str, struct_property, indent)
        self.root_dir_path = root_dir_path

        self.nlp_data_file_path = os.path.join(self.cached_dataset_files_dir,
                                               name + '_nlp_data_' + self.data_split_str)

        self.nlp_data = None

    """ Return a list of dictionaries with 'image_id' and 'caption' entries. """

    @abc.abstractmethod
    def get_caption_data(self):
        return

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
            passive_dataset.append((image_id, int(len(matcher(nlp_data)) > 0)))

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
                continue

            root = roots[0]
            if root.pos_ != 'VERB':
                # We're not interested in non-verb roots
                continue

            transitivity_dataset.append((image_id, is_transitive_sentence(nlp_data)))

        return transitivity_dataset

    """ Negation dataset: maps image ids to list of boolean stating whether each caption uses negation. """

    def generate_negation_dataset(self):
        negation_words = set(['not', 'isnt', 'arent', 'doesnt', 'dont', 'cant', 'cannot', 'shouldnt', 'wont', 'wouldnt',
                              'no', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor', 'never', 'without'])

        caption_data = self.get_caption_data()
        negation_dataset = []

        for sample in caption_data:
            image_id = sample['image_id']
            caption = sample['caption']
            tokenized_caption = tokenize_and_clean(caption)
            negation_words_in_caption = negation_words.intersection(tokenized_caption)
            negation_dataset.append((image_id, int(len(negation_words_in_caption) > 0)))

        return negation_dataset

    """ Numbers dataset: maps image ids to list of boolean stating whether each caption contains numbers. """

    def generate_numbers_dataset(self):
        caption_data = self.get_caption_data()
        numbers_dataset = []

        for sample in caption_data:
            image_id = sample['image_id']
            caption = sample['caption']
            numbers_dataset.append((image_id, int(len(recognize_number(caption, Culture.English)) > 0)))

        return numbers_dataset

    def create_struct_data_internal(self):
        self.log_print(f'Generating {self.name} {self.struct_property} dataset...')
        self.increment_indent()
        if self.struct_property == 'passive':
            struct_data = self.generate_passive_dataset()
        elif self.struct_property == 'transitivity':
            struct_data = self.generate_transitivity_dataset()
        elif self.struct_property == 'negation':
            struct_data = self.generate_negation_dataset()
        elif self.struct_property == 'numbers':
            struct_data = self.generate_numbers_dataset()
        self.decrement_indent()

        return struct_data

    @abc.abstractmethod
    def create_image_path_finder(self):
        return
