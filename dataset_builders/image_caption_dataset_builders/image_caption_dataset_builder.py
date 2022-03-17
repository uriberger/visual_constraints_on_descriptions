from recognizers_number import recognize_number, Culture

import os
import abc

from dataset_builders.dataset_builder import DatasetBuilder
from utils.general_utils import generate_dataset, for_loop_with_reports
from utils.text_utils import TextUtils


class ImageCaptionDatasetBuilder(DatasetBuilder):
    """ This is the dataset builder class for all datasets of image, caption pairs. """

    def __init__(self, root_dir_path, name, data_split_str, struct_property, indent):
        super(ImageCaptionDatasetBuilder, self).__init__(name, data_split_str, struct_property, indent)
        self.root_dir_path = root_dir_path

        self.dump_caption_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{name}_{TextUtils.get_language()}_dump_captions_{self.data_split_str}.txt'
        )

        self.nlp_data_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{name}_{TextUtils.get_language()}_nlp_data_{self.data_split_str}'
        )

        self.gt_classes_data_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_gt_classes_data_{self.data_split_str}'
        )

        self.nlp_data = None

    """ Return a list of dictionaries with 'image_id' and 'caption' entries. """

    @abc.abstractmethod
    def get_caption_data(self):
        return

    """ Return a mapping from image_id and a list of gt classes instantiated in the image. """

    def get_gt_classes_data(self):
        return generate_dataset(self.gt_classes_data_file_path, self.get_gt_classes_data_internal)

    @abc.abstractmethod
    def get_gt_classes_data_internal(self):
        return {}

    """ Return a mapping from gt class index to gt class name. """

    def get_class_mapping(self):
        return {}

    """ Dump all the captions in the dataset to a text file. """

    def dump_captions(self):
        self.log_print('Dumping captions...')
        caption_data = self.get_caption_data()

        with open(self.dump_caption_file_path, 'w') as dump_caption_fp:
            for sample in caption_data:
                dump_caption_fp.write(sample['caption'] + '\n')

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
        self.nlp_data.append(TextUtils.get_nlp()(caption))

    def caption_report(self, index, iterable_size, time_from_prev_checkpoint):
        self.log_print('Starting caption ' + str(index) +
                       ' out of ' + str(iterable_size) +
                       ', time from previous checkpoint ' + str(time_from_prev_checkpoint))

    """ Passive dataset: maps image ids to list of boolean stating whether each caption is passive. """

    def generate_passive_dataset(self):
        caption_data = self.get_caption_data()
        self.generate_nlp_data()
        passive_dataset = []
        matcher = TextUtils.get_passive_matcher()

        for i in range(len(caption_data)):
            image_id = caption_data[i]['image_id']
            nlp_data = self.nlp_data[i]

            # We're only in interested in captions with a single root which is a verb
            roots = [x for x in nlp_data if x.dep_ == 'ROOT']
            if len(roots) != 1 or roots[0].pos_ != 'VERB':
                continue

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

            transitivity_dataset.append((image_id, TextUtils.is_transitive_sentence(nlp_data)))

        return transitivity_dataset

    """ Negation dataset: maps image ids to list of boolean stating whether each caption uses negation. """

    def generate_negation_dataset(self):
        english_negation_words = set([
            'not', 'isnt', 'arent', 'doesnt', 'dont', 'cant', 'cannot', 'shouldnt', 'wont', 'wouldnt', 'no', 'none',
            'nobody', 'nothing', 'nowhere', 'neither', 'nor', 'never', 'without'
        ])
        german_negation_words = set([
            'nicht', 'kein', 'nie', 'niemals', 'niemand', 'nirgendwo', 'nirgendwohin', 'nirgends', 'weder', 'ohne',
            'nein', 'nichts', 'nee'
        ])
        chinese_negation_words = set([
            '不', '没', '没有'
        ])
        french_negation_words = set([
            'sans', 'rien', 'jamais'
        ])
        japanese_negation_words = set([
            'ない', 'ません', 'なかった', 'でした', 'いいえ'
        ])

        caption_data = self.get_caption_data()
        self.generate_nlp_data()
        negation_dataset = []
        language = TextUtils.get_language()

        for i in range(len(caption_data)):
            sample = caption_data[i]
            image_id = sample['image_id']
            caption = sample['caption']
            if language == 'English':
                tokenized_caption = TextUtils.tokenize_and_clean(caption)
                negation_words_in_caption = english_negation_words.intersection(tokenized_caption)
                negation_dataset.append((image_id, int(len(negation_words_in_caption) > 0)))
            elif language == 'German':
                sample_nlp_data = self.nlp_data[i]
                negation_words_in_caption = german_negation_words.intersection([
                    x.lemma_.lower() for x in sample_nlp_data
                ])
                negation_dataset.append((image_id, int(len(negation_words_in_caption) > 0)))
            elif language == 'Chinese':
                if len([x for x in self.nlp_data[i] if x.dep_ == 'neg']) > 0 or \
                        len([x for x in self.nlp_data[i] if x.text == '没有']) > 0:
                    negation_dataset.append((image_id, 1))
                else:
                    negation_dataset.append((image_id, 0))
            elif language == 'Japanese':
                tokenized_caption = TextUtils.tokenize_and_clean(caption)
                negation_words_in_caption = japanese_negation_words.intersection(tokenized_caption)
                negation_dataset.append((image_id, int(len(negation_words_in_caption) > 0)))

        return negation_dataset

    """ Numbers dataset: maps image ids to list of boolean stating whether each caption contains numbers. """

    def generate_numbers_dataset(self):
        caption_data = self.get_caption_data()
        numbers_dataset = []
        language = TextUtils.get_language()
        if language == 'English':
            culture_language = Culture.English
        elif language == 'French':
            culture_language = Culture.French
        elif language == 'Japanese':
            culture_language = Culture.Japanese
        elif language == 'German':
            culture_language = Culture.German
        elif language == 'Chinese':
            culture_language = Culture.Chinese
        else:
            self.log_print(f'Numbers property not implemented for language {language}')
            assert False

        for sample in caption_data:
            image_id = sample['image_id']
            caption = sample['caption']

            # The recognizers_number package has some bug in logographic languages: it doesn't work if there are no
            # spaces between words
            if language in ['Japanese', 'Chinese']:
                caption = ' '.join([char for char in caption])

            numbers_dataset.append((image_id, int(len(recognize_number(caption, culture_language)) > 0)))

        return numbers_dataset

    def create_struct_data_internal(self):
        if self.struct_property == 'passive':
            struct_data = self.generate_passive_dataset()
        elif self.struct_property == 'transitivity':
            struct_data = self.generate_transitivity_dataset()
        elif self.struct_property == 'negation':
            struct_data = self.generate_negation_dataset()
        elif self.struct_property == 'numbers':
            struct_data = self.generate_numbers_dataset()

        return struct_data

    @abc.abstractmethod
    def create_image_path_finder(self):
        return
