from recognizers_number import recognize_number, Culture

import os
import abc
import jieba

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

        with open(self.dump_caption_file_path, 'w', encoding='utf8') as dump_caption_fp:
            for sample in caption_data:
                dump_caption_fp.write(sample['caption'].strip().replace('\n', '.') + '\n')

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

    # def generate_passive_dataset_old(self):
    #     caption_data = self.get_caption_data()
    #     self.generate_nlp_data()
    #     passive_dataset = []
    #     matcher = TextUtils.get_passive_matcher()
    #
    #     for i in range(len(caption_data)):
    #         image_id = caption_data[i]['image_id']
    #         nlp_data = self.nlp_data[i]
    #
    #         # We're only in interested in captions with a single root which is a verb
    #         roots = [x for x in nlp_data if x.dep_ == 'ROOT']
    #         if len(roots) != 1 or roots[0].pos_ != 'VERB':
    #             continue
    #
    #         passive_dataset.append((image_id, int(len(matcher(nlp_data)) > 0)))
    #
    #     return passive_dataset

    def generate_passive_dataset(self):
        language = TextUtils.get_language()
        caption_data = self.get_caption_data()
        passive_dataset = []

        if language in ['English', 'German', 'French']:
            # For English, German and French we have an external tool that identifies passive for us
            cached_dataset_files_dir_name = self.cached_dataset_files_dir
            tmv_out_file_name = f'tmv_out_{language}_{self.name}.verbs'
            tmv_out_file_path = os.path.join(cached_dataset_files_dir_name, tmv_out_file_name)
            if not os.path.isfile(tmv_out_file_path):
                self.log_print(f'Couldn\'t find the tmv out file in path {tmv_out_file_path}. Stopping!')
                self.log_print('Did you run the english_german_french_passive/prepare_passive_data.sh script?')
                assert False

            with open(tmv_out_file_path, 'r', encoding='utf-8') as tmv_out_fp:
                prev_caption_ind = None
                passive_indicator = 0
                for line in tmv_out_fp:
                    line_parts = line.strip().split('\t')
                    caption_ind = int(line_parts[0]) - 1
                    if caption_ind != prev_caption_ind:
                        ''' A caption my have multiple lines, but successive. If we found a new caption ind, it means we
                         finished the previous caption, and we should store the results. '''
                        if prev_caption_ind is not None:
                            image_id = caption_data[prev_caption_ind]['image_id']
                            passive_dataset.append((image_id, passive_indicator))
                        prev_caption_ind = caption_ind
                        passive_indicator = 0
                    if language == 'English':
                        voice_index = -5
                    elif language == 'German':
                        voice_index = -3
                    elif language == 'French':
                        voice_index = -4
                    if line_parts[voice_index] == 'passive':
                        passive_indicator = 1

                # Now we need to store results for the last caption
                image_id = caption_data[caption_ind]['image_id']
                passive_dataset.append((image_id, passive_indicator))
        else:
            self.generate_nlp_data()

            if language == 'Japanese':
                passive_indicators = set(['れる', 'られる'])

            for i in range(len(caption_data)):
                sample = caption_data[i]
                sample_nlp_data = self.nlp_data[i]
                caption = sample['caption']
                image_id = sample['image_id']

                # We're only in interested in captions with at least a single verb
                verbs = [x for x in sample_nlp_data if x.pos_ == 'VERB']
                if len(verbs) == 0:
                    continue

                if language == 'Japanese':
                    lemmas = [x.lemma_ for x in sample_nlp_data]
                    sample_passive_indicators = passive_indicators.intersection(lemmas)
                    passive_dataset.append((image_id, int(len(sample_passive_indicators) > 0)))
                elif language == 'Chinese':
                    passive_dataset.append((image_id, int('被' in caption)))

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
            'nobody', 'nothing', 'nowhere', 'neither', 'nor', 'never', 'without',
            # New words
            'nope'
        ])
        german_negation_words = set([
            'nicht', 'kein', 'keiner', 'nie', 'niemals', 'niemand', 'nirgendwo', 'nirgendwohin', 'nirgends', 'weder',
            'ohne', 'nein', 'nichts', 'nee', 'noch'
        ])
        french_ne_negation_words = set([
            'aucun', 'aucune', 'ni', 'pas', 'personne'
        ])
        french_negation_words = set([
            'jamais', 'rien', 'non', 'sans', 'nan'
        ])
        french_negation_phrases = [
            ['nulle', 'part']
        ]
        chinese_negation_words = set([
            '不', '不是', '不能', '不可以', '没', '没有', '没什么', '从不', '并不', '从没有', '并没有', '无人', '无处', '无', '别',
            '绝不'
        ])
        japanese_negation_words = set([
            'ない', 'ませ', 'なし', 'なかっ', 'いいえ'
        ])

        caption_data = self.get_caption_data()
        negation_dataset = []
        language = TextUtils.get_language()
        if language == 'German':
            self.generate_nlp_data()

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
            elif language == 'French':
                ''' French negation has 3 types of elements:
                1. Negation words: words that, without any other words, are considered a negation.
                2. Negation phrases: a sequence of words that are considered a negation.
                3. 'ne' words: words that, when co-occurs with the 'ne' word, are considered a negation.
                I should check all three cases. '''
                tokenized_caption = TextUtils.tokenize(caption)
                negation = False
                # Case 1
                if len(french_negation_words.intersection(tokenized_caption)) > 0:
                    negation = True
                # Case 2
                if len([phrase for phrase in french_negation_phrases
                        if TextUtils.phrase_in_sent(tokenized_caption, phrase)]) > 0:
                    negation = True
                # Case 3
                if ('ne' in tokenized_caption or 'n\'' in tokenized_caption) and \
                        len(french_ne_negation_words.intersection(tokenized_caption)) > 0:
                    negation = True
                negation_dataset.append((image_id, int(negation)))
            elif language == 'Chinese':
                ''' Jieba is a better tokenizer for Chinese than spaCy. '''
                tokenized_caption = list(jieba.cut(caption, cur_all=False))
                negation_words_in_caption = chinese_negation_words.intersection(tokenized_caption)
                negation_dataset.append((image_id, int(len(negation_words_in_caption) > 0)))
            elif language == 'Japanese':
                tokenized_caption = TextUtils.tokenize(caption)
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

        nlp = TextUtils.get_nlp()

        for sample in caption_data:
            image_id = sample['image_id']
            caption = sample['caption']

            # The recognizers_number package has some bug in logographic languages: it doesn't work if there are no
            # spaces between words
            if language in ['Japanese', 'Chinese']:
                caption = ' '.join([char for char in caption])

            numbers_list = recognize_number(caption, culture_language)
            ''' In French, German and Chinese the words for "one" and "a" are the same.
            So we don't want to consider those words as numbers. '''
            if language == 'French':
                numbers_list = [x for x in numbers_list
                                if [y.lemma_ for y in nlp(x.text)] not in [['un'], ['un', 'sur', 'un']]]
            if language == 'German':
                # TODO: Should the 'eins' word be removed as well? I don't think so
                numbers_list = [x for x in numbers_list
                                if [y.lemma_ for y in nlp(x.text)][0] not in ['ein', 'einer', 'einen']]
            if language == 'Chinese':
                numbers_list = [x for x in numbers_list if x.text != '一']

            numbers_dataset.append((image_id, int(len(numbers_list) > 0)))

        return numbers_dataset

    """ Spatial relations dataset: maps image ids to list of boolean stating whether each caption contains a spatial
        relation word.
    """

    def generate_spatial_relations_dataset(self):
        english_spatial_words = set([
            'between', 'on', 'outside', 'inside', 'near', 'upon', 'over', 'beside', 'below', 'under', 'across',
            'behind', 'in', 'above',
            'toward', 'beyond',
            # Question?
            'into', 'alongside',
            # Remove
            'by', 'out'
        ])

        english_spatial_phrases = [
            ['next', 'to'], ['close', 'to'], ['in', 'front', 'of'], ['opposite', 'of'], ['the', 'left', 'of'],
            ['the', 'right', 'of'], ['middle', 'of'], ['away', 'from'], ['far', 'from'], ['adjacent', 'to']
        ]

        caption_data = self.get_caption_data()
        spatial_dataset = []
        language = TextUtils.get_language()
        for sample in caption_data:
            caption = sample['caption']
            image_id = sample['image_id']
            if language == 'English':
                tokenized_caption = TextUtils.tokenize_and_clean(caption)

                # Spatial words
                spatial_words_in_caption = english_spatial_words.intersection(tokenized_caption)

                # Spatial phrases
                spatial_phrases_in_caption = [phrase for phrase in english_spatial_phrases
                                              if TextUtils.phrase_in_sent(tokenized_caption, phrase)]

                spatial_dataset.append((image_id,
                                        int(len(spatial_words_in_caption) > 0 or len(spatial_phrases_in_caption) > 0)))

        return spatial_dataset

    def create_struct_data_internal(self):
        if self.struct_property == 'passive':
            struct_data = self.generate_passive_dataset()
        elif self.struct_property == 'transitivity':
            struct_data = self.generate_transitivity_dataset()
        elif self.struct_property == 'negation':
            struct_data = self.generate_negation_dataset()
        elif self.struct_property == 'numbers':
            struct_data = self.generate_numbers_dataset()
        elif self.struct_property == 'spatial_relations':
            struct_data = self.generate_spatial_relations_dataset()

        return struct_data

    @abc.abstractmethod
    def create_image_path_finder(self):
        return
