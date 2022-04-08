from recognizers_number import recognize_number, Culture

import os
import abc
import jieba
import random

from dataset_builders.single_dataset_builder import SingleDatasetBuilder
from utils.general_utils import generate_dataset, for_loop_with_reports
from utils.text_utils import TextUtils


class ImageCaptionDatasetBuilder(SingleDatasetBuilder):
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

        self.gt_bboxes_data_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_gt_bboxes_data_{self.data_split_str}'
        )

        self.parsed_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{name}_{TextUtils.get_language()}_{data_split_str}_parsed.txt'
        )

        self.train_val_split_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{name}_{TextUtils.get_language()}_train_val_split'
        )

        self.nlp_data = None

    """ Return a list of dictionaries with 'image_id' and 'caption' entries. """

    @abc.abstractmethod
    def get_caption_data(self):
        return

    """ Return a mapping from image_id to a list of gt classes instantiated in the image. """

    def get_gt_classes_data(self):
        return generate_dataset(self.gt_classes_data_file_path, self.get_gt_classes_data_internal)

    """ Return a mapping from image_id to a list of gt bounding boxes in the image. """

    def get_gt_bboxes_data(self):
        return generate_dataset(self.gt_bboxes_data_file_path, self.get_gt_bboxes_data_internal)

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
                self.log_print('Did you run the parse/prepare_passive_data.sh script?')
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
        language = TextUtils.get_language()

        if language == 'French':
            transitivity_dataset = self.generate_french_transitivity_dataset()
        else:
            transitivity_dataset = self.generate_non_french_transitivity_dataset()

        return transitivity_dataset

    def generate_french_transitivity_dataset(self):
        caption_data = self.get_caption_data()
        transitivity_dataset = []

        if not os.path.isfile(self.parsed_file_path):
            self.log_print(f'Couldn\'t find the parsed file in path {self.parsed_file_path}. Stopping!')
            self.log_print('Did you run the parse/parse.sh script?')
            assert False
        with open(self.parsed_file_path, 'r', encoding='utf-8') as fp:
            sample_ind = -1
            new_sentence = True
            for line in fp:
                if new_sentence:
                    sample_ind += 1
                    image_id = caption_data[sample_ind]['image_id']
                    obj_token_heads = []
                    root_ind = -1
                    new_sentence = False
                if line == '\n':
                    # Finished current caption
                    if root_ind >= 0:
                        # If we're here there's exactly one root
                        is_transitive = root_ind in obj_token_heads
                        transitivity_dataset.append((image_id, int(is_transitive)))
                    new_sentence = True
                else:
                    split_line = line.split('\t')
                    cur_token_ind = int(split_line[0])
                    pos_tag = split_line[5]
                    head_ind = int(split_line[9])
                    dep_tag = split_line[11]
                    # Check if current token is the root and is a verb
                    if dep_tag == 'root' and pos_tag.startswith('V'):
                        if root_ind == -1:
                            # This is the first root we found
                            root_ind = cur_token_ind
                        elif root_ind >= 0:
                            # This is the second root we found: we don't know how to handle these sentences
                            root_ind == -2
                    elif dep_tag == 'obj':
                        obj_token_heads.append(head_ind)
            assert sample_ind == len(caption_data) - 1

        return transitivity_dataset

    def generate_non_french_transitivity_dataset(self):
        caption_data = self.get_caption_data()
        transitivity_dataset = []

        self.generate_nlp_data()

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

            transitivity_dataset.append((image_id, int(TextUtils.is_transitive_sentence(nlp_data))))

        return transitivity_dataset

    """ Negation dataset: maps image ids to list of boolean stating whether each caption uses negation. """

    def generate_negation_dataset(self):
        language = TextUtils.get_language()

        if language == 'French':
            negation_dataset = self.generate_french_negation_dataset()
        else:
            negation_dataset = self.generate_non_french_negation_dataset()

        return negation_dataset

    def generate_french_negation_dataset(self):
        french_pos_neg_words = set([
            'pas', 'personne', 'aucun', 'aucune'
        ])
        french_negation_words = set([
            'ni', 'jamais', 'rien', 'non', 'sans'
        ])
        french_negation_phrases = [
            ['nulle', 'part']
        ]

        caption_data = self.get_caption_data()
        negation_dataset = []

        if not os.path.isfile(self.parsed_file_path):
            self.log_print(f'Couldn\'t find the parsed file in path {self.parsed_file_path}. Stopping!')
            self.log_print('Did you run the parse/parse.sh script?')
            assert False
        with open(self.parsed_file_path, 'r', encoding='utf-8') as fp:
            sample_ind = -1
            new_sentence = True
            for line in fp:
                if new_sentence:
                    sample_ind += 1
                    image_id = caption_data[sample_ind]['image_id']
                    caption = caption_data[sample_ind]['caption']
                    contains_neg_pos_word_in_negative_form = False
                    new_sentence = False
                if line == '\n':
                    # Finished current caption
                    ''' French negation has 3 types of elements:
                    1. Negation words: words that are always considered a negation.
                    2. Negation phrases: a sequence of words that are considered a negation.
                    3. Negative-positive words: words that can occur both in a negative and a
                    positive form.
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
                    if contains_neg_pos_word_in_negative_form:
                        negation = True
                    negation_dataset.append((image_id, int(negation)))

                    new_sentence = True
                else:
                    split_line = line.split('\t')
                    cur_token_str = split_line[1]
                    cur_token_feat = split_line[7]
                    if cur_token_str in french_pos_neg_words and 's=neg' in cur_token_feat:
                        contains_neg_pos_word_in_negative_form = True
            assert sample_ind == len(caption_data) - 1

        return negation_dataset

    def generate_non_french_negation_dataset(self):
        english_negation_words = set([
            'not', 'isnt', 'arent', 'doesnt', 'dont', 'cant', 'cannot', 'shouldnt', 'wont', 'wouldnt', 'no', 'none',
            'nobody', 'nothing', 'nowhere', 'neither', 'nor', 'never', 'without',
            # New words
            'nope'
        ])
        german_negation_words = set([
            'nicht', 'kein', 'nie', 'niemals', 'niemand', 'nirgendwo', 'nirgendwohin', 'nirgends', 'weder', 'ohne',
            'nein', 'nichts', 'nee'
        ])
        chinese_negation_words = set([
            '不', '不是', '不能', '不可以', '没', '没有', '没什么', '从不', '并不', '从没有', '并没有', '无人', '无处', '无', '别',
            '绝不'
        ])
        japanese_negation_words = set([
            'ない',  # When it's at the end of the sentence it's negation, otherwise not sure
            'ませ',  # Mase, but I need masen ません
            'なし',  # Basically ok, but sometimes wrong for example 砂漠に似た乾燥地帯に馬が群れをなしている
            'なかっ',  # Ok
            'いいえ'
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
            elif language == 'Chinese':
                ''' Jieba is a better tokenizer for Chinese than spaCy. '''
                tokenized_caption = list(jieba.cut(caption, cut_all=False))
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

    """ Root pos dataset: maps image ids to list of boolean stating whether each caption's root is a noun (0) or a verb
        (1).
    """

    def generate_root_pos_dataset(self):
        language = TextUtils.get_language()

        if language == 'French':
            root_pos_dataset = self.generate_french_root_pos_dataset()
        else:
            root_pos_dataset = self.generate_non_french_root_pos_dataset()

        return root_pos_dataset

    def generate_french_root_pos_dataset(self):
        caption_data = self.get_caption_data()
        root_pos_dataset = []

        if not os.path.isfile(self.parsed_file_path):
            self.log_print(f'Couldn\'t find the parsed file in path {self.parsed_file_path}. Stopping!')
            self.log_print('Did you run the parse/parse.sh script?')
            assert False
        with open(self.parsed_file_path, 'r', encoding='utf-8') as fp:
            sample_ind = -1
            new_sentence = True
            for line in fp:
                if new_sentence:
                    sample_ind += 1
                    image_id = caption_data[sample_ind]['image_id']
                    roots_pos = []
                    new_sentence = False
                if line == '\n':
                    # Finished current caption
                    if len(roots_pos) != 1:
                        # We don't know how to deal with zero or multiple roots, for now
                        continue

                    # If we're here there's exactly one root
                    root_pos = roots_pos[0]
                    if root_pos.startswith('N') or root_pos == 'CLS':
                        val = 0  # Noun
                    elif root_pos.startswith('V'):
                        val = 1  # Verb
                    else:
                        # We're only interested in verb or noun roots
                        continue

                    root_pos_dataset.append((image_id, val))
                    new_sentence = True
                else:
                    split_line = line.split('\t')
                    dep_tag = split_line[11]
                    if dep_tag == 'root':
                        pos_tag = split_line[5]
                        roots_pos.append(pos_tag)
            assert sample_ind == len(caption_data) - 1

        return root_pos_dataset

    def generate_non_french_root_pos_dataset(self):
        caption_data = self.get_caption_data()
        self.generate_nlp_data()
        root_pos_dataset = []

        for i in range(len(caption_data)):
            image_id = caption_data[i]['image_id']
            nlp_data = self.nlp_data[i]
            roots = [token for token in nlp_data if token.dep_ == 'ROOT']
            if len(roots) != 1:
                # We don't know how to deal with zero or multiple roots, for now
                continue

            root = roots[0]
            if root.pos_ in ['NOUN', 'PRON', 'PROPN']:
                val = 0
            elif root.pos_ == 'VERB':
                val = 1
            else:
                # We're only interested in verb or noun roots
                continue

            root_pos_dataset.append((image_id, val))

        return root_pos_dataset

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
        elif self.struct_property == 'root_pos':
            struct_data = self.generate_root_pos_dataset()
        elif self.struct_property == 'spatial_relations':
            struct_data = self.generate_spatial_relations_dataset()

        return struct_data

    @abc.abstractmethod
    def create_image_path_finder(self):
        return

    # Some of the datasets are not naturally divided to splits. Create functionality to do that

    def get_all_image_ids(self):
        caption_data = self.get_caption_data()
        return list(set([x['image_id'] for x in caption_data]))

    def create_train_val_split(self):
        all_image_ids = self.get_all_image_ids()
        train_split_size = int(len(all_image_ids) * 0.8)
        train_split = random.sample(all_image_ids, train_split_size)
        train_split_dict = {x: True for x in train_split}
        val_split = [x for x in all_image_ids if x not in train_split_dict]

        return {'train': train_split, 'val': val_split}

    def get_image_ids_for_split(self):
        if self.data_split_str == 'all':
            return self.get_all_image_ids()
        else:
            split_to_image_ids = generate_dataset(self.train_val_split_file_path, self.create_train_val_split)
            return split_to_image_ids[self.data_split_str]
