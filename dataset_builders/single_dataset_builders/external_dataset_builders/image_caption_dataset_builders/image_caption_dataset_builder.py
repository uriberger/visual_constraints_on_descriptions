from recognizers_number import recognize_number, Culture

import os
from collections import defaultdict
import abc
import jieba
import csv
import statistics

from dataset_builders.single_dataset_builders.external_dataset_builders.external_dataset_builder \
    import ExternalDatasetBuilder
from utils.general_utils import generate_dataset, for_loop_with_reports
from utils.text_utils import TextUtils


class ImageCaptionDatasetBuilder(ExternalDatasetBuilder):
    """ This is the dataset builder class for all datasets of image, caption pairs. """

    def __init__(self, root_dir_path, name, language, struct_property, indent):
        super(ImageCaptionDatasetBuilder, self).__init__(name, language, struct_property, indent)
        self.root_dir_path = root_dir_path
        self.language = language

        self.dump_caption_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{name}_{language}_dump_captions.txt'
        )

        self.nlp_data_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{name}_{language}_nlp_data'
        )

        self.gt_classes_data_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_gt_classes_data'
        )

        self.gt_bboxes_data_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_gt_bboxes_data'
        )

        self.parsed_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{name}_{language}_parsed.txt'
        )

        self.non_binarized_numbers_dataset_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_{language}_non_binarized_numbers_dataset'
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

    """ NLP data: the nlp data (stanza analysis of each caption) is expensive to generate. So we'll do it once and cache
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
        self.fac = 10
        tmp_caption_data = [caption_data[x*self.fac:(x+1)*self.fac] for x in range(len(caption_data)//self.fac)]
        if len(caption_data) % self.fac != 0:
            tmp_caption_data += [caption_data[len(tmp_caption_data)*self.fac:]]
        caption_data = tmp_caption_data
        for_loop_with_reports(caption_data, len(caption_data), 100, self.collect_nlp_data_from_caption,
                              self.caption_report)
        self.decrement_indent()

        self.log_print('Finished generating nlp data')
        return self.nlp_data

    def collect_nlp_data_from_caption(self, index, sample, should_print):
        captions = '\n\n'.join([TextUtils.prepare_caption_for_stanza(x['caption'], self.language) for x in sample])
        analyzed_captions = TextUtils.get_nlp(self.language)(captions)
        if len(analyzed_captions.sentences) != len(sample):
            print('Wrong number of analyzed sentences in batch ' + str(index))
            assert False
        self.nlp_data += TextUtils.extract_nlp_info(analyzed_captions)

    def caption_report(self, index, iterable_size, time_from_prev_checkpoint):
        self.log_print('Starting caption ' + str(index) +
                       ' out of ' + str(iterable_size) +
                       ', time from previous checkpoint ' + str(time_from_prev_checkpoint))

    """ Passive dataset: maps image ids to list of boolean stating whether each caption is passive. """

    def generate_passive_dataset(self, use_image_ids=True):
        caption_data = self.get_caption_data()
        passive_dataset = []

        if self.language in ['English', 'German', 'French']:
            # For English, German and French we have an external tool that identifies passive for us
            cached_dataset_files_dir_name = self.cached_dataset_files_dir
            tmv_out_file_name = f'tmv_out_{self.extended_name}.verbs'
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
                        ''' A caption may have multiple lines, but successive. If we found a new caption ind, it means we
                         finished the previous caption, and we should store the results. '''
                        if prev_caption_ind is not None:
                            image_id = caption_data[prev_caption_ind]['image_id']
                            caption_id = caption_data[prev_caption_ind]['caption_id']
                            passive_dataset.append((image_id if use_image_ids else caption_id, passive_indicator))
                        prev_caption_ind = caption_ind
                        passive_indicator = 0
                    if self.language == 'English':
                        voice_index = -5
                    elif self.language == 'German':
                        voice_index = -3
                    elif self.language == 'French':
                        voice_index = -4
                    if line_parts[voice_index] == 'passive':
                        passive_indicator = 1

                # Now we need to store results for the last caption
                image_id = caption_data[caption_ind]['image_id']
                caption_id = caption_data[caption_ind]['caption_id']
                passive_dataset.append((image_id if use_image_ids else caption_id, passive_indicator))
        else:
            self.generate_nlp_data()

            if self.language == 'Japanese':
                passive_indicators = set(['れる', 'られる'])

            for i in range(len(caption_data)):
                sample = caption_data[i]
                sample_nlp_data = self.nlp_data[i]
                caption = sample['caption']
                image_id = sample['image_id']
                caption_id = sample['caption_id']

                # We're only in interested in captions with at least a single verb
                verbs = [x for x in sample_nlp_data if x['pos'] == 'VERB']
                if len(verbs) == 0:
                    continue

                if self.language == 'Japanese':
                    lemmas = [x['lemma'] for x in sample_nlp_data]
                    sample_passive_indicators = passive_indicators.intersection(lemmas)
                    passive_indicator = int(len(sample_passive_indicators) > 0)
                elif self.language == 'Chinese':
                    # For Chinese we search for the passive indicator (被). Just need to rule out the cases where
                    # it's part of another word (被子)
                    passive_indicator_inds = [i for i in range(len(caption))
                                              if caption[i] == '被' and (i == len(caption) - 1 or caption[i+1] != '子')]
                    passive_indicator = int(len(passive_indicator_inds) > 0)
                
                passive_dataset.append((image_id if use_image_ids else caption_id, passive_indicator))

        return passive_dataset

    """ Transitivity dataset: maps image ids to list of boolean stating whether the main verb in each caption is
        passive.
    """

    def generate_transitivity_dataset(self, use_image_ids=True):
        if self.language == 'French':
            transitivity_dataset = self.generate_transitivity_dataset_using_mate(use_image_ids)
        else:
            transitivity_dataset = self.generate_transitivity_dataset_using_stanza(use_image_ids)

        return transitivity_dataset

    def generate_transitivity_dataset_using_mate(self, use_image_ids=True):
        caption_data = self.get_caption_data()
        transitivity_dataset = []

        if not os.path.isfile(self.parsed_file_path):
            self.log_print(f'Couldn\'t find the parsed file in path {self.parsed_file_path}. Stopping!')
            self.log_print('Did you run the parse/parse.sh script?')
            assert False

        if self.language == 'French':
            direct_obj_dep_tag = 'obj'
            root_dep_tag = 'root'
            be_lemma = 'être'
        elif self.language == 'German':
            direct_obj_dep_tag = 'OA'
            root_dep_tag = '--'
            be_lemma = 'sein'

        with open(self.parsed_file_path, 'r', encoding='utf-8') as fp:
            sample_ind = -1
            new_sentence = True
            for line in fp:
                if new_sentence:
                    sample_ind += 1
                    image_id = caption_data[sample_ind]['image_id']
                    caption_id = caption_data[sample_ind]['caption_id']
                    obj_token_heads = []
                    root_ind = -1
                    new_sentence = False
                if line == '\n':
                    # Finished current caption
                    if root_ind >= 0:
                        # If we're here there's exactly one root
                        is_transitive = root_ind in obj_token_heads and root_lemma != be_lemma
                        transitivity_dataset.append((image_id if use_image_ids else caption_id, int(is_transitive)))
                    new_sentence = True
                else:
                    split_line = line.split('\t')
                    cur_token_ind = int(split_line[0])
                    cur_token_lemma = split_line[3].lower()
                    pos_tag = split_line[5]
                    head_ind = int(split_line[9])
                    dep_tag = split_line[11]
                    # Check if current token is the root and is a verb
                    if dep_tag == root_dep_tag and pos_tag.startswith('V'):
                        if root_ind == -1:
                            # This is the first root we found
                            root_ind = cur_token_ind
                            root_lemma = cur_token_lemma
                        elif root_ind >= 0:
                            # This is the second root we found: we don't know how to handle these sentences
                            root_ind == -2
                    elif dep_tag == direct_obj_dep_tag:
                        obj_token_heads.append(head_ind)
            assert sample_ind == len(caption_data) - 1

        return transitivity_dataset

    def generate_transitivity_dataset_using_stanza(self, use_image_ids=True):
        caption_data = self.get_caption_data()
        transitivity_dataset = []

        self.generate_nlp_data()

        for i in range(len(caption_data)):
            image_id = caption_data[i]['image_id']
            caption_id = caption_data[i]['caption_id']
            nlp_data = self.nlp_data[i]
            roots = [token for token in nlp_data if token['dep'].lower() == 'root']
            if len(roots) != 1:
                # We don't know how to deal with zero or multiple roots, for now
                continue

            root = roots[0]
            if root['pos'] != 'VERB':
                # We're not interested in non-verb roots
                continue

            if TextUtils.is_existential_sentence(nlp_data, self.language):
                # The 'be' verb is not really an action, so we filter these cases
                continue

            transitivity_dataset.append((image_id if use_image_ids else caption_id, int(TextUtils.is_transitive_sentence(nlp_data, self.language))))

        return transitivity_dataset

    """ Negation dataset: maps image ids to list of boolean stating whether each caption uses negation. """

    def generate_negation_dataset(self, use_image_ids=True):
        if self.language == 'French':
            negation_dataset = self.generate_french_negation_dataset(use_image_ids)
        else:
            negation_dataset = self.generate_non_french_negation_dataset(use_image_ids)

        return negation_dataset

    def generate_french_negation_dataset(self, use_image_ids=True):
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
                    caption_id = caption_data[sample_ind]['caption_id']
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
                    tokenized_caption = TextUtils.tokenize(caption, self.language)
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
                    negation_dataset.append((image_id if use_image_ids else caption_id, int(negation)))

                    new_sentence = True
                else:
                    split_line = line.split('\t')
                    cur_token_str = split_line[1]
                    cur_token_feat = split_line[7]
                    if cur_token_str in french_pos_neg_words and 's=neg' in cur_token_feat:
                        contains_neg_pos_word_in_negative_form = True
            assert sample_ind == len(caption_data) - 1

        return negation_dataset

    def generate_non_french_negation_dataset(self, use_image_ids=True):
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
        chinese_non_negation_words = defaultdict(list, {
            '别': ['着'],
            '不': ['小心', '一样']
        })
        japanese_negation_words = set([
            'ない',  # When it's at the end of the sentence it's negation, otherwise not sure
            'ませ',  # Mase, but I need masen ません
            'なし',  # Basically ok, but sometimes wrong for example 砂漠に似た乾燥地帯に馬が群れをなしている
            'なかっ',  # Ok
            'いいえ'
        ])

        caption_data = self.get_caption_data()
        negation_dataset = []
        if self.language == 'German':
            self.generate_nlp_data()

        for i in range(len(caption_data)):
            sample = caption_data[i]
            image_id = sample['image_id']
            caption_id = sample['caption_id']
            caption = sample['caption']
            if self.language == 'English':
                tokenized_caption = TextUtils.tokenize_and_clean(caption, self.language)
                negation_words_in_caption = english_negation_words.intersection(tokenized_caption)
            elif self.language == 'German':
                sample_nlp_data = self.nlp_data[i]
                negation_words_in_caption = german_negation_words.intersection([
                    x['lemma'].lower() for x in sample_nlp_data
                ])
            elif self.language == 'Chinese':
                ''' Jieba is a better tokenizer for Chinese than stanza. '''
                tokenized_caption = list(jieba.cut(caption, cut_all=False))
                negation_words_in_caption = []
                caption_suffix = caption
                for token in tokenized_caption:
                    caption_suffix = caption_suffix[len(token):]
                    if token in chinese_negation_words:
                        relevant_non_negation_words = chinese_non_negation_words[token]
                        if len([x for x in relevant_non_negation_words if caption_suffix.startswith(x)]) == 0:
                            negation_words_in_caption.append(token)
            elif self.language == 'Japanese':
                tokenized_caption = TextUtils.tokenize(caption, self.language)
                negation_words_in_caption = japanese_negation_words.intersection(tokenized_caption)
            negation_dataset.append((image_id if use_image_ids else caption_id, int(len(negation_words_in_caption) > 0)))

        return negation_dataset

    """ Numbers dataset: maps image ids to list of boolean stating whether each caption contains numbers. """

    def generate_numbers_dataset(self, binarized, use_image_ids=True):
        if binarized:
            return self.generate_numbers_dataset_internal(True, use_image_ids)
        else:
            return generate_dataset(
                self.non_binarized_numbers_dataset_file_path, self.generate_numbers_dataset_internal, False, use_image_ids
            )

    def generate_numbers_dataset_internal(self, binarized, use_image_ids=True):
        caption_data = self.get_caption_data()
        numbers_dataset = []
        if self.language == 'English':
            culture_language = Culture.English
        elif self.language == 'French':
            self.generate_nlp_data()
            culture_language = Culture.French
        elif self.language == 'Japanese':
            culture_language = Culture.Japanese
        elif self.language == 'German':
            self.generate_nlp_data()
            culture_language = Culture.German
        elif self.language == 'Chinese':
            culture_language = Culture.Chinese
        else:
            self.log_print(f'Numbers property not implemented for language {self.language}')
            assert False

        for i in range(len(caption_data)):
            sample = caption_data[i]
            image_id = sample['image_id']
            caption_id = sample['caption_id']
            caption = sample['caption']

            # The recognizers_number package has some bug in logographic languages: it doesn't work if there are no
            # spaces between words
            if self.language in ['Japanese', 'Chinese']:
                caption = ' '.join([char for char in caption])

            numbers_list = recognize_number(caption, culture_language)
            ''' Ignore the numeral one. '''
            numbers_list = [x for x in numbers_list if x.resolution['value'] != '1']

            if binarized:
                numbers_dataset.append((image_id if use_image_ids else caption_id, int(len(numbers_list) > 0)))
            else:
                numbers_dataset.append((image_id if use_image_ids else caption_id, [float(x.resolution['value']) for x in numbers_list
                                                   if ',' not in x.resolution['value']]))

        return numbers_dataset

    """ Root pos dataset: maps image ids to list of boolean stating whether each caption's root is a noun (0) or a verb
        (1).
    """

    def generate_root_pos_dataset(self, use_image_ids=True):
        if self.language == 'French':
            root_pos_dataset = self.generate_french_root_pos_dataset(use_image_ids)
        else:
            root_pos_dataset = self.generate_non_french_root_pos_dataset(use_image_ids)

        return root_pos_dataset

    def generate_french_root_pos_dataset(self, use_image_ids):
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
                    caption_id = caption_data[sample_ind]['caption_id']
                    roots_pos = []
                    new_sentence = False
                if line == '\n':
                    new_sentence = True

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

                    root_pos_dataset.append((image_id if use_image_ids else caption_id, val))
                else:
                    split_line = line.split('\t')
                    dep_tag = split_line[11]
                    if dep_tag == 'root':
                        pos_tag = split_line[5]
                        roots_pos.append(pos_tag)
            assert sample_ind == len(caption_data) - 1

        return root_pos_dataset

    def generate_non_french_root_pos_dataset(self, use_image_ids=True):
        caption_data = self.get_caption_data()
        self.generate_nlp_data()
        root_pos_dataset = []

        for i in range(len(caption_data)):
            image_id = caption_data[i]['image_id']
            caption_id = caption_data[i]['caption_id']
            nlp_data = self.nlp_data[i]
            roots = [token for token in nlp_data if token['dep'].lower() == 'root']
            if len(roots) != 1:
                # We don't know how to deal with zero or multiple roots, for now
                continue

            root = roots[0]
            if root['pos'] in ['NOUN', 'PRON', 'PROPN']:
                val = 0
            elif root['pos'] == 'VERB':
                val = 1
                if TextUtils.is_existential_sentence(nlp_data, self.language):
                    # The 'be' verb is not really an action, so we consider it as a noun root
                    val = 0
            else:
                # We're only interested in verb or noun roots
                continue

            root_pos_dataset.append((image_id if use_image_ids else caption_id, val))

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
        for sample in caption_data:
            caption = sample['caption']
            image_id = sample['image_id']
            if self.language == 'English':
                tokenized_caption = TextUtils.tokenize_and_clean(caption, self.language)

                # Spatial words
                spatial_words_in_caption = english_spatial_words.intersection(tokenized_caption)

                # Spatial phrases
                spatial_phrases_in_caption = [phrase for phrase in english_spatial_phrases
                                              if TextUtils.phrase_in_sent(tokenized_caption, phrase)]

                spatial_dataset.append((image_id,
                                        int(len(spatial_words_in_caption) > 0 or len(spatial_phrases_in_caption) > 0)))

        return spatial_dataset
    
    def generate_length_dataset(self, use_image_ids=True, length_type='chars'):
        length_file_path = f'{self.name}_{self.language}_lengths.tsv'
        length_dataset = []
        with open(length_file_path, 'r') as fp:
            my_reader = csv.reader(fp, delimiter='\t')
            for row in my_reader:
                image_id = int(row[0])
                    
                if length_type == 'chars':
                    length = int(row[3])
                elif length_type == 'words':
                    length = int(row[4])
                elif length_type == 'content_words':
                    length = int(row[5])
                else:
                    assert False, f'Unknown length type {length_type}'
                length_dataset.append((image_id if use_image_ids else int(row[1]), length))

        return length_dataset
    
    def generate_bin_length_dataset(self, use_image_ids=True, length_type='chars'):
        length_dataset = self.generate_length_dataset(use_image_ids, length_type)
        all_vals = [x[1] for x in length_dataset]
        median = statistics.median(all_vals)
        larger_than_median = len([x for x in all_vals if x > median])
        larger_equal_median = len([x for x in all_vals if x >= median])
        lt_dist = abs(0.5 - larger_than_median/len(all_vals))
        le_dist = abs(0.5 - larger_equal_median/len(all_vals))
        if lt_dist < le_dist:
            bin_length_dataset = [(x[0], int(x[1] > median)) for x in length_dataset]
        else:
            bin_length_dataset = [(x[0], int(x[1] >= median)) for x in length_dataset]

        return bin_length_dataset

    def get_struct_data_internal(self, use_image_ids=True):
        if self.struct_property == 'passive':
            struct_data = self.generate_passive_dataset(use_image_ids)
        elif self.struct_property == 'transitivity':
            struct_data = self.generate_transitivity_dataset(use_image_ids)
        elif self.struct_property == 'negation':
            struct_data = self.generate_negation_dataset(use_image_ids)
        elif self.struct_property == 'numbers':
            struct_data = self.generate_numbers_dataset(True, use_image_ids)
        elif self.struct_property == 'root_pos':
            struct_data = self.generate_root_pos_dataset(use_image_ids)
        elif self.struct_property == 'spatial_relations':
            struct_data = self.generate_spatial_relations_dataset(use_image_ids)
        elif self.struct_property.startswith('length_'):
            length_type = self.struct_property.split('length_')[1]
            struct_data = self.generate_length_dataset(use_image_ids, length_type=length_type)
        elif self.struct_property.startswith('bin_length_'):
            length_type = self.struct_property.split('bin_length_')[1]
            struct_data = self.generate_bin_length_dataset(use_image_ids, length_type=length_type)

        return struct_data

    @abc.abstractmethod
    def create_image_path_finder(self):
        return

    # Some of the datasets are not naturally divided to splits. Create functionality to do that

    def get_all_image_ids(self):
        caption_data = self.get_caption_data()
        return list(set([x['image_id'] for x in caption_data]))
