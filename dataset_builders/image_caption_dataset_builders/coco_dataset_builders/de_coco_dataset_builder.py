import os
from utils.general_utils import generate_dataset
from dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.image_caption_dataset_builders.coco_dataset_builders.coco_dataset_builder import \
    CocoDatasetBuilder


class DeCocoDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the DeCOCO dataset, described in the paper 'Multimodal Pivots for Image
        Caption Translation' by Hitschler et al.
        This dataset is based on the COCO dataset.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(DeCocoDatasetBuilder, self).__init__(root_dir_path, 'de_coco', data_split_str, struct_property,
                                                   CocoDatasetBuilder, 'COCO', indent)

        self.caption_file_name_prefixes = ['dev', 'devtest', 'test']

        self.line_ind_to_image_id_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_line_ind_to_image_id'
        )

        # We need to override a parent class behavior: no matter what data split is used in this dataset, we want the
        # base dataset builder (COCO builder) to use the train split, since all the images in this dataset are from the
        # COCO train split
        self.base_dataset_builder = CocoDatasetBuilder(os.path.join(root_dir_path, '..', 'COCO'),
                                                       'train', self.struct_property, self.indent + 1)

    def get_line_ind_to_image_id_mappings_internal(self):
        """ For some reason, even though the README file of this dataset states that there supposed to be files
        indicating to which image id each line refers, there is none. So I need to do it manually by searching for the
        English caption in the original COCO dataset and using the fact that the English and German files are aligned.
        """
        known_mappings = {'dev': {}, 'devtest': {130: 471312}, 'test': {15: 453695, 386: 553184, 498: 488853}}

        english_coco_caption_data = self.base_dataset_builder.get_caption_data()
        line_ind_to_image_id_mappings = {}
        for caption_file_name_prefix in self.caption_file_name_prefixes:
            line_ind_to_image_id_mappings[caption_file_name_prefix] = []
            caption_file_name = caption_file_name_prefix + '.en'
            caption_file_path = os.path.join(self.root_dir_path, caption_file_name)
            line_ind = 0
            with open(caption_file_path, 'r', encoding='utf8') as caption_fp:
                for line in caption_fp:
                    if line_ind in known_mappings[caption_file_name_prefix]:
                        image_id = known_mappings[caption_file_name_prefix][line_ind]
                    else:
                        caption = line.strip()
                        english_coco_caption_samples = [x for x in english_coco_caption_data
                                                        if x['caption'] == caption]
                        if len(english_coco_caption_samples) == 0:
                            english_coco_caption_samples = [x for x in english_coco_caption_data
                                                            if x['caption'].strip() == caption]
                        if len(english_coco_caption_samples) == 0:
                            english_coco_caption_samples = [x for x in english_coco_caption_data
                                                            if caption in x['caption']
                                                            and len(caption) >= len(x['caption']) - 1]
                        assert len(english_coco_caption_samples) > 0
                        image_id = english_coco_caption_samples[0]['image_id']
                    line_ind_to_image_id_mappings[caption_file_name_prefix].append(image_id)
                    line_ind += 1

        return line_ind_to_image_id_mappings

    def get_line_ind_to_image_id_mappings(self):
        return generate_dataset(self.line_ind_to_image_id_file_path, self.get_line_ind_to_image_id_mappings_internal)

    def get_all_image_ids(self):
        line_ind_to_image_id_mappings = self.get_line_ind_to_image_id_mappings()
        image_ids_lists = list(line_ind_to_image_id_mappings.values())
        image_id_list = [x for outer in image_ids_lists for x in outer]
        return list(set(image_id_list))

    def get_caption_data(self):
        line_ind_to_image_id_mappings = self.get_line_ind_to_image_id_mappings()
        caption_data = []
        for caption_file_name_prefix in self.caption_file_name_prefixes:
            caption_file_name = caption_file_name_prefix + '.de'
            caption_file_path = os.path.join(self.root_dir_path, caption_file_name)
            line_ind_to_image_id_mapping = line_ind_to_image_id_mappings[caption_file_name_prefix]
            line_ind = 0
            with open(caption_file_path, 'r', encoding='utf8') as caption_fp:
                for line in caption_fp:
                    caption = line.strip()
                    image_id = line_ind_to_image_id_mapping[line_ind]
                    caption_data.append({'image_id': image_id, 'caption': caption})
                    line_ind += 1

        data_split_image_ids = self.get_image_ids_for_split()
        data_split_image_ids_dict = {x: True for x in data_split_image_ids}
        return [x for x in caption_data if x['image_id'] in data_split_image_ids_dict]