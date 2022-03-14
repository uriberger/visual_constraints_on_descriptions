import os
from xml.dom import minidom
from dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_path_finder import ImagePathFinder
from utils.general_utils import generate_dataset


class Flickr30kImagePathFinder(ImagePathFinder):

    def __init__(self, images_dir_path):
        super(Flickr30kImagePathFinder, self).__init__()

        self.images_dir_path = images_dir_path

    def get_image_path(self, image_id):
        image_file_name = f'{image_id}.jpg'
        image_path = os.path.join(self.images_dir_path, image_file_name)

        return image_path


class Flickr30kDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the dataset builder class for the flickr30k dataset, described in the paper 'From image descriptions to
        visual denotations: New similarity metrics for semantic inference over event descriptions' by Young et al.
    """

    def __init__(self, root_dir_path, struct_property, indent):
        super(Flickr30kDatasetBuilder, self).__init__(root_dir_path, 'flickr30k', 'all', struct_property,
                                                      indent)

        tokens_dir_name = 'tokens'
        tokens_file_name = 'results_20130124.token'
        self.tokens_file_path = os.path.join(self.root_dir_path, tokens_dir_name, tokens_file_name)

        images_dir_name = 'images'
        self.images_dir_path = os.path.join(self.root_dir_path, images_dir_name)

        bbox_dir_name = os.path.join('annotations', 'Annotations')
        self.bbox_dir_path = os.path.join(self.root_dir_path, bbox_dir_name)

        sentences_dir_name = os.path.join('annotations', 'Sentences')
        self.sentences_dir_path = os.path.join(self.root_dir_path, sentences_dir_name)

        self.chains_filename = os.path.join(self.cached_dataset_files_dir, 'flickr30_chains')
        self.chains_and_classes_file_name = os.path.join(self.cached_dataset_files_dir, 'flickr30_chains_and_classes')

    def get_caption_data(self):
        fp = open(self.tokens_file_path, encoding='utf-8')
        image_id_captions_pairs = []
        for line in fp:
            split_line = line.strip().split('#')
            img_file_name = split_line[0]
            image_id = self.image_file_name_to_id(img_file_name)
            caption = split_line[1].split('\t')[1]  # The first token is caption number

            image_id_captions_pairs.append({'image_id': image_id, 'caption': caption})

        return image_id_captions_pairs

    def create_image_path_finder(self):
        return Flickr30kImagePathFinder(self.images_dir_path)

    def extract_chains(self):
        return generate_dataset(self.chains_filename, self.extract_chains_internal)

    def extract_chains_internal(self):
        extracted_chains = {}
        boxes_chains = {}
        self.log_print('Extracting coreference chains...')
        for _, _, files in os.walk(self.bbox_dir_path):
            for filename in files:
                # Extract image file name from current file name
                image_id = int(filename.split('.')[0])

                # Extract bounding boxes from file
                bounding_boxes = []

                xml_filepath = os.path.join(self.bbox_dir_path, filename)
                xml_doc = minidom.parse(xml_filepath)
                for child_node in xml_doc.childNodes[0].childNodes:
                    # The bounding boxes are located inside a node named "object"
                    if child_node.nodeName == u'object':
                        # Go over all of the children of this node: if we find bndbox, this object is a bounding box
                        box_chain = None
                        for inner_child_node in child_node.childNodes:
                            if inner_child_node.nodeName == u'name':
                                box_chain = int(inner_child_node.childNodes[0].data)
                            if inner_child_node.nodeName == u'bndbox':
                                # This is a bounding box node
                                bounding_boxes.append(box_chain)

                                # Document chain
                                if box_chain not in extracted_chains:
                                    extracted_chains[box_chain] = True
                boxes_chains[image_id] = bounding_boxes

        self.log_print('Extracted coreference chains')
        chain_list = list(extracted_chains.keys())

        return boxes_chains, chain_list

    def get_chain_to_class_mapping(self, chain_list):
        return generate_dataset(self.chains_and_classes_file_name, self.get_chain_to_class_mapping_internal, chain_list)

    def get_chain_to_class_mapping_internal(self, chain_list):
        chain_to_class_ind = {}
        class_str_to_ind = {}
        found_chains = {x: False for x in chain_list}
        self.log_print('Extracting chain to class mapping...')
        file_names = os.listdir(self.sentences_dir_path)
        for file_ind in range(len(file_names)):
            # Extract annotated sentences from file
            filename = file_names[file_ind]
            filepath = os.path.join(self.sentences_dir_path, filename)
            fp = open(filepath, 'r', encoding='utf-8')
            for line in fp:
                split_by_annotations = line.split('[/EN#')[1:]
                for line_part in split_by_annotations:
                    annotation = line_part.split()[0].split('/')
                    chain_ind = int(annotation[0])
                    class_str = annotation[1]

                    if class_str in class_str_to_ind:
                        class_ind = class_str_to_ind[class_str]
                    else:
                        class_ind = len(class_str_to_ind)
                        class_str_to_ind[class_str] = class_ind

                    chain_to_class_ind[chain_ind] = class_ind
                    if chain_ind in found_chains and not found_chains[chain_ind]:
                        found_chains[chain_ind] = True

        # Add the 'unknown' class for chains we couldn't find
        unknown_class_ind = len(class_str_to_ind)
        class_str_to_ind['unknown'] = unknown_class_ind
        chains_not_found = [x for x in chain_list if not found_chains[x]]
        for chain_ind in chains_not_found:
            chain_to_class_ind[chain_ind] = unknown_class_ind

        self.log_print('Extracted chain to class mapping')
        class_ind_to_str = {class_str_to_ind[k]: k for k in class_str_to_ind.keys()}

        return chain_to_class_ind, class_ind_to_str

    def get_gt_classes_data_internal(self):
        boxes_chains, chain_list = self.extract_chains()
        chain_to_class_ind, class_ind_to_str = self.get_chain_to_class_mapping(chain_list)
        img_classes_dataset = {y: [chain_to_class_ind[x] for x in boxes_chains[y]]
                               for y in boxes_chains.keys()}

        return img_classes_dataset

    def get_class_mapping(self):
        _, chain_list = self.extract_chains()
        _, class_ind_to_str = self.get_chain_to_class_mapping(chain_list)
        return class_ind_to_str

    @staticmethod
    def image_file_name_to_id(image_file_name):
        return int(image_file_name.split('.')[0])
