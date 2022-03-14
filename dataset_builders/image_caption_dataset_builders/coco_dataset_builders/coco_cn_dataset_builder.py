import os
from dataset_builders.image_caption_dataset_builders.coco_dataset_builders.coco_based_dataset_builder import \
    CocoBasedDatasetBuilder


class CocoCNDatasetBuilder(CocoBasedDatasetBuilder):
    """ This is the dataset builder class for the COCO-CN dataset, described in the paper 'COCO-CN for Cross-Lingual
        Image Tagging, Captioning, and Retrieval' by Li et al.
        This dataset is based on the COCO dataset.
    """

    def __init__(self, root_dir_path, data_split_str, struct_property, indent):
        super(CocoCNDatasetBuilder, self).__init__(root_dir_path, 'coco_cn', data_split_str, struct_property,
                                                   indent)

        captions_file_name = 'imageid.human-written-caption.txt'

        self.captions_file_path = os.path.join(root_dir_path, captions_file_name)

    def get_caption_data(self):
        caption_data = []
        external_caption_file_path = self.captions_file_path
        with open(external_caption_file_path, 'r', encoding='utf8') as caption_fp:
            for line in caption_fp:
                line = line.strip()
                line_parts = line.split()
                image_file_name = line_parts[0]

                # Check if current image is from the relevant data split
                image_file_name_parts = image_file_name.split('_')
                data_split_str = image_file_name_parts[1].split('2014')[0]
                assert data_split_str in ['train', 'val']
                if data_split_str != self.data_split_str:
                    continue

                caption = line_parts[-1]
                image_id = int(image_file_name_parts[-1].split('#')[0])

                caption_data.append({'caption': caption, 'image_id': image_id})
        return caption_data
