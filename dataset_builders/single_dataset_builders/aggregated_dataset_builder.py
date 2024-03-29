from dataset_builders.single_dataset_builders.single_dataset_builder import SingleDatasetBuilder


class AggregatedDatasetBuilder(SingleDatasetBuilder):
    """ This class builds a large dataset containing multiple versions of the same dataset. """

    def __init__(self, original_dataset_name, builder_list, struct_property, indent):
        super(AggregatedDatasetBuilder, self).__init__(original_dataset_name, 'agg', struct_property, indent)

        self.builder_list = builder_list

    def get_struct_data(self):
        struct_data = []
        for i in range(len(self.builder_list)):
            builder = self.builder_list[i]
            cur_struct_data = builder.get_struct_data()
            struct_data += cur_struct_data

        return struct_data

    def create_image_path_finder(self):
        return self.builder_list[0].create_image_path_finder()

    def get_gt_classes_data(self):
        return self.builder_list[0].get_gt_classes_data()

    def get_gt_bboxes_data(self):
        return self.builder_list[0].get_gt_bboxes_data()

    def get_class_mapping(self):
        return self.builder_list[0].get_class_mapping()

    def get_unwanted_image_ids(self):
        unwanted_image_ids = []
        for builder in self.builder_list:
            unwanted_image_ids += builder.get_unwanted_image_ids()
        return list(set(unwanted_image_ids))

    # Generates a dataset of image -> list of numerals identified in each captions of this image
    def generate_numbers_dataset(self):
        numbers_dataset = []
        for i in range(len(self.builder_list)):
            builder = self.builder_list[i]
            cur_numbers_dataset = builder.generate_numbers_dataset(False)
            numbers_dataset += cur_numbers_dataset

        return numbers_dataset
