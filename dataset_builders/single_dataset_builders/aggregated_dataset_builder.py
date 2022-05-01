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

    def get_unwanted_image_ids(self):
        unwanted_image_ids = []
        for builder in self.builder_list:
            unwanted_image_ids += builder.get_unwanted_image_ids()
        return list(set(unwanted_image_ids))
