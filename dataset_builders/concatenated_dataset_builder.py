import math
from dataset_builders.dataset_builder import DatasetBuilder
from dataset_builders.image_path_finder import ImagePathFinder


class ConcatImagePathFinder(ImagePathFinder):

    def __init__(self, image_path_finders_list, mult_fact):
        super(ConcatImagePathFinder, self).__init__()

        self.image_path_finders_list = image_path_finders_list
        self.mult_fact = mult_fact

    def get_image_path(self, image_id):
        orig_image_id = image_id % self.mult_fact
        dataset_ind = image_id // self.mult_fact
        return self.image_path_finders_list[dataset_ind].get_image_path(orig_image_id)


class ConcatenatedDatasetBuilder(DatasetBuilder):
    """ This class builds a large dataset containing multiple smaller datasets. """

    def __init__(self, builder_list, data_split_str, struct_property, indent):
        name = 'concat'
        for builder in builder_list:
            name += '_' + builder.name
        super(ConcatenatedDatasetBuilder, self).__init__(name, data_split_str, struct_property, indent)

        self.builder_list = builder_list
        self.mult_fact = self.find_mult_fact()

    def find_mult_fact(self):
        max_image_id = 0
        for builder in self.builder_list:
            image_ids = builder.get_all_image_ids()
            cur_max = max(image_ids)
            if cur_max > max_image_id:
                max_image_id = cur_max
        return 10 ** (int(math.log10(max_image_id)) + 1)

    def orig_to_new_image_id(self, orig_image_id, dataset_ind):
        return dataset_ind*self.mult_fact + orig_image_id

    def create_struct_data(self):
        struct_data = []
        for i in range(len(self.builder_list)):
            builder = self.builder_list[i]
            cur_struct_data = builder.create_struct_data()
            struct_data += [(self.orig_to_new_image_id(x[0], i), x[1]) for x in cur_struct_data]

        return struct_data

    def create_image_path_finder(self):
        image_path_finders = [builder.create_image_path_finder() for builder in self.builder_list]
        return ConcatImagePathFinder(image_path_finders, self.mult_fact)

    def get_all_image_ids(self):
        all_image_ids = []
        for i in range(len(self.builder_list)):
            builder = self.builder_list[i]
            cur_image_ids = builder.get_all_image_ids()
            all_image_ids += [self.orig_to_new_image_id(x, i) for x in cur_image_ids]

        return all_image_ids

    def get_unwanted_image_ids(self):
        unwanted_image_ids = []
        for i in range(len(self.builder_list)):
            builder = self.builder_list[i]
            cur_image_ids = builder.get_unwanted_image_ids()
            unwanted_image_ids += [self.orig_to_new_image_id(x, i) for x in cur_image_ids]

        return unwanted_image_ids
