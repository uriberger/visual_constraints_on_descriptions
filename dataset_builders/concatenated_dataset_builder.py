import math
from dataset_builders.dataset_builder import DatasetBuilder
from dataset_builders.image_path_finder import ImagePathFinder


class ConcatImagePathFinder(ImagePathFinder):

    def __init__(self, image_path_finders_list, mult_fact):
        super(ConcatImagePathFinder, self).__init__()

        self.image_path_finders_list = image_path_finders_list
        self.mult_fact = mult_fact

    def new_to_orig_image_id(self, image_id):
        orig_image_id = image_id % self.mult_fact
        dataset_ind = image_id // self.mult_fact
        return orig_image_id, dataset_ind

    def get_image_path(self, image_id):
        orig_image_id, dataset_ind = self.new_to_orig_image_id(image_id)
        return self.image_path_finders_list[dataset_ind].get_image_path(orig_image_id)


class ConcatenatedDatasetBuilder(DatasetBuilder):
    """ This class builds a large dataset containing multiple smaller datasets. """

    def __init__(self, builder_list, struct_property, indent):
        name = 'concat'
        for builder in builder_list:
            name += '_' + builder.name
        super(ConcatenatedDatasetBuilder, self).__init__(name, struct_property, indent)

        self.builder_list = builder_list
        self.mult_fact = self.find_mult_fact()

    def find_mult_fact(self):
        max_image_id = 0
        for builder in self.builder_list:
            cur_builder_struct_data = builder.get_struct_data()
            image_ids = list(set([x[0] for x in cur_builder_struct_data]))
            cur_max = max(image_ids)
            if cur_max > max_image_id:
                max_image_id = cur_max
        return 10 ** (int(math.log10(max_image_id)) + 1)

    def orig_to_new_image_id(self, orig_image_id, dataset_ind):
        return dataset_ind*self.mult_fact + orig_image_id

    def get_struct_data(self):
        struct_data = []
        for i in range(len(self.builder_list)):
            builder = self.builder_list[i]
            cur_struct_data = builder.get_struct_data()
            struct_data += [(self.orig_to_new_image_id(x[0], i), x[1]) for x in cur_struct_data]

        return struct_data

    def create_image_path_finder(self):
        image_path_finders = [builder.create_image_path_finder() for builder in self.builder_list]
        return ConcatImagePathFinder(image_path_finders, self.mult_fact)

    def get_class_mapping(self):
        self.mapping_list = [x.get_class_mapping() for x in self.builder_list]
        class_mapping = []
        self.dataset_to_old_class_ind_to_new_class_ind = []
        class_name_to_new_ind = {}
        for i in range(len(self.mapping_list)):
            self.dataset_to_old_class_ind_to_new_class_ind.append({})
            cur_mapping = self.mapping_list[i]
            for j in range(len(cur_mapping)):
                class_name = cur_mapping[j]
                if class_name in class_name_to_new_ind:
                    cur_new_class_ind = class_name_to_new_ind[class_name]
                else:
                    cur_new_class_ind = len(class_mapping)
                    class_name_to_new_ind[class_name] = cur_new_class_ind
                    class_mapping.append(class_name)
                self.dataset_to_old_class_ind_to_new_class_ind[-1][j] = cur_new_class_ind
        return class_mapping

    def get_gt_classes_data(self):
        self.get_class_mapping()
        gt_classes_data = {}
        gt_classes_data_list = [x.get_gt_classes_data() for x in self.builder_list]
        for i in range(len(gt_classes_data_list)):
            cur_gt_classes_data = gt_classes_data_list[i]
            # Not sure how to handle cases where the same image is in different datasets
            intersection_with_existing = set(cur_gt_classes_data.keys()).intersection(gt_classes_data.keys())
            assert len(intersection_with_existing) == 0

            for orig_image_id, orig_class_list in cur_gt_classes_data.items():
                gt_classes_data[self.orig_to_new_image_id(orig_image_id, i)] = \
                    [self.dataset_to_old_class_ind_to_new_class_ind[i][x] for x in orig_class_list]

            return gt_classes_data

    def get_gt_bboxes_data(self):
        gt_bboxes_data = {}
        gt_bboxes_data_list = [x.get_gt_bboxes_data() for x in self.builder_list]
        for i in range(len(gt_bboxes_data_list)):
            cur_gt_bboxes_data = gt_bboxes_data_list[i]
            # Not sure how to handle cases where the same image is in different datasets
            intersection_with_existing = set(cur_gt_bboxes_data.keys()).intersection(gt_bboxes_data.keys())
            assert len(intersection_with_existing) == 0

            for orig_image_id, bbox_data in cur_gt_bboxes_data.items():
                gt_bboxes_data[self.orig_to_new_image_id(orig_image_id, i)] = bbox_data

            return gt_bboxes_data

    def get_labeled_data(self, bin_num=10):
        labeled_data = []
        for i in range(len(self.builder_list)):
            builder = self.builder_list[i]
            cur_labeled_data = builder.get_labeled_data()
            labeled_data += [(self.orig_to_new_image_id(x[0], i), x[1]) for x in cur_labeled_data]

        return labeled_data

    def get_labeled_data_for_split(self, data_split_str, bin_num=10):
        labeled_data = []
        for i in range(len(self.builder_list)):
            builder = self.builder_list[i]
            cur_labeled_data = builder.get_labeled_data_for_split(data_split_str)
            labeled_data += [(self.orig_to_new_image_id(x[0], i), x[1]) for x in cur_labeled_data]

        return labeled_data

    def generate_cross_validation_data(self, split_num):
        self.data_splits = []
        for _ in range(split_num):
            self.data_splits.append([])
        for i in range(len(self.builder_list)):
            builder = self.builder_list[i]
            builder.generate_cross_validation_data(split_num)
            for j in range(split_num):
                self.data_splits[j] += [(self.orig_to_new_image_id(x[0], i), x[1]) for x in builder.data_splits[j]]
