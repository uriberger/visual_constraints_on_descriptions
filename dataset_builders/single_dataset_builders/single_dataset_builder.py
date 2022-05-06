import abc
import os
import random
from utils.general_utils import generate_dataset, get_image_id_to_prob
from dataset_builders.dataset_builder import DatasetBuilder


class SingleDatasetBuilder(DatasetBuilder):
    """ This class is the base class for all single datasets builders. """

    def __init__(self, name, language, struct_property, indent):
        super(SingleDatasetBuilder, self).__init__(name, struct_property, indent)

        self.language = language
        self.extended_name = f'{name}_{language}'

        self.balanced_labeled_data_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.extended_name}_{self.struct_property}_balanced_labeled_data'
        )

        self.train_val_split_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.extended_name}_{self.struct_property}_train_val_split'
        )

    """ Train/Val splits:
        Since we're creating a new annotation and we want it to contain as much data as possible for all classes, we
        don't care about the original train/val splits.
        So, what we're going to do is unite all the data together, annotate it, and split so we'd have an equal number
        of instances of each class in the train and val sets.
    """

    """ The struct data list is a list of (image_id, val) where image ids are not unique and val is binary value
        representing whether a specific caption (related to the image id) expresses the relevant property.
        We want to:
        1. Convert this list to a mapping of image id -> probability of the relevant property, which is calculated as
        the proportion of captions expressing this property.
        2. Set a threshold for the probability so the final dataset will be binary.
        3. Create a new list of unique (image id, binary value) pairs, where the binary value indicates if this image
        id exceeds the threshold. This is referred to as the labeled data list.

        To choose the threshold we search for the value that, if chosen, will make the data be the closest to half 1
        half 0.
    """

    def get_labeled_data(self):
        # 1. Convert the struct_data list to a mapping of image id -> probability of the relevant property, which is
        # calculated as the proportion of captions expressing this property.
        struct_data = self.get_struct_data()
        image_id_to_prob = get_image_id_to_prob(struct_data)

        # 2. Set a threshold for the probability so the final dataset will be binary.
        all_values = list(set(image_id_to_prob.values()))
        assert len(all_values) > 1
        val_to_count = {x: 0 for x in all_values}
        for val in image_id_to_prob.values():
            val_to_count[val] += 1
        val_to_count_list = sorted(val_to_count.items(), key=lambda x: x[0])
        current_count = 0
        cur_ind = -1
        total_count = len(image_id_to_prob)
        while current_count < total_count / 2:
            cur_ind += 1
            current_count += val_to_count_list[cur_ind][1]

        # Found the ind of the median, now check if this one is closer to half or the previous one
        current_ratio = current_count / total_count
        prev_ratio = (current_count - val_to_count_list[cur_ind][1]) / total_count
        if abs(0.5 - current_ratio) < abs(0.5 - prev_ratio):
            threshold = val_to_count_list[cur_ind][0]
        else:
            threshold = val_to_count_list[cur_ind - 1][0]

        labeled_data = [(x[0], int(x[1] > threshold)) for x in image_id_to_prob.items()]

        return labeled_data

    """ Next, balance the data so that we have the same number of instances for each label. """

    @staticmethod
    def find_samples_for_labels(labeled_data):
        all_labels = list(set([x[1] for x in labeled_data]))
        label_to_data_samples = {x: [] for x in all_labels}
        for image_id, label in labeled_data:
            label_to_data_samples[label].append(image_id)

        return label_to_data_samples

    def get_balanced_labeled_data(self):
        return generate_dataset(self.balanced_labeled_data_file_path, self.get_balanced_labeled_data_internal)

    def get_balanced_labeled_data_internal(self):
        # Find the class with the lowest number of instances
        labeled_data = self.get_labeled_data()
        unwanted_image_ids = self.get_unwanted_image_ids()
        unwanted_image_ids_dict = {image_id: True for image_id in unwanted_image_ids}  # For more efficient retrieval
        labeled_data = [x for x in labeled_data if x[0] not in unwanted_image_ids_dict]

        label_to_data_samples = self.find_samples_for_labels(labeled_data)
        wanted_sample_num_for_each_label = min([len(x) for x in label_to_data_samples.values()])
        balanced_data = []
        for label, image_id_list in label_to_data_samples.items():
            sampled_image_ids = random.sample(image_id_list, wanted_sample_num_for_each_label)
            balanced_data += [(x, label) for x in sampled_image_ids]

        return balanced_data

    """ Next, automatically split the labeled and balanced data to training/val. """

    def create_train_val_split(self):
        labeled_data = self.get_balanced_labeled_data()
        image_ids = [x[0] for x in labeled_data]
        train_split_size = int(len(image_ids) * 0.8)
        train_split = random.sample(image_ids, train_split_size)
        train_split_dict = {x: True for x in train_split}
        val_split = [x for x in image_ids if x not in train_split_dict]

        return {'train': train_split, 'val': val_split}

    """ Finally, we can now get the data for a specific split. """

    def get_labeled_data_for_split(self, data_split_str):
        balanced_labeled_data = self.get_balanced_labeled_data()
        split_to_image_ids = generate_dataset(self.train_val_split_file_path, self.create_train_val_split)
        split_image_ids = split_to_image_ids[data_split_str]
        split_image_ids_dict = {x: True for x in split_image_ids}
        return [x for x in balanced_labeled_data if x[0] in split_image_ids_dict]

    @abc.abstractmethod
    def get_unwanted_image_ids(self):
        return
