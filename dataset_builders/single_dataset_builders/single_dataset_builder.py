import abc
import os
import random
from utils.general_utils import generate_dataset, get_image_id_to_prob
from dataset_builders.dataset_builder import DatasetBuilder
from collections import defaultdict


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

    """ In case the property is a length property, the struct data list is a list of (image_id, val) where image ids are
        not unique and val is an integer.
        We want to change it to a classification problem, in the following manner:
        1. Split all values to bins so that we'll have similar number of samples in each bin.
        2. Create a new list of unique (image id, bin index) pairs. This is referred to as the labeled data list.
    """

    """ In all other cases, the struct data list is a list of (image_id, val) where image ids are not unique and val is
        binary value representing whether a specific caption (related to the image id) expresses the relevant property.
        We want to:
        1. Convert this list to a mapping of image id -> probability of the relevant property, which is calculated as
        the proportion of captions expressing this property.
        2. Set a threshold for the probability so the final dataset will be binary.
        3. Create a new list of unique (image id, binary value) pairs, where the binary value indicates if this image
        id exceeds the threshold. This is referred to as the labeled data list.

        To choose the threshold we search for the value that, if chosen, will make the data be the closest to half 1
        half 0.
    """

    def get_labeled_data(self, bin_num=10):
        if self.struct_property.startswith('length_'):
            struct_data = self.get_struct_data()
            val_to_count = defaultdict(int)
            for sample in struct_data:
                val_to_count[sample[1]] += 1

            val_count_pairs = list(val_to_count.items())
            val_count_pairs.sort(key=lambda x:x[0])
            frac = 1/bin_num
            all_count = sum(val_to_count.values())
            bin_size = all_count/bin_num

            bin_start_list = [0]
            samples_so_far = 0
            for i in range(len(val_count_pairs)):
                val, count = val_count_pairs[i]
                samples_so_far += count
                cur_bin_ind = len(bin_start_list)
                if samples_so_far > cur_bin_ind*bin_size:
                    # Filled current bin, now check if this one is closer to the wanted bin size or the previous one
                    cur_fraction = samples_so_far/all_count
                    prev_fraction = (samples_so_far - count)/all_count
                    if abs(cur_bin_ind*frac - cur_fraction) < abs(cur_bin_ind*frac - prev_fraction):
                        bin_start_list.append(val_count_pairs[i+1][0])
                    else:
                        bin_start_list.append(val_count_pairs[i][0])

            val_to_bin_ind = {}
            sorted_val_list = sorted(list(val_to_count.keys()))
            cur_bin_ind = 0
            for val in sorted_val_list:
                if cur_bin_ind < (bin_num - 1) and val >= bin_start_list[cur_bin_ind+1]:
                    cur_bin_ind += 1
                val_to_bin_ind[val] = cur_bin_ind

            labeled_data = [(x[0], val_to_bin_ind[x[1]]) for x in struct_data]
        else:
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

        unwanted_image_ids = self.get_unwanted_image_ids()
        unwanted_image_ids_dict = {image_id: True for image_id in unwanted_image_ids}  # For more efficient retrieval
        labeled_data = [x for x in labeled_data if x[0] not in unwanted_image_ids_dict]

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

        label_to_data_samples = self.find_samples_for_labels(labeled_data)
        wanted_sample_num_for_each_label = min([len(x) for x in label_to_data_samples.values()])
        balanced_data = []
        for label, image_id_list in label_to_data_samples.items():
            sampled_image_ids = random.sample(image_id_list, wanted_sample_num_for_each_label)
            balanced_data += [(x, label) for x in sampled_image_ids]

        return balanced_data

    """ Next, automatically split the labeled and balanced data to training/val. """

    def create_train_val_split(self, bin_num=10):
        if self.struct_property.startswith('length_'):
            labeled_data = self.get_labeled_data(bin_num)
        else:
            labeled_data = self.get_balanced_labeled_data()
        image_ids = list(set([x[0] for x in labeled_data]))
        train_split_size = int(len(image_ids) * 0.8)
        train_split = random.sample(image_ids, train_split_size)
        train_split_dict = {x: True for x in train_split}
        val_split = [x for x in image_ids if x not in train_split_dict]

        return {'train': train_split, 'val': val_split}

    """ Finally, we can now get the data for a specific split. """

    def get_labeled_data_for_split(self, data_split_str, bin_num=10):
        if self.struct_property.startswith('length_'):
            res = self.get_labeled_data(bin_num)
        else:
            res = self.get_balanced_labeled_data()
        split_to_image_ids = generate_dataset(self.train_val_split_file_path, self.create_train_val_split, bin_num)
        split_image_ids = split_to_image_ids[data_split_str]
        split_image_ids_dict = {x: True for x in split_image_ids}
        return [x for x in res if x[0] in split_image_ids_dict]

    def generate_cross_validation_data(self, split_num):
        balanced_labeled_data = self.get_balanced_labeled_data()
        split_size = len(balanced_labeled_data) // split_num
        self.data_splits = []
        for cur_split_ind in range(split_num):
            if cur_split_ind == split_num - 1:
                self.data_splits.append(balanced_labeled_data)
            else:
                cur_data_split = random.sample(balanced_labeled_data, split_size)
                split_image_ids = [x[0] for x in cur_data_split]
                split_image_ids_dict = {x: True for x in split_image_ids}
                balanced_labeled_data = [x for x in balanced_labeled_data if x[0] not in split_image_ids_dict]
                self.data_splits.append(cur_data_split)

    @abc.abstractmethod
    def get_unwanted_image_ids(self):
        return
