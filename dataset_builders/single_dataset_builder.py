import abc
import os
import random
from utils.general_utils import project_root_dir, generate_dataset, for_loop_with_reports, get_image_id_to_prob
from utils.visual_utils import get_image_shape
from utils.text_utils import TextUtils
from dataset_builders.dataset_builder import DatasetBuilder


class SingleDatasetBuilder(DatasetBuilder):
    """ This class is the base class for all single datasets builders. """

    def __init__(self, name, data_split_str, struct_property, indent):
        super(SingleDatasetBuilder, self).__init__(name, data_split_str, struct_property, indent)

        # This is the directory in which we will keep the cached files of the datasets we create
        self.cached_dataset_files_dir = os.path.join(project_root_dir, 'cached_dataset_files')
        if not os.path.isdir(self.cached_dataset_files_dir):
            os.mkdir(self.cached_dataset_files_dir)

        self.struct_data_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_{TextUtils.get_language()}_{self.struct_property}_struct_data'
        )

        self.balanced_labeled_data_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_{TextUtils.get_language()}_{self.struct_property}_balanced_labeled_data'
        )

        self.train_val_split_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_{TextUtils.get_language()}_train_val_split'
        )

        self.unwanted_image_ids_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_unwanted_image_ids'
        )

    """ Train/Val splits:
        Since we're creating a new annotation and we want it to contain as much data as possible for all classes, we
        don't care about the original train/val splits.
        So, what we're going to do is unite all the data together, annotate it, and split so we'd have an equal number
        of instances of each class in the train and val sets.
    """

    """ Start by annotating the entire dataset: Create the struct data list, which is a list of (image id, val) pairs
        where the image ids are not unique and val is a binary value indicating whether the current struct property is
        expressed in a specific caption of this image.
        This is list is for all the images in the dataset (from all original splits).
    """

    def get_struct_data(self):
        return generate_dataset(self.struct_data_file_path, self.get_struct_data_internal)

    @abc.abstractmethod
    def get_struct_data_internal(self):
        return

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

    def find_samples_for_labels(self, labeled_data):
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

    def get_labeled_data_for_split(self):
        balanced_labeled_data = self.get_balanced_labeled_data()
        split_to_image_ids = generate_dataset(self.train_val_split_file_path, self.create_train_val_split)
        split_image_ids = split_to_image_ids[self.data_split_str]
        split_image_ids_dict = {x: True for x in split_image_ids}
        return [x for x in balanced_labeled_data if x[0] in split_image_ids_dict]

    # Functionality for filtering unwanted images

    """ We want to filter images that are:
            - Grayscale
            - Missing
        This function returns a list of image ids of images we want to filter.
    """

    def get_unwanted_image_ids(self):
        return generate_dataset(self.unwanted_image_ids_file_path, self.get_unwanted_image_ids_internal)

    def get_unwanted_image_ids_internal(self):
        self.log_print('Filtering unwanted images from ' + self.name + '...')
        struct_data = self.get_struct_data()
        image_ids_by_struct_data = list(set([x[0] for x in struct_data]))

        self.unwanted_images_info = {
            'grayscale_count': 0,
            'missing_count': 0,
            'unwanted_image_ids': []
        }

        self.increment_indent()
        for_loop_with_reports(image_ids_by_struct_data, len(image_ids_by_struct_data),
                              10000, self.is_unwanted_image, self.unwanted_images_progress_report)
        self.decrement_indent()

        self.log_print('Out of ' + str(len(image_ids_by_struct_data)) + ' images:')
        self.log_print('Found ' + str(self.unwanted_images_info['grayscale_count']) + ' grayscale images')
        self.log_print(str(self.unwanted_images_info['missing_count']) + ' images were missing')

        self.log_print('Finished filtering unwanted images from ' + self.name)
        return self.unwanted_images_info['unwanted_image_ids']

    """ This function checks if current image should be filtered, and if so, adds it to the unwanted image list. """

    def is_unwanted_image(self, index, item, print_info):
        image_id = item

        image_path = self.get_image_path_finder().get_image_path(image_id)
        image_shape = get_image_shape(image_path)
        if image_shape is None:
            # Missing image
            self.unwanted_images_info['unwanted_image_ids'].append(image_id)
            self.unwanted_images_info['missing_count'] += 1
        elif len(image_shape) == 2:
            # Grayscale images only has 2 dims
            self.unwanted_images_info['unwanted_image_ids'].append(image_id)
            self.unwanted_images_info['grayscale_count'] += 1

    def unwanted_images_progress_report(self, index, dataset_size, time_from_prev):
        self.log_print('Starting image ' + str(index) +
                       ' out of ' + str(dataset_size) +
                       ', time from previous checkpoint ' + str(time_from_prev))
