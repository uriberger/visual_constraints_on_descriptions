import abc
import os
from utils.general_utils import project_root_dir, generate_dataset, for_loop_with_reports
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
            f'{self.name}_{TextUtils.get_language()}_{self.data_split_str}_set_{self.struct_property}'
        )

        self.unwanted_image_ids_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_{TextUtils.get_language()}_unwanted_image_ids_{self.data_split_str}'
        )

    """ Create the image id to linguistic structural info mapping. """

    def create_struct_data(self):
        return generate_dataset(self.struct_data_file_path, self.create_struct_data_internal)

    @abc.abstractmethod
    def create_struct_data_internal(self, struct_property):
        return

    # Functionality for filtering unwanted images

    """ We want to filter images that are:
            - Grayscale
            - Missing
        This function returns a list of image ids of images we want to filter.
    """

    def get_unwanted_image_ids(self):
        return generate_dataset(self.unwanted_image_ids_file_path, self.get_unwanted_image_ids_internal)

    def get_unwanted_image_ids_internal(self):
        struct_data = self.create_struct_data()
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

        return self.unwanted_images_info['unwanted_image_ids']

    """ This function checks if current image should be filtered, and if so, adds it to the unwanted image list. """

    def is_unwanted_image(self, index, item, print_info):
        image_id = item

        image_path = self.image_path_finder.get_image_path(image_id)
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
