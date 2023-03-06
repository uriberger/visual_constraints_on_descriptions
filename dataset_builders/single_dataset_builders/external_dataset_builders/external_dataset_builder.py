import abc
import os
from utils.general_utils import generate_dataset, for_loop_with_reports
from utils.visual_utils import get_image_shape
from dataset_builders.single_dataset_builders.single_dataset_builder import SingleDatasetBuilder


class ExternalDatasetBuilder(SingleDatasetBuilder):
    """ This class is the base class for all external dataset builders. """

    def __init__(self, name, language, struct_property, indent):
        super(ExternalDatasetBuilder, self).__init__(name, language, struct_property, indent)

        self.struct_data_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.extended_name}_{self.struct_property}_struct_data'
        )

        self.unwanted_image_ids_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_unwanted_image_ids'
        )

    """ Annotate the entire dataset: Create the struct data list, which is a list of (image id, val) pairs
        where the image ids are not unique and val is a binary value indicating whether the current struct property is
        expressed in a specific caption of this image. Alternatively, if image_id=False this is a list of
        (caption id, val) pairs where the caption ids are unique.
        This is list is for all the images in the dataset (from all original splits).
    """

    def get_struct_data(self, use_image_id=True):
        if use_image_id:
            return generate_dataset(self.struct_data_file_path, self.get_struct_data_internal, use_image_id)
        else:
            return generate_dataset(self.struct_data_file_path + '_caption_ids', self.get_struct_data_internal, use_image_id)

    @abc.abstractmethod
    def get_struct_data_internal(self):
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
        self.log_print('Filtering unwanted images from ' + self.name + '...')
        struct_data = self.get_struct_data()
        image_ids_by_struct_data = list(set([x[0] for x in struct_data]))

        self.unwanted_images_info = {
            'grayscale_count': 0,
            'missing_count': 0,
            '4_dim_count': 0,
            'unwanted_image_ids': []
        }

        self.increment_indent()
        for_loop_with_reports(image_ids_by_struct_data, len(image_ids_by_struct_data),
                              10000, self.is_unwanted_image, self.unwanted_images_progress_report)
        self.decrement_indent()

        self.log_print('Out of ' + str(len(image_ids_by_struct_data)) + ' images:')
        self.log_print('Found ' + str(self.unwanted_images_info['grayscale_count']) + ' grayscale images')
        self.log_print('Found ' + str(self.unwanted_images_info['4_dim_count']) + ' 4-dim images')
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
        elif image_shape[2] == 4:
            # Some images have 4 dims, don't know what to do with those
            self.unwanted_images_info['unwanted_image_ids'].append(image_id)
            self.unwanted_images_info['4_dim_count'] += 1

    def unwanted_images_progress_report(self, index, dataset_size, time_from_prev):
        self.log_print('Starting image ' + str(index) +
                       ' out of ' + str(dataset_size) +
                       ', time from previous checkpoint ' + str(time_from_prev))
