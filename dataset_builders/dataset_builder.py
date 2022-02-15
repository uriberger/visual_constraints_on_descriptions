import abc
from loggable_object import LoggableObject
import os
import torch
from utils.general_utils import project_root_dir, generate_dataset, for_loop_with_reports
from utils.visual_utils import get_image_shape
from dataset_src.image_linguistic_structural_info_dataset import ImLingStructInfoDataset


datasets_dir = os.path.join(project_root_dir, '../..', 'datasets')


class DatasetBuilder(LoggableObject):
    """ This class is the base class for all external datasets builders.
        We have two dataset related objects: dataset builders and ImLingStructInfoDataset objects.
        The dataset builders save the data from external datasets in an organized manner to an external file.
        Each dataset builder has a 'build_dataset' function that creates a ImLingStructInfoDataset object. The constructors
        of these objects expect 2 things: the path to the file were we store the data, and an ImagePathFinder object that,
        given an image id, returns the path to the relevant image (since the stored data only holds image id and not the
        images themselves).
    """

    def __init__(self, name, data_split_str, indent):
        super(DatasetBuilder, self).__init__(indent)
        self.name = name
        self.data_split_str = data_split_str

        # The datasets are assumed to be located in a sibling directory named 'datasets'
        self.datasets_dir = datasets_dir

        # This is the directory in which we will keep the cached files of the datasets we create
        self.cached_dataset_files_dir = os.path.join(project_root_dir, 'cached_dataset_files')
        if not os.path.isdir(self.cached_dataset_files_dir):
            os.mkdir(self.cached_dataset_files_dir)

        self.struct_data_file_path = os.path.join(self.cached_dataset_files_dir,
                                                  self.name + '_' + self.data_split_str + '_set')

    # General static setting of the datasets dir, for all datasets

    @staticmethod
    def set_datasets_dir(dir_path):
        global datasets_dir
        datasets_dir = dir_path

    @staticmethod
    def get_datasets_dir():
        return datasets_dir

    # Instance specific functionality

    """ Load the dataset if it's cached, otherwise build it. """

    def build_dataset(self):
        self.log_print('Generating ' + self.name + ' ' + self.data_split_str + ' dataset...')

        self.create_struct_data()
        image_path_finder = self.create_image_path_finder()

        return ImLingStructInfoDataset(self.struct_data_file_path, image_path_finder)

    """ Create the image id to linguistic structural info mapping. """

    def create_struct_data(self):
        return generate_dataset(self.struct_data_file_path, self.create_struct_data_internal)

    @abc.abstractmethod
    def create_struct_data_internal(self):
        return

    """ Create the ImagePathFinder object for this dataset. """

    @abc.abstractmethod
    def create_image_path_finder(self):
        return

    # Functionality for filtering unwanted images

    """ We want to filter images that are:
            - Grayscale
        This function returns a list of image ids of images we want to filter.
    """

    def find_unwanted_images(self):
        struct_data = self.create_struct_data()
        image_ids_by_struct_data = list(set([x[0] for x in struct_data]))

        self.unwanted_images_info = {
            'grayscale_count': 0,
            'unwanted_image_ids': []
        }

        self.increment_indent()
        for_loop_with_reports(image_ids_by_struct_data, len(image_ids_by_struct_data),
                              10000, self.is_unwanted_image, self.unwanted_images_progress_report)
        self.decrement_indent()

        self.log_print('Out of ' + str(len(image_ids_by_struct_data)) + ' images:')
        self.log_print('Found ' + str(self.unwanted_images_info['grayscale_count']) + ' grayscale images')

        return self.unwanted_images_info['unwanted_image_ids']

    """ This function checks if current image should be filtered, and if so, adds it to the unwanted image list. """

    def is_unwanted_image(self, index, item, print_info):
        image_id = item

        # Grayscale
        image_path = self.image_path_finder.get_image_path(image_id)
        image_shape = get_image_shape(image_path)
        if len(image_shape) == 2:
            # Grayscale images only has 2 dims
            self.unwanted_images_info['unwanted_image_ids'].append(image_id)
            self.unwanted_images_info['grayscale_count'] += 1
            return

    def unwanted_images_progress_report(self, index, dataset_size, time_from_prev):
        self.log_print('Starting image ' + str(index) +
                       ' out of ' + str(dataset_size) +
                       ', time from previous checkpoint ' + str(time_from_prev))

    """ We want to filter images that are:
        - Grayscale
    """

    def filter_unwanted_images(self):
        self.image_path_finder = self.create_image_path_finder()
        unwanted_image_ids = self.find_unwanted_images()

        struct_data = self.create_struct_data()
        new_struct_data = [x for x in struct_data if x[0] not in unwanted_image_ids]

        torch.save(new_struct_data, self.struct_data_file_path)
