import abc
from loggable_object import LoggableObject
import os
from utils.general_utils import project_root_dir
from dataset_src.image_linguistic_info_dataset import ImLingStructInfoDataset


datasets_dir = os.path.join(project_root_dir, '..', 'datasets')


class DatasetBuilder(LoggableObject):
    """ This class is the base class for all external datasets builders.
        We have two dataset related objects: dataset builders and ImLingStructInfoDataset objects.
        The dataset builders save the data from external datasets in an organized manner to an external file.
        Each dataset builder has a 'build_dataset' function that creates a ImLingStructInfoDataset object. The constructors
        of these objects expect 2 inputs: the path to the file were we store the data, and an ImagePathFinder object that,
        given an image id, returns the path to the relevant image (since the stored data only holds image id and not the
        images themselves).
    """

    def __init__(self, name, data_split_str, struct_property, indent):
        super(DatasetBuilder, self).__init__(indent)
        self.name = name
        self.data_split_str = data_split_str
        self.struct_property = struct_property

        # The datasets are assumed to be located in a sibling directory named 'datasets'
        self.datasets_dir = datasets_dir

        self.image_path_finder = None

    # General static setting of the datasets dir, for all datasets

    @staticmethod
    def set_datasets_dir(dir_path):
        global datasets_dir
        datasets_dir = dir_path

    @staticmethod
    def get_datasets_dir():
        return datasets_dir

    """ Build the dataset object. """

    def build_dataset(self):
        self.log_print('Generating ' + self.name +
                       ' ' + self.data_split_str +
                       ' ' + self.struct_property + ' dataset...')

        self.increment_indent()
        labeled_data_for_split = self.get_labeled_data_for_split()
        self.decrement_indent()

        return ImLingStructInfoDataset(labeled_data_for_split, self.get_image_path_finder())

    """ Get the (image id, label) pair list for the relevant split. """

    @abc.abstractmethod
    def get_labeled_data_for_split(self):
        return

    """ Create the ImagePathFinder object for this dataset. """

    @abc.abstractmethod
    def create_image_path_finder(self):
        return

    def get_image_path_finder(self):
        if self.image_path_finder is None:
            self.image_path_finder = self.create_image_path_finder()
        return self.image_path_finder
