import abc
from collections import defaultdict
from loggable_object import LoggableObject
import os
from utils.general_utils import project_root_dir
from dataset_src.image_linguistic_structural_info_dataset import ImLingStructInfoDataset


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

        self.image_path_finder = self.create_image_path_finder()

    # General static setting of the datasets dir, for all datasets

    @staticmethod
    def set_datasets_dir(dir_path):
        global datasets_dir
        datasets_dir = dir_path

    @staticmethod
    def get_datasets_dir():
        return datasets_dir

    """ Load the dataset if it's cached, otherwise build it. Arguments:
        aggregation_func: Each image usually has multiple captions. If aggregation_func isn't None we use it to
        aggregate the labels of all the caption of an image to a single label.
    """

    def build_dataset(self, aggregation_func=None):
        self.log_print('Generating ' + self.name +
                       ' ' + self.data_split_str +
                       ' ' + self.struct_property + ' dataset...')

        self.increment_indent()
        struct_data = self.create_struct_data()

        if aggregation_func is not None:
            image_id_to_labels = defaultdict(list)
            for image_id, label in struct_data:
                image_id_to_labels[image_id].append(label)
            struct_data = [(x[0], aggregation_func(x[1])) for x in image_id_to_labels.items()]

        self.decrement_indent()

        self.log_print('Filtering unwanted images from ' + self.name + ' ' + self.data_split_str + ' set...')
        self.increment_indent()
        unwanted_image_ids = self.get_unwanted_image_ids()
        unwanted_image_ids = {image_id: True for image_id in unwanted_image_ids}  # For more efficient retrieval
        self.decrement_indent()

        struct_data = [x for x in struct_data if x[0] not in unwanted_image_ids]

        return ImLingStructInfoDataset(struct_data, self.image_path_finder)

    """ Create the image id to linguistic structural info mapping. """

    @abc.abstractmethod
    def create_struct_data(self):
        return

    """ Create the ImagePathFinder object for this dataset. """

    @abc.abstractmethod
    def create_image_path_finder(self):
        return

    """ Return the list of all image ids in this dataset. """

    @abc.abstractmethod
    def get_all_image_ids(self):
        return

    @abc.abstractmethod
    def get_unwanted_image_ids(self):
        return
