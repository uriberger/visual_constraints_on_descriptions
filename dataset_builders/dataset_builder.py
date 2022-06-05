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

    def __init__(self, name, struct_property, indent):
        super(DatasetBuilder, self).__init__(indent)

        # This is the directory in which we will keep the cached files of the datasets we create
        self.cached_dataset_files_dir = os.path.join(project_root_dir, 'cached_dataset_files')
        if not os.path.isdir(self.cached_dataset_files_dir):
            os.mkdir(self.cached_dataset_files_dir)

        self.name = name
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

    def build_dataset(self, data_split_str, cross_validation_ind=-1):
        self.log_print('Generating ' + self.name +
                       ' ' + data_split_str +
                       ' ' + self.struct_property + ' dataset...')

        self.increment_indent()
        if data_split_str == 'all':
            labeled_data = self.get_labeled_data()
        elif cross_validation_ind >= 0:
            labeled_data = self.get_cross_validation_data(data_split_str, cross_validation_ind)
        else:
            labeled_data = self.get_labeled_data_for_split(data_split_str)
        self.decrement_indent()

        return ImLingStructInfoDataset(labeled_data, self.get_image_path_finder())

    """ Get the (image id, label) pair list. """

    @abc.abstractmethod
    def get_labeled_data(self):
        return

    """ Get the (image id, label) pair list for the relevant split. """

    @abc.abstractmethod
    def get_labeled_data_for_split(self, data_split_str):
        return

    """ Create random splits for cross validation. """

    @abc.abstractmethod
    def generate_cross_validation_data(self, split_num):
        return

    """ Get cross validation splits; For train, get all splits except split_ind, for val, get only split_ind. """

    def get_cross_validation_data(self, train_or_val, split_ind):
        if train_or_val == 'val':
            return self.data_splits[split_ind]
        else:
            res = []
            for cur_split_ind in range(len(self.data_splits)):
                if cur_split_ind != split_ind:
                    res += self.data_splits[cur_split_ind]
            return res

    """ Create the ImagePathFinder object for this dataset. """

    @abc.abstractmethod
    def create_image_path_finder(self):
        return

    def get_image_path_finder(self):
        if self.image_path_finder is None:
            self.image_path_finder = self.create_image_path_finder()
        return self.image_path_finder

    def get_gt_classes_data(self):
        return None

    def get_gt_bboxes_data(self):
        return None

    def get_class_mapping(self):
        return None
