from utils.general_utils import models_dir
from loggable_object import LoggableObject


class Executor(LoggableObject):
    """ A general class for all executors (trainer, evaluator, etc.) to
        inherit from. """

    def __init__(self, indent):
        super(Executor, self).__init__(indent)

        self.models_dir = models_dir

    """ Report progress in a for-loop, given the current index, the size of the entire iterable, and the time from the
        previous checkpoint. """

    def progress_report(self, index, iterable_size, time_from_prev):
        self.log_print('Starting batch ' + str(index) +
                       ' out of ' + str(iterable_size) +
                       ', time from previous checkpoint ' + str(time_from_prev))
