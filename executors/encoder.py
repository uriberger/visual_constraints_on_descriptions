import os
import torch
import torch.utils.data as data

from utils.general_utils import for_loop_with_reports
from utils.visual_utils import wanted_image_size
from executors.executor import Executor
from model_src.model_factory import ModelFactory

BATCH_REPORT_NUM = 1000


class Encoder(Executor):

    def __init__(self, data_set, output_dir, loaded_model_dir, loaded_model_name, indent):
        super().__init__(indent)

        self.data_set = data_set
        self.encoding_file_path = os.path.join(
            output_dir,
            f'{loaded_model_name}_encodings.enc'
        )

        model_factory = ModelFactory(self.indent + 1)
        self.model, self.model_config = model_factory.load_model(loaded_model_dir, loaded_model_name)
        self.model.eval()

        # Find sample num on output size, to be used as dimensions for the encoding matrix
        sample_num = len(data_set)
        dummy_input = torch.zeros(1, 3, wanted_image_size[0], wanted_image_size[1]).to(self.device)
        dummy_output = self.model(dummy_input)
        output_size = dummy_output.shape[1]

        self.encoding_mat = torch.zeros(sample_num, output_size)
        self.encoding_mat_index = 0
        self.encoding_file_path = os.path.join(
            output_dir,
            f'{loaded_model_name}_encodings.enc'
        )

    """ The core function: encode the given batch. """

    def encode_batch(self, index, sampled_batch, print_info):
        # Load data
        image_tensor = sampled_batch['image'].to(self.device)
        output = self.model(image_tensor)
        batch_size = image_tensor.shape[0]

        start = self.encoding_mat_index
        end = self.encoding_mat_index + batch_size
        self.encoding_mat[start:end, :] = output
        self.encoding_mat_index += batch_size

    """ Encode the data set; This is the entry point of this class. """

    def encode(self):
        dataloader = data.DataLoader(self.data_set, batch_size=50, shuffle=False)

        checkpoint_len = BATCH_REPORT_NUM
        self.increment_indent()
        for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                              self.encode_batch, self.progress_report)
        self.decrement_indent()

        torch.save(self.encoding_mat, self.encoding_file_path)
