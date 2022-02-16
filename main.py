from utils.general_utils import init_entry_point, log_print, project_root_dir
from model_src.model_config import ModelConfig
from executors.trainer import Trainer
from dataset_builders.image_caption_dataset_builders.coco_dataset_builder import CocoDatasetBuilder
import os

function_name = 'main'
timestamp = init_entry_point(False)

model_config = ModelConfig(struct_property='passive')
log_print(function_name, 0, str(model_config))

log_print(function_name, 0, 'Generating training set...')
dataset_builder = CocoDatasetBuilder(os.path.join('..', 'datasets', 'coco'), 'train', 'passive', 1)
training_set = dataset_builder.build_dataset()
log_print(function_name, 0, 'Training set generated')

log_print(function_name, 0, 'Training model...')
model_root_dir = os.path.join(project_root_dir, timestamp)
trainer = Trainer(model_root_dir, training_set, 5, 50, model_config, 1)
trainer.train()
log_print(function_name, 0, 'Finished training model')
