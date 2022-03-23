from utils.general_utils import init_entry_point, log_print, project_root_dir
from model_src.model_config import ModelConfig
from executors.trainer import Trainer
from dataset_builders.dataset_builder import DatasetBuilder
from dataset_builders.dataset_builder_creator import create_dataset_builder
import os
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser(description='Train and evaluate a multimodal word learning model.')
parser.add_argument('--write_to_log', action='store_true', default=False, dest='write_to_log',
                    help='redirect output to a log file')
parser.add_argument('--datasets_dir', type=str, default=os.path.join('..', 'datasets'), dest='datasets_dir',
                    help='the path to the datasets dir')
parser.add_argument('--language', type=str, default='English', dest='language',
                    help='the language of the used dataset')
parser.add_argument('--struct_property', type=str, dest='struct_property',
                    help='the linguistic structural property to be examined')
parser.add_argument('--dataset', type=str, dest='dataset',
                    help='the name of the used dataset')
parser.add_argument('--translated', action='store_true', default=False, dest='translated',
                    help='use translated captions')
parser.add_argument('--dump_captions', action='store_true', default=False, dest='dump_captions',
                    help='only dump the captions of the dataset and exit')
args = parser.parse_args()
write_to_log = args.write_to_log
datasets_dir = args.datasets_dir
language = args.language
struct_property = args.struct_property
dataset_name = args.dataset
translated = args.translated
dump_captions = args.dump_captions

DatasetBuilder.set_datasets_dir(datasets_dir)


def get_image_id_to_count(struct_data):
    image_id_to_count = defaultdict(int)
    image_id_to_pos_count = defaultdict(int)
    for image_id, has_num in struct_data:
        image_id_to_count[image_id] += 1
        image_id_to_pos_count[image_id] += has_num
    return image_id_to_count, image_id_to_pos_count


def get_image_id_to_prob(image_id_to_count, image_id_to_pos_count):
    image_id_to_prob = {x: image_id_to_pos_count[x] / image_id_to_count[x] for x in image_id_to_count.keys()}
    return image_id_to_prob


def get_class_to_image_list(gt_class_data):
    from collections import defaultdict
    class_to_image_list = defaultdict(list)
    for image_id, gt_class_list in gt_class_data.items():
        for gt_class in gt_class_list:
            class_to_image_list[gt_class].append(image_id)
    return class_to_image_list


def main(should_write_to_log):
    function_name = 'main'
    timestamp = init_entry_point(should_write_to_log, language)

    model_config = ModelConfig(struct_property=struct_property)
    log_print(function_name, 0, str(model_config))

    log_print(function_name, 0, 'Generating datasets...')
    training_set_builder = create_dataset_builder(dataset_name, 'train', struct_property, translated)
    if dump_captions:
        training_set_builder.dump_captions()
        return
    training_set = training_set_builder.build_dataset()
    # Delete start
    # gt_class_mapping = training_set_builder.get_gt_classes_data()
    # struct_data = training_set.struct_data
    # all_image_ids = list(set([x[0] for x in struct_data if x[0] in gt_class_mapping]))
    # gt_class_mapping = {x: gt_class_mapping[x] for x in all_image_ids}
    # all_image_ids_dict = {x: True for x in all_image_ids}
    # struct_data = [x for x in training_set.struct_data if x[0] in all_image_ids_dict]
    # class_to_image_list = get_class_to_image_list(gt_class_mapping)
    # image_id_to_num_prob = get_image_id_to_num_prob(struct_data)
    # class_to_num_prob_mean = {
    #     i: sum([image_id_to_num_prob[x] for x in class_to_image_list[i]]) / len(class_to_image_list[i]) for i in
    #     range(80)}
    # a = list(class_to_num_prob_mean.items())
    # a.sort(reverse=True, key=lambda x: x[1])
    # class_mapping = training_set_builder.get_class_mapping()
    # a = [(class_mapping[x[0]], x[1]) for x in a]
    # Delete end
    test_set_builder = create_dataset_builder(dataset_name, 'val', struct_property, translated)
    test_set = test_set_builder.build_dataset()
    log_print(function_name, 0, 'datasets generated')

    log_print(function_name, 0, 'Training model...')
    model_root_dir = os.path.join(project_root_dir, timestamp)
    trainer = Trainer(model_root_dir, training_set, test_set, 5, 50, model_config, 1)
    trainer.train()
    log_print(function_name, 0, 'Finished training model')


main(write_to_log)
