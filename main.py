from dataset_builders.imsitu_dataset_builder import ImSituDatasetBuilder
from dataset_builders.image_caption_dataset_builders.coco_dataset_builder import CocoDatasetBuilder
import os

a1 = ImSituDatasetBuilder(os.path.join('..', 'datasets', 'ImSitu'), 'train', 'empty_frame_slots_num', 1).build_dataset()
a2 = ImSituDatasetBuilder(os.path.join('..', 'datasets', 'ImSitu'), 'dev', 'empty_frame_slots_num', 1).build_dataset()
a3 = ImSituDatasetBuilder(os.path.join('..', 'datasets', 'ImSitu'), 'test', 'empty_frame_slots_num', 1).build_dataset()
b1 = CocoDatasetBuilder(os.path.join('..', 'datasets', 'coco'), 'train', 'passive', 1).build_dataset()
b2 = CocoDatasetBuilder(os.path.join('..', 'datasets', 'coco'), 'val', 'passive', 1).build_dataset()
c1 = CocoDatasetBuilder(os.path.join('..', 'datasets', 'coco'), 'train', 'transitivity', 1).build_dataset()
c2 = CocoDatasetBuilder(os.path.join('..', 'datasets', 'coco'), 'val', 'transitivity', 1).build_dataset()

print(len(a1))
print(len(a2))
print(len(a3))
print(len(b1))
print(len(b2))
print(len(c1))
print(len(c2))
