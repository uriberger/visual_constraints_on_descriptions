from dataset_builders.imsitu_dataset_builder import ImSituDatasetBuilder
from dataset_builders.coco_dataset_builder import CocoDatasetBuilder
import os

a1 = ImSituDatasetBuilder(os.path.join('..', 'datasets', 'ImSitu'), 'train', 1).build_dataset()
a2 = ImSituDatasetBuilder(os.path.join('..', 'datasets', 'ImSitu'), 'dev', 1).build_dataset()
a3 = ImSituDatasetBuilder(os.path.join('..', 'datasets', 'ImSitu'), 'test', 1).build_dataset()
b1 = CocoDatasetBuilder(os.path.join('..', 'datasets', 'coco'), 'train', 1).build_dataset()
b2 = CocoDatasetBuilder(os.path.join('..', 'datasets', 'coco'), 'val', 1).build_dataset()

print(len(a1))
print(len(a2))
print(len(a3))
print(len(b1))
print(len(b2))
