import sys
from model_src.model_factory import ModelFactory

if len(sys.argv) != 3:
    print('Please enter two arguments: a path to a model directory and a model name')
    assert False
model_dir = sys.argv[1]
model_name = sys.argv[2]

model_factory = ModelFactory(1)
_, config = model_factory.load_model(model_dir, model_name)
print(config)
