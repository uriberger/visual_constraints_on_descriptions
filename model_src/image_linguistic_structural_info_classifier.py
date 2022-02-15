import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.visual_utils import wanted_image_size


class ImLingStructInfoClassifier(nn.Module):

    """ This class predicts linguistic structural info of image descriptions, given only the images.
        It is based on a backbone visual model with a classification head for each predicted property.
    """

    def __init__(self, struct_properties, freeze_backbone):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.backbone_model = self.generate_backbone_model()

        dummy_input = torch.zeros(1, 3, wanted_image_size[0], wanted_image_size)
        dummy_output = self.backbone_model_inference(dummy_input)
        backbone_output_size = dummy_output.shape[1]

        self.classification_heads = []
        for struct_property in struct_properties:
            if struct_property == 'passive':
                fc = nn.Linear(backbone_output_size, 2)
                sm = nn.Softmax(dim=1)
                self.classification_heads.append(nn.Sequential(fc, F.relu, sm))
            elif struct_property == 'empty_frame_slots_num':
                fc = nn.Linear(backbone_output_size, 6)
                sm = nn.Softmax(dim=1)
                self.classification_heads.append(nn.Sequential(fc, F.relu, sm))

        self.freeze_backbone = freeze_backbone

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.backbone_model_inference(x)
        else:
            x = self.backbone_model_inference(x)
        res = []
        for classification_head in self.classification_heads:
            res.append(classification_head(x))
        return res

    @abc.abstractmethod
    def generate_backbone_model(self):
        return

    @abc.abstractmethod
    def backbone_model_inference(self, input_tensor):
        return
