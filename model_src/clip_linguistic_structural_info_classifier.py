import clip
from model_src.image_linguistic_structural_info_classifier import ImLingStructInfoClassifier


class CLIPLingStructInfoClassifier(ImLingStructInfoClassifier):

    def __init__(self, struct_properties, use_pretrained):
        super(CLIPLingStructInfoClassifier, self).__init__(
            struct_properties=struct_properties,
            freeze_backbone=use_pretrained
        )

        if use_pretrained:
            self.backbone_model.eval()

    def backbone_model_inference(self, input_tensor):
        return self.backbone_model_model.encode_image(input_tensor).float()

    def generate_backbone_model(self):
        return clip.load('RN50', self.device)[0]
