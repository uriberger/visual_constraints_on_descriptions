import clip
from model_src.image_linguistic_structural_info_classifiers.image_linguistic_structural_info_classifier import ImLingStructInfoClassifier


class CLIPLingStructInfoClassifier(ImLingStructInfoClassifier):

    def __init__(self, config, model_dir, model_name):
        super(CLIPLingStructInfoClassifier, self).__init__(config, model_dir, model_name)

        if config.freeze_backbone:
            self.backbone_model.eval()

    def backbone_model_inference(self, input_tensor):
        return self.backbone_model_model.encode_image(input_tensor).float()

    def generate_backbone_model(self):
        return clip.load('RN50', self.device)[0]
