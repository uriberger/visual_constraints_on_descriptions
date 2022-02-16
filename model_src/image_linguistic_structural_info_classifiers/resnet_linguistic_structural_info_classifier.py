import torchvision.models as models
from model_src.image_linguistic_structural_info_classifiers.image_linguistic_structural_info_classifier import ImLingStructInfoClassifier


class ResNetLingStructInfoClassifier(ImLingStructInfoClassifier):

    def __init__(self, config, model_dir, model_name):
        super(ResNetLingStructInfoClassifier, self).__init__(config, model_dir, model_name)

        if config.freeze_backbone:
            self.backbone_model.eval()

    def backbone_model_inference(self, input_tensor):
        return self.backbone_model(input_tensor)

    def generate_backbone_model(self):
        return models.resnet50(pretrained=self.config.freeze_backbone).to(self.device)
