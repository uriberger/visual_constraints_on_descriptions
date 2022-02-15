import torchvision.models as models
from model_src.image_linguistic_structural_info_classifier import ImLingStructInfoClassifier


class ResNetLingStructInfoClassifier(ImLingStructInfoClassifier):

    def __init__(self, struct_properties, use_pretrained):
        super(ResNetLingStructInfoClassifier, self).__init__(
            struct_properties=struct_properties,
            freeze_backbone=use_pretrained
        )

        if use_pretrained:
            self.backbone_model.eval()

    def backbone_model_inference(self, input_tensor):
        return self.backbone_model(input_tensor)

    def generate_backbone_model(self):
        return models.resnet50(pretrained=self.use_pretrained).to(self.device)
