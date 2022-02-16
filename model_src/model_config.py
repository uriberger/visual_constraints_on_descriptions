class ModelConfig:
    """ Contains configuration settings for our clustering models.


    backbone_model: The underlying visual model, used to extract image embeddings

    freeze_backbone: A flag indicating whether the visual underlying model should be trained along with the classifier

    struct_property: The linguistic structural property that the model is trained to predict

    learning_rate: The learning rate for the visual encoder's training
    """

    def __init__(self,
                 backbone_model='resnet50',
                 freeze_backbone=True,
                 struct_property='passive',
                 learning_rate=1e-4,
                 ):
        super(ModelConfig, self).__init__()

        self.backbone_model = backbone_model
        self.freeze_backbone = freeze_backbone
        self.struct_property = struct_property
        self.learning_rate = learning_rate

    def __str__(self):
        return 'Configuration: ' + str(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, ModelConfig):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.__dict__ == other.__dict__
