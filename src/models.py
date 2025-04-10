import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import timm


def set_vit_mlp_dropout(model, p=0.1):
    """
    Adjusts dropout in the MLP layers of ViT encoder blocks.
    """
    for block in model.encoder.layers:
        if hasattr(block, "mlp"):
            mlp = block.mlp
            for i, layer in enumerate(mlp):
                if isinstance(layer, nn.Dropout):
                    mlp[i] = nn.Dropout(p)


def get_model(model_name: str):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 102)

    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout before the final layer
            nn.Linear(model.heads.head.in_features, 102),
        )
        # Set MLP dropout inside Transformer blocks
        set_vit_mlp_dropout(model, p=0.2)

    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 102)

    elif model_name == "vit_small_patch16_224":
        model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=102,
            drop_rate=0.2,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,  # stochastic depth
        )

    else:
        raise ValueError(
            "Unsupported model. Choose 'resnet50', 'vit_b_16', or 'efficientnet_b0'."
        )

    return model
