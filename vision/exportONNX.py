import torch
from torchvision import models
import torch.nn as nn

numClasses = 14
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, numClasses)
model.load_state_dict(torch.load("gestureModel/best_model.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)


torch.onnx.export( 
    model, 
    dummy_input,
    "gestureModel/gesturesModel.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=17,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

