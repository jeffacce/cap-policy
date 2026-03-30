import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # TODO: set model
        self.model = None

    def to(self, device):
        # TODO: move model to device
        if self.model is not None:
            self.model.to(device)
        return self

    def forward(self, x):
        # TODO: forward pass of model
        pass
    
    def step(self, data, *args, **kwargs):
        images, actions = data
        # images is a tensor of shape (1, image_buffer_size, 3, 256, 256), where the images are in chronological order
        # actions is a tensor of shape (1, image_buffer_size, 7), where the actions are in chronological order, and the final action is padding
        
        # TODO: implement pass that takes in observations as described above out outputs a 7-dimensional action
        action = torch.zeros(7)
        logs = {}
        
        return action, logs

    def reset(self):
        # TODO: optional; this method is called once the robot is homed
        pass