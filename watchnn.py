
from policy_value_net import PolicyValueNet
import torch
from torchvision.models import AlexNet
from torchviz import make_dot
 
x=torch.rand(1,3,8,8)
model=PolicyValueNet(8)
y=model(x)
g=make_dot(y, params=dict(model.named_parameters()))
g.view()

