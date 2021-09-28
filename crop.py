import PIL
from PIL import Image
import numpy as np

a = np.array(Image.open("ownership/res18_cifar100_qrcode_0.5.png"))
a = np.array(a)
a = a[57:429, 142:514, :]
a = Image.fromarray(a)
a.save("res18_cifar100_qrcode_0.5_crop.png")

"""
from models.resnet import resnet18,resnet50
m1 = resnet18(num_classes=1000, imagenet=True)
m2 = resnet50(num_classes=1000, imagenet=True)

from torchsummary import summary
summary(m1, (3, 224, 224))
"""