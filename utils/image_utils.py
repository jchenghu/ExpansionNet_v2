
import torchvision
from PIL import Image as PIL_Image


def preprocess_image(image_path, img_size):
    transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size))])
    transf_2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    pil_image = PIL_Image.open(image_path)
    if pil_image.mode != 'RGB':
        pil_image = PIL_Image.new("RGB", pil_image.size)
    preprocess_pil_image = transf_1(pil_image)
    image = torchvision.transforms.ToTensor()(preprocess_pil_image)
    image = transf_2(image)
    return image.unsqueeze(0)
