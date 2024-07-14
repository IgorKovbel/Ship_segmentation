import matplotlib.pyplot as plt
from skimage.io import imread
import albumentations as A
import numpy as np
import argparse
import torch

from model import *

def get_augs():
    return A.Compose([
        A.Resize(256, 256)
    ])

def process_image(image_path):
    image = imread(image_path)

    augs = get_augs()
    augmented = augs(image=image)
    image = augmented['image']
    
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.Tensor(image) / 255.0

    return image

def segment_image(image_path, model):
    image = process_image(image_path)

    logits_mask = model(image.to('cuda').unsqueeze(0))
    pred_mask = torch.sigmoid(logits_mask)

    return image, pred_mask

def main(image_path):
    
    encoder = ResNetEncoder()
    decoder = UnetDecoder()
    segmentation_head = SegmentationHead()

    model = SegmentationModel(encoder, decoder, segmentation_head)

    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    model.to('cuda')

    image, pred_mask = segment_image(image_path, model)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image.permute(1, 2, 0))
    ax1.axis('off')

    ax2.imshow(pred_mask[0][0].to('cpu').detach().numpy(), cmap='gray')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Inference")
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    
    args = parser.parse_args()
    main(args.image_path)
