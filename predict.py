import numpy as np
import torch
import cv2
import sys

sys.path.append(r"D:\CS640-Project-TransUnet\TransUnet\model\networks")

from torchvision import transforms
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'D:/CS640-Project-TransUnet/TransUnet/checkpoint/trans_builddice.pth'
model = torch.load(model_path)
palette = [[0], [252], [253], [254], [255]]
model.eval()

test_path = os.listdir('D:/Segthordataset/test/Patient_41_png/Patient_41.nii.gz/')
save_root = 'D:/Segthordataset/test/Patient_41_png/predict/'


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


if __name__ == '__main__':
    root_path = 'D:/Segthordataset/test/Patient_41_png/Patient_41.nii.gz/'
    with torch.no_grad():
        for img_name in test_path:
            image_path = os.path.join(root_path, img_name)
            img = cv2.imread(image_path, 0)
            img = np.expand_dims(img, axis=2)
            img = np.array(img, np.float32).transpose([2, 0, 1])
            img = torch.tensor(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.cuda()
            model = model.cuda()
            # print(img.shape)
            pred = model(img)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().detach()
            pred[pred < 0.5] = 0
            pred[pred > 0.5] = 1
            pred = pred.cpu().detach().numpy()[0].transpose([1, 2, 0])
            pred = onehot_to_mask(pred, palette)
            pred = torch.tensor(pred)
            pred = torch.squeeze(pred)
            pred = np.array(pred)
            # print(pred.shape)
            # pred = torch.squeeze(pred)
            pred = np.uint8(pred)
            save_path = os.path.join(save_root, img_name)
            cv2.imwrite(save_path, pred)

        print('Finished. ')




