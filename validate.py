import cv2
import torch
import sys

sys.path.append(r"D:\CS640-Project-TransUnet\TransUnet\model\networks")

from model.networks.TransUnet import get_transNet
from data import ImageFolder
from TransUnet.loss.dice import *
from TransUnet.loss.loss import SoftDiceLossV2

LOSS = False
batch_size = 1

data_root = 'D:/CS640-Project-TransUnet/TransUnet/dataset/build'
val_set = ImageFolder(data_root, mode='val')
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
)

palette = [[0], [50], [100], [175], [255]]
criterion = SoftDiceLossV2(activation='sigmoid').cuda()
model_path = 'D:/CS640-Project-TransUnet/TransUnet/checkpoint/trans_builddice.pth'
model = torch.load(model_path)
model.eval()


def auto_val(model):
    iters = 0
    size = 8
    imgs = []
    preds = []
    gts = []
    dices = 0
    dice_Esophagus = 0
    dice_Heart = 0
    dice_Trachea = 0
    dice_Aorta = 0
    for i, (img, mask) in enumerate(val_loader):
        im = img
        img = img.cuda()
        model = model.cuda()
        pred = model(img)
        pred = torch.sigmoid(pred)
        iters += batch_size
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        # dice_Esophagus = diceCoeffv2(pred[:, 0:1, :], mask[:, 0:1, :], activation=None)
        # dice_Heart = diceCoeffv2(pred[:, 1:2, :], mask[:, 1:2, :], activation=None)
        # dice_Trachea = diceCoeffv2(pred[:, 2:3, :], mask[:, 2:3, :], activation=None)
        # dice_Aorta = diceCoeffv2(pred[:, 3:4, :], mask[:, 3:4, :], activation=None)
        mean_dice = (dice_Aorta+dice_Trachea+dice_Esophagus+dice_Heart)/4
        gt = mask.numpy()[0].transpose([1, 2, 0])
        gt = onehot_to_mask(gt, palette)
        # print(pred.shape)
        pred = pred.cpu().detach().numpy()[0].transpose([1, 2, 0])
        pred = onehot_to_mask(pred, palette)
        im = im[0].numpy().transpose([1, 2, 0])
        if len(imgs) < size:
            imgs.append(im * 255)
            preds.append(pred)
            gts.append(gt)
    imgs = np.hstack([*imgs])
    preds = np.hstack([*preds])
    gts = np.hstack([*gts])
    # print(imgs[0].shape, preds[0].shape, gts[0].shape)
    # print(len(imgs), len(preds), len(gts))

    show_res = np.vstack(np.uint8([imgs, preds, gts]))
    cv2.imshow("top is mri , middle is pred,  bottom is gt", show_res)
    cv2.waitKey(0)


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


if __name__ == '__main__':
    # val(model)
    auto_val(model)
