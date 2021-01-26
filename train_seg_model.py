import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import image
import numpy as np
from random import seed
from sim import get_tableau_palette
from seg_model_parts import *

# ==================================================
mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]


# ==================================================

class RGBDataset(Dataset):
    def __init__(self, img_dir):
        """
            Initialize instance variables.
            :param img_dir (str): path of train or test folder.
            :return None:
        """
        # TODO: complete this method
        # ===============================================================================
        self.img_dir = img_dir
        # transform to be applied on a sample.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_rgb, std_rgb)
        ])
        # ===============================================================================

    def __len__(self):
        """
            Return the length of the dataset.
            :return dataset_length (int): length of the dataset, i.e. number of samples in the dataset
        """
        # TODO: complete this method
        # ===============================================================================
        dataset_length = len([file for file in os.listdir(os.path.join(self.img_dir, "rgb")) if file.endswith(".png")])
        return dataset_length
        # ===============================================================================

    def __getitem__(self, idx):
        """
            Given an index, return paired rgb image and ground truth mask as a sample.
            :param idx (int): index of each sample, in range(0, dataset_length)
            :return sample: a dictionary that stores paired rgb image and corresponding ground truth mask.
        """
        # TODO: complete this method
        # Hint:
        # - Use image.read_rgb() and image.read_mask() to read the images.
        # - Think about how to associate idx with the file name of images.
        # - Remember to apply transform on the sample.
        # ===============================================================================
        rgb_img = self.transform(image.read_rgb(os.path.join(self.img_dir, "rgb", "{}_rgb.png".format(idx))))
        gt_mask = torch.LongTensor(image.read_mask(os.path.join(self.img_dir, "gt", "{}_gt.png".format(idx))))
        sample = {'input': rgb_img, 'target': gt_mask}
        return sample
        # ===============================================================================


class miniUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super(miniUNet, self).__init__()
        # TODO: complete this method
        # ===============================================================================

        self.inConv = DoubleConv(n_channels, 64)
        self.down_Conv1 = Down(64, 128)
        self.down_Conv2 = Down(128, 256)
        self.down_Conv3 = Down(256, 512)
        self.down_Conv4 = Down(512, 1024 // 2)

        self.up_Conv4 = Up(1024, 512 // 2, bilinear=True)
        self.up_Conv3 = Up(512, 256 // 2, bilinear=True)
        self.up_Conv2 = Up(256, 128 // 2, bilinear=True)
        self.up_Conv1 = Up(128, 64, bilinear=True)

        self.outConv = OutConv(64, n_classes)

        # ===============================================================================

    def forward(self, x):
        # TODO: complete this method
        # ===============================================================================
        x1 = self.inConv(x)
        x2 = self.down_Conv1(x1)
        x3 = self.down_Conv2(x2)
        x4 = self.down_Conv3(x3)
        x5 = self.down_Conv4(x4)

        x = self.up_Conv4(x5, x4)
        x = self.up_Conv3(x, x3)
        x = self.up_Conv2(x, x2)
        x = self.up_Conv1(x, x1)

        output = self.outConv(x)
        return output
        # ===============================================================================


def save_chkpt(model, epoch, test_miou, chkpt_path):
    """
        Save the trained model.
        :param model (torch.nn.module object): miniUNet object in this homework, trained model.
        :param epoch (int): current epoch number.
        :param test_miou (float): miou of the test set.
        :return: None
    """
    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch,
             'model_miou': test_miou, }
    torch.save(state, chkpt_path)
    print("checkpoint saved at epoch", epoch)


def load_chkpt(model, chkpt_path, device):
    """
        Load model parameters from saved checkpoint.
        :param model (torch.nn.module object): miniUNet model to accept the saved parameters.
        :param chkpt_path (str): path of the checkpoint to be loaded.
        :return model (torch.nn.module object): miniUNet model with its parameters loaded from the checkpoint.
        :return epoch (int): epoch at which the checkpoint is saved.
        :return model_miou (float): miou of the test set at the checkpoint.
    """
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    model_miou = checkpoint['model_miou']
    print("epoch, model_miou:", epoch, model_miou)
    return model, epoch, model_miou


def save_prediction(model, dataloader, dump_dir, device, BATCH_SIZE):
    """
        For all datapoints d in dataloader, save  ground truth segmentation mask (as {id}.png)
          and predicted segmentation mask (as {id}_pred.png) in dump_dir.
        :param model (torch.nn.module object): trained miniUNet model
        :param dataloader (torch.utils.data.DataLoader object): dataloader to use for getting predictions
        :param dump_dir (str): dir path for dumping predictions
        :param device (torch.device object): pytorch cpu/gpu device object
        :param BATCH_SIZE (int): batch size of dataloader
        :return: None
    """
    print(f"Saving predictions in directory {dump_dir}")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    model.eval()
    with torch.no_grad():
        for batch_ID, sample_batched in enumerate(dataloader):
            data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            for i in range(pred.shape[0]):
                gt_image = convert_seg_split_into_color_image(target[i].cpu().numpy())
                pred_image = convert_seg_split_into_color_image(pred[i].cpu().numpy())
                combined_image = np.concatenate((gt_image, pred_image), axis=1)
                test_ID = batch_ID * BATCH_SIZE + i
                image.write_mask(combined_image, f"{dump_dir}/{test_ID}_gt_pred.png")


def iou(prediction, target):
    """
    This iou function is from hw2.

    In:
        prediction: Tensor [batchsize, class, height, width], predicted mask.
        target: Tensor [batchsize, height, width], ground truth mask.
    Out:
        batch_ious: a list of floats, storing IoU on each batch.
    Purpose:
        Compute IoU on each data and return as a list.
    """
    _, pred = torch.max(prediction, dim=1)
    batch_num = prediction.shape[0]
    class_num = prediction.shape[1]
    batch_ious = list()
    for batch_id in range(batch_num):
        class_ious = list()
        for class_id in range(1, class_num):  # class 0 is background
            mask_pred = (pred[batch_id] == class_id).int()
            mask_target = (target[batch_id] == class_id).int()
            if mask_target.sum() == 0:  # skip the occluded object
                continue
            intersection = (mask_pred * mask_target).sum()
            union = (mask_pred + mask_target).sum() - intersection
            class_ious.append(float(intersection) / float(union))
        batch_ious.append(np.mean(class_ious))
    return batch_ious


def train(model, train_loader, criterion, optimizer, epoch):
    """
        Loop over each sample in the dataloader. Do forward + backward + optimize procedure and print mean IoU on train set.
        :param model (torch.nn.module object): miniUNet model object
        :param train_loader (torch.utils.data.DataLoader object): train dataloader
        :param criterion (torch.nn.module object): Pytorch criterion object
        :param optimizer (torch.optim.Optimizer object): Pytorch optimizer object
        :param epoch (int): current epoch number
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    model.train()
    # TODO: complete this function
    # ===============================================================================
    train_loss_list, train_iou_list = [], []
    for batch in train_loader:
        inputs = batch['input']
        labels = batch['target']
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()

        preds = model(inputs)
        train_loss = criterion(preds, labels)
        train_loss_list.append(train_loss.item())
        train_iou = iou(preds, labels)
        train_iou_list.append(sum(train_iou) / len(train_iou))

        train_loss.backward()
        optimizer.step()

    mean_epoch_loss, mean_iou = sum(train_loss_list) / len(train_loss_list), sum(train_iou_list) / len(train_loss_list)
    print('[Epoch %d] Train loss & mIoU: %0.2f %0.2f' % (epoch, mean_epoch_loss, mean_iou))
    # ===============================================================================


def test(model, test_loader, criterion):
    """
        Similar to train(), but no need to backward and optimize.
        :param model (torch.nn.module object): miniUNet model object
        :param test_loader (torch.utils.data.DataLoader object): test dataloader
        :param criterion (torch.nn.module object): Pytorch criterion object
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    model.eval()
    # TODO: complete this function
    # ===============================================================================
    test_loss_list, test_iou_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input']
            labels = batch['target']
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            preds = model(inputs)
            test_loss = criterion(preds, labels)
            test_loss_list.append(test_loss.item())
            test_iou = iou(preds, labels)
            test_iou_list.append(sum(test_iou) / len(test_iou))

    mean_epoch_loss, mean_iou = sum(test_loss_list) / len(test_loss_list), sum(test_iou_list) / len(test_iou_list)
    print('Test loss & mIoU: %0.2f %0.2f' % (mean_epoch_loss, mean_iou))
    return mean_epoch_loss, mean_iou
    # ===============================================================================


def convert_seg_split_into_color_image(img):
    color_palette = get_tableau_palette()
    colored_mask = np.zeros((*img.shape, 3))

    print(np.unique(img))

    for i, unique_val in enumerate(np.unique(img)):
        if unique_val == 0:
            obj_color = np.array([0, 0, 0])
        else:
            obj_color = np.array(color_palette[i - 1]) * 255
        obj_pixel_indices = (img == unique_val)
        colored_mask[:, :, 0][obj_pixel_indices] = obj_color[0]
        colored_mask[:, :, 1][obj_pixel_indices] = obj_color[1]
        colored_mask[:, :, 2][obj_pixel_indices] = obj_color[2]
    return colored_mask.astype(np.uint8)


if __name__ == "__main__":
    # ==============Part 4 (a) Training Segmentation model ================
    # Complete all the TODO's in this file
    # - HINT: Most TODO's in this file are exactly the same as homework 2.

    seed(0)
    torch.manual_seed(0)

    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # TODO: Prepare train and test datasets
    # Load the "dataset" directory using RGBDataset class as a pytorch dataset
    # Split the above dataset into train and test dataset in 9:1 ratio using `torch.utils.data.random_split` method
    # ===============================================================================
    root_dir = './dataset/'
    dataset = RGBDataset(root_dir)
    train_ratio = 0.9
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # ===============================================================================

    # TODO: Prepare train and test Dataloaders. Use appropriate batch size
    # ===============================================================================
    train_loader = DataLoader(train_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset)
    # ===============================================================================

    # TODO: Prepare model
    # ===============================================================================
    model = miniUNet(n_channels=3, n_classes=4)
    model.to(device)
    # ===============================================================================

    # TODO: Define criterion, optimizer and learning rate scheduler
    # ===============================================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    # ===============================================================================

    # TODO: Train and test the model.
    # Tips:
    # - Remember to save your model with best mIoU on objects using save_chkpt function
    # - Try to achieve Test mIoU >= 0.9 (Note: the value of 0.9 only makes sense if you have sufficiently large test set)
    # - Visualize the performance of a trained model using save_prediction method. Make sure that the predicted segmentation mask is almost correct.
    # ===============================================================================
    epoch, max_epochs = 1, 5
    best_miou = float('-inf')
    while epoch <= max_epochs:
        print('Epoch (', epoch, '/', max_epochs, ')')
        train(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_miou = test(model, test_loader, criterion)
        print('---------------------------------')
        if test_miou > best_miou:
            best_miou = test_miou
            save_chkpt(model, epoch, test_miou, chkpt_path='checkpoint_multi.pth.tar')
        epoch += 1

    # Load the best checkpoint, use save_prediction() on the test set
    model, epoch, best_miou = load_chkpt(model, 'checkpoint_multi.pth.tar', device)
    save_prediction(model, test_loader, root_dir + 'test/', device, BATCH_SIZE=1)
    # ===============================================================================