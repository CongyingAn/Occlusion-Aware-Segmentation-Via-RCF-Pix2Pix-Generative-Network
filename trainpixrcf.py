"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import csv
import time

import cv2


from options.train_options import TrainOptions
from options.test_options import TestOptions
from models import Rcfmodels
from data import create_dataset
from models import create_model
from util import util
from util.visualizer import Visualizer
import numpy as np
from models.tool_iou import boundary_iou
from PIL import Image
import copy
import torch
from sklearn.metrics import precision_recall_curve, auc

def calculate_iou(image1, image2):
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def calculate_ap(result_img, label_img):
    # 将结果图和标签图转换为一维数组
    result = result_img.flatten()
    label = label_img.flatten()

    # 根据结果图的值进行排序
    sorted_indices = np.argsort(result)[::-1]  # 降序排列的索引

    # 初始化变量
    true_positives = 0  # 正确预测的正样本数
    false_positives = 0  # 错误预测的正样本数
    total_positives = np.sum(label)  # 标签图中正样本的总数
    precision = []  # 存储每个阈值下的精确度
    recall = []  # 存储每个阈值下的召回率

    # 遍历排序后的结果数组
    for i in range(len(result)):
        index = sorted_indices[i]
        if label[index] == 1:
            true_positives += 1
        else:
            false_positives += 1
        # 计算当前阈值下的精确度和召回率
        current_precision = true_positives / (
                    true_positives + false_positives) if true_positives + false_positives != 0 else 0
        current_recall = true_positives / total_positives
        precision.append(current_precision)
        recall.append(current_recall)

    # 计算平均精度（AP）
    ap = 0
    previous_recall = 0
    for i in range(len(precision)):
        if recall[i] != previous_recall:
            ap += precision[i] * (recall[i] - previous_recall)
            previous_recall = recall[i]

    return ap


def calculate_ap_torch(result_img, label_img):
    # 将 numpy 数组转换为 Torch 张量
    result_tensor = torch.from_numpy(result_img)
    label_tensor = torch.from_numpy(label_img)

    # 计算 Precision 和 Recall
    true_positives = torch.sum(torch.logical_and(result_tensor == 1, label_tensor == 1)).item()
    false_positives = torch.sum(torch.logical_and(result_tensor == 1, label_tensor == 0)).item()
    false_negatives = torch.sum(torch.logical_and(result_tensor == 0, label_tensor == 1)).item()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # 计算 Precision-Recall 曲线上的点
    precision, recall, _ = precision_recall_curve(label_tensor.flatten(), result_tensor.flatten())

    # 计算 AP
    ap = auc(recall, precision)

    return ap

# 范围检查函数，用于确定是否在目标颜色范围内
def extract_binary_mask_v2(array, color, variance=30):
    # 基于向量化操作，快速检查像素值是否属于指定颜色范围
    lower_bound = np.array(color) - variance
    upper_bound = np.array(color) + variance
    mask = np.all((array >= lower_bound) & (array <= upper_bound), axis=-1).astype(int)
    return mask


def calculate_miou_for_class_v2(real_B_array, fake_B_array, color, variance=30):
    iou_list = []
    for real_img, fake_img in zip(real_B_array, fake_B_array):
        A_binary = extract_binary_mask_v2(real_img, color, variance)
        B_binary = extract_binary_mask_v2(fake_img, color, variance)
        intersection = np.logical_and(A_binary, B_binary).sum()
        union = np.logical_or(A_binary, B_binary).sum()
        if union == 0:
            iou = float('nan')  # 或者选择一个策略来处理这种情况，比如设置iou为0或者跳过这个样本
        else:
            iou = intersection / union
        iou_list.append(iou)

    # 移除NaN值后计算平均值
    iou_list = [iou for iou in iou_list if not np.isnan(iou)]
    if len(iou_list) == 0:  # 如果全部是NaN值，则返回NaN
        return float('nan')
    return np.mean(iou_list)




if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options
    # Rcf边界
    opt.rcfmodelpath = 'checkpoint_epoch10.pth'
    model_rcf = Rcfmodels.RCF().cuda()
    model_rcf.eval()
    checkpoint = torch.load(opt.rcfmodelpath, map_location='cuda')
    model_rcf.load_state_dict(checkpoint['state_dict'])

    opt.newmodel = model_rcf

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # 计算IOU
    optiou = TestOptions().parse()  # get test options
    optiou.newmodel = model_rcf
    datasetiou = create_dataset(optiou)  # create a dataset given opt.dataset_mode and other options

    # 写入数据到 CSV 文件
    filename = 'iou.csv'

    best_iou=0
    best_iouB = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        # 测试iou
        if epoch % 1 == 0:
            testmodel= model
            testmodel.eval()
            real_B_array = []
            fake_B_array = []
            for i, data in enumerate(datasetiou):
                testmodel.set_input(data)  # unpack data from data loader
                testmodel.test()  # run inference
                visuals = testmodel.get_current_visuals()  # get image results

                for label, im_data in visuals.items():
                    im = util.tensor2im(im_data)
                    if label == "real_B":
                        real_B_array.append(im)
                    elif label == "fake_B":
                        fake_B_array.append(im)

            iou_list = []
            boundaryiou_list=[]
            boundaryiou_list2 = []
            ap_list=[]

            # d单一灰度图计算方法
            for i in range(len(real_B_array)):
                A_binary = (real_B_array[i].sum(axis=2) > 0).astype(int)
                B_binary = (fake_B_array[i].sum(axis=2) > 100).astype(int)
                intersection = np.logical_and(A_binary, B_binary)
                union = np.logical_or(A_binary, B_binary)
                iou = np.sum(intersection) / np.sum(union)
                iou_list.append(iou)
                # ap=calculate_ap_torch(B_binary, A_binary)
                # ap_list.append(ap)
                # boundary_iou()
                gray_imageG = cv2.cvtColor(real_B_array[i], cv2.COLOR_BGR2GRAY) if len(real_B_array[i].shape) == 3 else real_B_array[i]
                gray_imageF = cv2.cvtColor(fake_B_array[i], cv2.COLOR_BGR2GRAY) if len(fake_B_array[i].shape) == 3 else fake_B_array[i]
                # 对 np 数组进行二值化处理
                threshold_value = 100
                # binary_imageF = np.where(gray_imageF > threshold_value, 255, 0)
                binary_imageF = (gray_imageF > threshold_value) * 255
                testlist=boundary_iou(gray_imageG,binary_imageF,0)
                boundaryiou_list.append(testlist[0])
                boundaryiou_list2.append(testlist[1])
                # im = Image.fromarray(binary_imageF.astype('uint8'), mode='L')
                # # # 保存图像
                # im.save('image.bmp')
            mean_iou = np.mean(iou_list)
            print("MIou：", mean_iou)
            # mean_ap = np.mean(ap_list)
            # print("MAP：", mean_ap)
            mean_boundaryiou=np.mean(boundaryiou_list)
            print("boundaryiou：", mean_boundaryiou)
            mean_boundaryiou2=np.mean(boundaryiou_list2)
            print("boundaryiou：", mean_boundaryiou2)



            # 类别颜色定义，紫色、红色、橙色、黄色
            colors = [
                (128, 0, 128),
                (0, 0, 255),
                (0, 165, 255),
                (0, 255, 255)
            ]

            # 假设 real_B_array 和 fake_B_array 已经是你的输入图像数组
            iou_scores = []
            for color in colors:
                iou = calculate_miou_for_class_v2(real_B_array, fake_B_array, color)
                iou_scores.append(iou)
                print(f"miou for color {color}: {iou}")

            # 计算总体miou
            mean_iou = np.mean(iou_scores)
            print(f"Total miou: {mean_iou}")



            testmodel.train()



            # 写入数据
            with open(filename, 'a', newline='') as file:  # 使用 "a" 模式来追加数据
                writer = csv.writer(file)

                # 如果是第一轮epoch，则写入表头
                if epoch == 1 and file.tell() == 0:  # 检查文件是否为空
                    writer.writerow(['Epoch', 'Iou', 'MAP', 'BIOU'])
             # 写入数据行
                writer.writerow([epoch, mean_iou, mean_boundaryiou, mean_boundaryiou2])

            if mean_iou > best_iou:
                best_iou = mean_iou
                model.save_networks('best')
                model.save_networks(epoch)

            if mean_boundaryiou2 > best_iouB:
                best_iouB = mean_boundaryiou2
                model.save_networks('best_Biou')
                model.save_networks(epoch)
        #     结束IOU测试

        # if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)



        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    with open(filename, 'a', newline='') as file:  # 使用 "a" 模式来追加数据
        writer = csv.writer(file)
        writer.writerow(['Iou-best', 'BIOU-best'])
        writer.writerow([best_iou, best_iouB])