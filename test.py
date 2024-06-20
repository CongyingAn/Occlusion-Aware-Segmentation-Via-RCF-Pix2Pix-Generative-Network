"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os

import cv2

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from models import Rcfmodels
import torch
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_curve




try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def clear_image(clear_img, hazy_img):
    """
    该函数用于读取图像并计算去雾前后图像的PSNR与SSIM值。
    :param clear_img_path: 清晰图像路径
    :param hazy_img_path: 待去雾图像路径
    :return: None
    """
    # 读取清晰图像和待去雾图像
    # clear_img = cv2.imread(clear_img_path)
    # hazy_img = cv2.imread(hazy_img_path)

    # 检查图像尺寸是否一致，若不一致则进行缩放
    if clear_img.shape[0] != hazy_img.shape[0] or clear_img.shape[1] != hazy_img.shape[1]:
        pil_img = Image.fromarray(hazy_img)
        pil_img = pil_img.resize((clear_img.shape[1], clear_img.shape[0]))
        hazy_img = np.array(pil_img)



    # 计算PSNR和SSIM值，PSNR越大表示图像质量越好，SSIM越大表示两图像越相似
    PSNR = peak_signal_noise_ratio(clear_img, hazy_img)
    SSIM = structural_similarity(clear_img, hazy_img, channel_axis=2)
    return PSNR, SSIM


def compute_ap_ar(predictions, labels):
    num_classes = np.max(labels) + 1  # 类别的数量

    ap_sum = 0  # 每个类别的AP之和
    ar_sum = 0  # 每个类别的AR之和

    for i in range(num_classes):
        true_positives = np.sum((predictions == i) & (labels == i))  # 预测为正类且标签为正类的数量
        false_positives = np.sum((predictions == i) & (labels != i))  # 预测为正类但标签为负类的数量
        false_negatives = np.sum((predictions != i) & (labels == i))  # 预测为负类但标签为正类的数量

        precision = true_positives / (true_positives + false_positives + 1e-6)  # 计算精确度时加上一个很小的数以避免除以0
        recall = true_positives / (true_positives + false_negatives + 1e-6)  # 计算召回率时加上一个很小的数以避免除以0

        ap_sum += precision  # 累加类别的精确度
        ar_sum += recall  # 累加类别的召回率

    ap = ap_sum / num_classes  # 平均精确度
    ar = ar_sum / num_classes  # 平均召回率

    return ap, ar


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # opt.rcfmodelpath = 'avocado.pth'
    # model_rcf = Rcfmodels.RCF().cuda()
    # model_rcf.eval()
    # checkpoint = torch.load(opt.rcfmodelpath, map_location='cuda')
    # model_rcf.load_state_dict(checkpoint['state_dict'])
    #
    # opt.newmodel = model_rcf


    # 自己修改部分
    # opt.dataset_mode="aligned"
    # opt.num_threads = 4
    # opt.serial_batches=False
    # opt.model="pix2pix"

    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers



    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    PSNR_array = []
    SSIM_array = []
    AP_array=[]
    AR_array = []
    iou_list=[]



    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        img_psng=save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    #     fake_l=img_psng[1]
    #     real_l = img_psng[2]
    #     fake_l=fake_l.astype(np.uint8)
    #     real_l = real_l.astype(np.uint8)
    #     fake_l = np.expand_dims(fake_l, axis=2)  # 形状变为 (512, 512, 1)
    #     real_l = np.expand_dims(real_l, axis=2)  # 形状变为 (512, 512, 1)
    #
    #
    #     PSNR, SSIM=clear_image(real_l,fake_l)
    #     PSNR_array.append(PSNR)
    #     SSIM_array.append(SSIM)
    #
    #
    #
    #     # 将预测结果和真实标签展平为一维数组
    #     fake_th = np.where(img_psng[1] > 100, 1, 0)
    #     real_th = np.where(img_psng[2] > 100, 1, 0)
    #     labels_flat = real_th.flatten()
    #     predictions_flat= fake_th.flatten()
    #     # 计算精确率和召回率曲线
    #     precision, recall, _ = precision_recall_curve(labels_flat, predictions_flat)
    #
    #     ap, ar = compute_ap_ar(fake_th, real_th)
    #
    #     # 计算平均精确率（Average Precision）
    #     # ap = np.mean(precision)
    #     AP_array.append(ap)
    #     # 计算平均召回率（Average Recall）
    #     # ar = np.mean(recall)
    #     AR_array.append(ar)
    #
    #     intersection = np.logical_and(real_th, fake_th)
    #     union = np.logical_or(real_th, fake_th)
    #     iou = np.sum(intersection) / np.sum(union)
    #     iou_list.append(iou)
    #
    #
    # mean_PSNR = np.mean(PSNR_array)
    # print("M-PSNR：", mean_PSNR)
    # mean_SSIM = np.mean(SSIM_array)
    # print("M-SSIM：", mean_SSIM)
    # mean_AP = np.mean(AP_array)
    # print("M-AP：", mean_AP)
    # mean_AR = np.mean(AP_array)
    # print("M-AR：", mean_AR)
    # mean_IOU = np.mean(iou_list)
    # print("M-IOU：", mean_IOU)


    webpage.save()  # save the HTML
