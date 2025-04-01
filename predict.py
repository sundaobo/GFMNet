# coding=utf-8
import os
import torch.utils.data
from data_utils import TestDatasetFromFolder, calMetric_iou, calMetric_somemetric
import cv2
from tqdm import tqdm
import csv
import argparse
from torch.utils.data import DataLoader
import numpy as np
from model.network import GFMNet
from data_utils import calMetric_somemetric

parser = argparse.ArgumentParser(description='Test Change Detection Models')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--dataset_name', required=False,default='CLCD', type=str, help='model save path')
parser.add_argument('--pth_name', required=False,default='netCD_epoch_best0.1.pth', type=str, help='model save path')
args = parser.parse_args()
# parser.add_argument('--model_dir', default='/mnt/ssd/sdb/Datasets/结果/PGNET-FREQ.tar/PGNET-FREQ/CropLand-CD-new-vig-lossFREQ/epochs/CLCD/netCD_epoch_best98CNN.pth', type=str)
parser.add_argument('--model_dir', default='/mnt/ssd/sdb/Datasets/PGNET-FREQOur/GRSL/CropLand-CD-new-vig-lossFREQ/epochs/'+args.dataset_name+'/'+args.pth_name, type=str)
parser.add_argument('--hr1_dir', default='/mnt/ssd/sdb/Datasets/'+args.dataset_name+'/test/A', type=str)
parser.add_argument('--lr2_dir', default='/mnt/ssd/sdb/Datasets/'+args.dataset_name+'/test/B', type=str)
parser.add_argument('--label_dir', default='/mnt/ssd/sdb/Datasets/'+args.dataset_name+'/test/label', type=str)
parser.add_argument('--save_cd', default='/mnt/ssd/sdb/Datasets/PGNET-FREQOur/GRSL/CropLand-CD-new-vig-lossFREQ/result/'+args.dataset_name+'/', type=str)

# parser = argparse.ArgumentParser(description='Test Change Detection Models')
# parser.add_argument('--gpu_id', default="1", type=str, help='which gpu to run.')
# parser.add_argument('--model_dir', default='epochs/DSIFN-Dataset/netCD_epoch_best.pth', type=str)
# parser.add_argument('--hr1_dir', default='/root/autodl-tmp/Dataset/DSIFN-Dataset/test/A', type=str)
# parser.add_argument('--lr2_dir', default='/root/autodl-tmp/Dataset/DSIFN-Dataset/test/B', type=str)
# parser.add_argument('--label_dir', default='/root/autodl-tmp/Dataset/DSIFN-Dataset/test/label', type=str)
# parser.add_argument('--save_cd', default='/root/autodl-tmp/model/Cropland/DSIFN-Dataset/', type=str)

parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
if not os.path.exists(args.save_cd):
    os.mkdir(args.save_cd)

netCD = CDNet().to(device, dtype=torch.float)
print("Let's use", torch.cuda.device_count(), "GPUs!")

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     netCD = networks.BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8).to(device, dtype=torch.float)

netCD.load_state_dict(torch.load(args.model_dir))
netCD.eval()

if __name__ == '__main__':
    test_set = TestDatasetFromFolder(args, args.hr1_dir, args.lr2_dir, args.label_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=24, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader)

    inter, unin= 0,0
    test_results = {'batch_sizes': 0, 'IoU': 0,'tp':0,'fp':0,'fn':0,'tn':0,'cd_f1scores':0,'cd_precisions':0,'cd_recalls':0}#tp, fp, fn
    for hr_img1, lr_img2, label, image_name in test_bar:

        hr_img1 = hr_img1.to(device, dtype=torch.float)
        lr_img2 = lr_img2.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.float)

        prob,_ = netCD(hr_img1, lr_img2)

        prob=prob.squeeze(0)
        prob = torch.argmax(prob, 0)
        prob = prob.cpu().data.numpy()
        result = np.squeeze(prob)

        label = label.squeeze(0)
        label = torch.argmax(label, 0)
        gt_value = label.cpu().detach().numpy()
        gt_value = np.squeeze(gt_value)

        intr, unn = calMetric_iou(gt_value, result)
        inter = inter + intr
        unin = unin + unn

        test_bar.set_description(
            desc='IoU: %.4f' % (inter * 1.0 / unin))
        #
        cv2.imwrite(args.save_cd + image_name[0], result*255)
        tp,tn,fp,fn = calMetric_somemetric(gt_value, result)#
        intr, unn = calMetric_iou(gt_value, result)
        inter = inter + intr
        unin = unin + unn
        #pr = calMetric_pr(gt_value, result)

        test_results['IoU'] = (inter * 1.0 / unin)
        test_results['tp'] += tp
        test_results['fp'] += fp
        test_results['fn'] += fn
        test_results['tn'] += tn
        with open("CLCD.csv", "a", newline="") as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow(["Filename", "IOU", "F1" ])
          f1temp = 2 * (tp / (tp + fp)) * (tp / (tp + fn)) / (tp / (tp + fn) + tp / (tp + fp))
          writer.writerow([image_name, (inter * 1.0 / unin), f1temp])
        
        #test_results['pr'] = (pr * 1.0 / unin)
    print(test_results)
    fp, fn, tp,tn =  test_results['fp'], test_results['fn'], test_results['tp'],test_results['tn'] #tn
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (R + P)
    IOU = tp / (fn + tp + fp)

    ttt_test = tn + fp + fn + tp
    TA_test = (tp + tn) / ttt_test
    Pcp1_test = (tp + fn) / ttt_test
    Pcp2_test = (tp + fp) / ttt_test
    Pcn1_test = (fp + tn) / ttt_test
    Pcn2_test = (fn + tn) / ttt_test
    Pc_test = Pcp1_test * Pcp2_test + Pcn1_test * Pcn2_test
    kappa_test = (TA_test - Pc_test) / (1 - Pc_test)

    test_results['cd_f1scores'] = F1
    test_results['cd_precisions'] = P
    test_results['cd_recalls'] = R
    print("TEST Result"+ str(test_results))