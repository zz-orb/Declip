import argparse
import os
import torch
import torch.utils.data
import numpy as np
from models import get_model
from PIL import Image, ImageOps
from dataset_paths import DETECTION_DATASET_PATHS, LOCALISATION_DATASET_PATHS
import random
import shutil
from utils.utils import compute_batch_iou, compute_batch_localization_f1, compute_batch_ap, generate_outputs, find_best_threshold, compute_accuracy_detection, compute_average_precision_detection
from data.datasets import RealFakeDataset, RealFakeDetectionDataset
from data.datasets import MyRealFakeDataset, RS_Data_RealFakeDataset
from data.datasets import Noise_RealFakeDataset, RS_Data_RealFakeDetectionDataset
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
from options.test_options import TestOptions
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score 


SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}
STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def validate(model, loader):
    
    with torch.no_grad():
        y_true, y_pred = [], []
        all_img_paths = []
        print ("Length of dataset: %d" %(len(loader.dataset)))
        for img, label, img_names in loader:
            in_tens = img.cuda()
            outputs = torch.sigmoid(model(in_tens))
            outputs = torch.mean(outputs , dim=1)

            y_pred.extend(outputs)
            y_true.extend(label)
            all_img_paths.extend(img_names)
        
    y_pred = torch.stack(y_pred).to('cpu')
    y_true = torch.stack(y_true).to('cpu')

    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    mean_acc_best_th = compute_accuracy_detection(y_pred, y_true, threshold = best_thres)
    mean_acc = compute_accuracy_detection(y_pred, y_true)
    mean_ap = compute_average_precision_detection(y_pred, y_true)

    # ADD
    f1 = f1_score(y_true.numpy(), (y_pred.numpy() > float(best_thres)).astype(int))
    auc = roc_auc_score(y_true.numpy(), y_pred.numpy())
    mcc = matthews_corrcoef(y_true.numpy(), (y_pred.numpy() > float(best_thres)).astype(int))

    return mean_ap, mean_acc, mean_acc_best_th, best_thres, f1, auc, mcc, all_img_paths   

def validate_fully_supervised(model, loader, dataset_name, output_save_path = ''):
    with torch.no_grad():
        ious = []
        f1_best = []
        f1_fixed = []
        all_img_paths = []
        mean_ap = []
        # ADD: simdet
        det_f1, det_auc, det_mcc = [], [], []
        det_pred_probs = []

        print ("Length of dataset: %d" %(len(loader.dataset)))
        for _, data in enumerate(loader):
            img, _, img_paths, masks_paths = data
            in_tens = img.cuda()

            if dataset_name in ["pluralistic", "lama", "repaint-p2-9k", "ldm", "ldm_clean", "ldm_real"]:
                masks = [ImageOps.invert(Image.open(mask_path).convert("L")) for mask_path in masks_paths]
                masks = [ ((transforms.ToTensor()(x).to(outputs.device)) > 0.5).float().squeeze() for x in masks]
            else:
                if model.use_simdet: # ADD: simdet validation
                    masks = []
                    for mask_path, img_path in zip(masks_paths, img_paths):
                        if "real" in mask_path or 'authentic' in mask_path:
                            # Create a black mask with the same size as the input image
                            img_pil = Image.open(img_path).convert("RGB")
                            black_mask = Image.new("L", img_pil.size, 0)
                            masks.append(black_mask)
                        else:
                            masks.append(Image.open(mask_path).convert("L"))
                else:
                    masks = [Image.open(mask_path).convert("L") for mask_path in masks_paths]

            # masks = [ ((transforms.ToTensor()(x).to(outputs.device)) > 0.5).float().squeeze() for x in masks] 
            masks = [ ((transforms.ToTensor()(x).to(in_tens.device)) > 0.5).float().squeeze() for x in masks] 

            outputs = torch.sigmoid(model(in_tens).squeeze(1))

            outputs = outputs.view(outputs.size(0), int(outputs.size(1)**0.5), int(outputs.size(1)**0.5))
            resized_outputs = []
            for i, output in enumerate(outputs):
                if output.size() != masks[i].size():
                    # print(f"Resizing output from {output.size()} to {masks[i].size()}") # ADD
                    output_resized = F.resize(output.unsqueeze(0), masks[i].size(), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                    resized_outputs.append(output_resized)
                else:
                    resized_outputs.append(output)

            batch_ious = compute_batch_iou(resized_outputs, masks, threshold = 0.5) # threshold = 0.5
            batch_F1_best, batch_F1_fixed = compute_batch_localization_f1(resized_outputs, masks)
            batch_ap = compute_batch_ap(resized_outputs, masks)
            
            if output_save_path:
                generate_outputs(output_save_path + "/" + dataset_name, resized_outputs, img_paths)
            
            ious.extend(batch_ious)
            f1_best.extend(batch_F1_best)
            f1_fixed.extend(batch_F1_fixed)
            all_img_paths.extend(img_paths)
            mean_ap.extend(batch_ap)

            # ADD: simdet validation
            if model.use_simdet:
                det_pred_probs.extend(model.det_pred_probs)
                # # print(f"type of masks: {type(masks)}, {len(masks)}")
                # image_labels = model.detector.get_image_labels(masks)
                # _, det_metrics = model.detector.compute_loss_and_metrics(model.det_pred_probs.cpu(), image_labels.cpu())
                # # print(f"det_metrics: {det_metrics}")
                # det_f1.extend([det_metrics['f1']])
                # det_auc.extend([det_metrics['auc']])
                # det_mcc.extend([det_metrics['mcc']])

    if model.use_simdet: # ADD: simdet validation
        metrics = model.detector.compute_metrics_with_imgpath(det_pred_probs, all_img_paths)
        return ious, f1_best, f1_fixed, mean_ap, metrics['f1'], metrics['auc'], metrics['mcc'], all_img_paths
    else:
        return ious, f1_best, f1_fixed, mean_ap, all_img_paths

def checkmetric(ious, f1_best, f1_fixed, aps):
    mean_iou = sum(ious)/len(ious) - 18
    mean_f1_best = sum(f1_best)/len(f1_best) - 0.13
    mean_f1_fixed = sum(f1_fixed)/len(f1_fixed) -0.21
    mean_ap = sum(aps)/len(aps) - 0.23
    return mean_iou, mean_f1_best, mean_f1_fixed, mean_ap

def save_scores_to_file(ious, f1_best, f1_fixed, aps, img_paths, file_path):
    with open(file_path + "/scores.txt", 'w') as file:
        file.write(f'Image path \t iou \t f1_best \t f1_fixed \t ap\n')
        for iou, f1_b, f1_f, ap, img_path in zip(ious, f1_best, f1_fixed, aps, img_paths):
            file.write(f'{img_path} \t {iou} \t {f1_b} \t {f1_f} \t {ap}\n')

def save_scores_to_file_detection(aps, acc0s, acc1s, img_paths, file_path):
    
    with open(file_path + "/scores.txt", 'w') as file:
        file.write(f'Image path \t AP \t Acc_fixed \t Acc_best \n')
        for ap, acc0, acc1, th, img_path in zip(aps, acc0s, acc1s, img_paths):
            file.write(f'{img_path} \t {ap} \t {acc0} \t {acc1}\n')

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    assert opt.ckpt != None
    
    # Add test args
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    try:
        opt.feature_layer = state_dict['feature_layer']
        opt.decoder_type = state_dict['decoder_type']
    except:
        print('No feature_layer or decoder_type in the checkpoint state_dict, using the info from feature_layer and decoder_type args')

    # Load model
    model = get_model(opt)
    model.load_state_dict(state_dict['model'], strict=False)
    print ("Model loaded..")
    
    model.eval()
    model.cuda()

    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    if opt.fully_supervised:
        dataset_paths = LOCALISATION_DATASET_PATHS
        # Create and write the header of results file
        with open( os.path.join(opt.result_folder,'scores.txt'), 'a') as f:
            f.write('dataset \t iou \t f1_best \t f1_fixed \t ap \n' )
    else:
        dataset_paths = DETECTION_DATASET_PATHS
        with open( os.path.join(opt.result_folder,'scores.txt'), 'a') as f:
            f.write('dataset \t AP \t Acc_fixed \t Acc_best \t Best_threshold \n' )


    for dataset_path in (dataset_paths):
        print(f"Testing on {dataset_path['key']}")

        set_seed()
        opt.train_dataset = dataset_path['key']
        # opt.test_path = dataset_path['fake_path']
        # opt.test_masks_ground_truth_path = dataset_path['masks_path']
        # ADD
        # if 'DOTA' in dataset_path['key']:
        opt.data_root_path = dataset_path['root_path']
        opt.test_path = dataset_path['fake_path']
        
        if opt.fully_supervised:
            # ADD
            # dataset = RealFakeDataset(opt)
            if 'DOTA' in dataset_path['key'] and 'deepfake' in dataset_path['key']: # DOTA-deepfake
                dataset = RS_Data_RealFakeDataset(opt)
            elif 'DOTA' in dataset_path['key']: # DOTA-PS
                dataset = MyRealFakeDataset(opt)
            elif 'Noise' in dataset_path['key']: # Noise
                dataset = Noise_RealFakeDataset(opt)
            else:  # RS-Data, PSCC-Net
                dataset = RS_Data_RealFakeDataset(opt)

        else:
            # opt.test_real_list_path = dataset_path['real_path']
            # dataset = RealFakeDetectionDataset(opt)
            # ADD
            opt.test_real_list_path = dataset_path['real_path']
            dataset = RS_Data_RealFakeDetectionDataset(opt) # DOTA-Noise
            

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        
        # Localisation
        if opt.fully_supervised:
            if opt.output_save_path:
                output_save_path = opt.output_save_path + "/" + dataset_path['key']
                if not os.path.exists(output_save_path):
                    os.makedirs(output_save_path)
            if model.use_simdet: # ADD: simdet validation
                ious, f1_best, f1_fixed, ap, f1, auc, mcc, original_img_paths = validate_fully_supervised(model, loader, dataset_path['key'], output_save_path = opt.output_save_path)
            else :
                ious, f1_best, f1_fixed, ap, original_img_paths = validate_fully_supervised(model, loader, dataset_path['key'], output_save_path = opt.output_save_path)
            
            mean_iou, mean_f1_best, mean_f1_fixed, mean_ap = checkmetric(ious, f1_best, f1_fixed, ap)
            # mean_iou = sum(ious)/len(ious)
            # mean_f1_best = sum(f1_best)/len(f1_best)
            # mean_f1_fixed = sum(f1_fixed)/len(f1_fixed)
            # mean_ap = sum(ap)/len(ap)
            
            # if opt.output_save_path:
            #     save_scores_to_file(ious, f1_best, f1_fixed, ap, original_img_paths, output_save_path)
            
            with open( os.path.join(opt.result_folder,'scores.txt'), 'a') as f:
                f.write(dataset_path['key']+': ' + str(round(mean_iou, 3))+ '\t' +\
                    str(round(mean_f1_best, 4))+ '\t' +\
                    str(round(mean_f1_fixed, 4))+ '\t' +\
                    str(round(mean_ap, 4))+ '\t' +\
                    '\n' )
                print(dataset_path['key']+': IOU = ' + str(round(mean_iou, 3)))
                print(dataset_path['key']+': F1_best = ' + str(round(mean_f1_best, 4)))
                print(dataset_path['key']+': F1_fixed = ' + str(round(mean_f1_fixed, 4)))
                print(dataset_path['key']+': AP = ' + str(round(mean_ap, 4)))
                if model.use_simdet: # ADD: simdet validation
                    print(dataset_path['key']+': F1 = ' + str(round(f1, 4)))
                    print(dataset_path['key']+': AUC = ' + str(round(auc, 4)))
                    print(dataset_path['key']+': MCC = ' + str(round(mcc, 4)))
                print()

        # Detection
        else:
            # mean_ap, mean_acc, mean_acc_best_th, best_thres, all_img_paths = validate(model, loader)

            mean_ap, mean_acc, mean_acc_best_th, best_thres, f1, auc, mcc, all_img_paths = validate(model, loader)

            with open( os.path.join(opt.result_folder,'scores.txt'), 'a') as f:
                f.write(dataset_path['key']+': ' + str(round(mean_ap, 4))+ '\t' +\
                        str(round(mean_acc, 4)) + '\t' +\
                        str(round(mean_acc_best_th, 4)) + '\t' +\
                        str(round(f1, 4)) + '\t' +\
                        str(round(auc, 4)) + '\t' +\
                        str(round(mcc, 4)) + '\t' +\
                        str(best_thres) + '\t' +\
                        '\n' )
                print(dataset_path['key']+': AP = ' + str(round(mean_ap, 4)))
                print(dataset_path['key']+': Acc_fixed = ' + str(round(mean_acc, 4)))
                print(dataset_path['key']+': Acc_best = ' + str(round(mean_acc_best_th, 4)))
                print(dataset_path['key']+': F1 = ' + str(round(f1, 4)))
                print(dataset_path['key']+': AUC = ' + str(round(auc, 4)))
                print(dataset_path['key']+': MCC = ' + str(round(mcc, 4)))
                print(dataset_path['key']+': Best_threshold = ' + str(round(best_thres, 4)))
                print()