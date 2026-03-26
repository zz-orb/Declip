from copy import deepcopy
import os
import time
from tensorboardX import SummaryWriter

from validate import validate, validate_fully_supervised
from utils.utils import compute_accuracy_detection, compute_average_precision_detection
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
from utils.utils import derive_datapaths
import torch.multiprocessing

import warnings
warnings.filterwarnings("ignore", 
                        category=UserWarning, 
                        module="torchmetrics")

def get_val_opt(opt):
    val_opt = deepcopy(opt)
    val_opt.data_label = 'valid'
    return val_opt


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    opt = TrainOptions().parse()
    # if opt.data_root_path:
    #     opt = derive_datapaths(opt)
    val_opt = get_val_opt(opt)
    
    model = Trainer(opt)

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    best_iou = best_f1 = best_acc = best_ap = best_mcc = 0
    print ("Length of training dataset: %d" %(len(data_loader.dataset)))
    # exit(0)
    for epoch in range(opt.niter):
        print(f"Epoch {epoch}")
        epoch_loss = 0    
        for i, data in enumerate(data_loader):
            # try:
            model.total_steps += 1
            torch.cuda.empty_cache()
            model.set_input(data)
            model.optimize_parameters(epoch)
            torch.cuda.empty_cache()
            if model.total_steps % opt.loss_freq == 0:
                print(f"Train loss: {round(model.loss.item(), 4)} at step: {model.total_steps};\t Iter time: {round((time.time() - start_time) / model.total_steps, 4)}")
                epoch_loss += model.loss
                train_writer.add_scalar('loss', model.loss, model.total_steps)
            # except Exception as e:
            #     print(f"Error at step {model.total_steps}: data: {data}")
            #     print(f"Exception: {e}")
            #     exit(0)
                

            # if i > 10:
            #     print(i)
            #     break

        model.finalize_optimizer_step()
            

        epoch_loss /= len(data_loader)
        if opt.fully_supervised:
            # compute train performance  
            mean_iou = sum(model.ious)/len(model.ious)
            model.ious = []
            print(f"Epoch mean train IOU: {round(mean_iou, 2)}")
            
            mean_F1_best = sum(model.F1_best)/len(model.F1_best)
            model.F1_best = []
            print(f"Epoch mean train F1_best: {round(mean_F1_best, 4)}")
            mean_F1_fixed = sum(model.F1_fixed)/len(model.F1_fixed)
            model.F1_fixed = []
            print(f"Epoch mean train F1_fixed: {round(mean_F1_fixed, 4)}")
            
            mean_ap = sum(model.ap)/len(model.ap)
            model.ap = []
            print(f"Epoch mean train Mean AP: {round(mean_ap, 4)}")

            # ADD
            if opt.use_simdet:
                print("SimDet train metrics!")
                print(f"Epoch mean train F1: {round(sum(model.det_f1)/len(model.det_f1), 4)}")
                print(f"Epoch mean train AUC: {round(sum(model.det_auc)/len(model.det_auc), 4)}")
                print(f"Epoch mean train MCC: {round(sum(model.det_mcc)/len(model.det_mcc), 4)}")
                model.det_f1 = []
                model.det_auc = []
                model.det_mcc = []

        else:
            model.format_output()
            mean_acc = compute_accuracy_detection(model.logits, model.labels)
            print(f"Epoch mean train ACC: {round(mean_acc, 2)}")
            mean_ap = compute_average_precision_detection(model.logits, model.labels)
            print(f"Epoch mean train AP: {round(mean_ap, 4)}")
        
            model.logits = []
            model.labels = []

        # continue

        # Validation
        model.eval()
        print('Validation')
        torch.cuda.empty_cache()
        if opt.fully_supervised:
            model.model.set_label(None)  # reset label for validation
            # ious, f1_best, f1_fixed, mean_ap, _ = validate_fully_supervised(model.model, val_loader, opt.train_dataset)

            # ADD: SimDet validation
            results = validate_fully_supervised(model.model, val_loader, opt.train_dataset)
            if opt.use_simdet:
                ious, f1_best, f1_fixed, mean_ap, det_f1, det_auc, det_mcc, _ = results
            else:
                ious, f1_best, f1_fixed, mean_ap, _ = results

            mean_iou = sum(ious)/len(ious)
            val_writer.add_scalar('iou', mean_iou, model.total_steps)
            print(f"(Val @ epoch {epoch}) IOU: {round(mean_iou, 2)}")
            
            mean_f1_best = sum(f1_best)/len(f1_best)
            val_writer.add_scalar('F1_best', mean_f1_best, model.total_steps)
            print(f"(Val @ epoch {epoch}) F1 best: {round(mean_f1_best, 4)}")
            
            mean_f1_fixed = sum(f1_fixed)/len(f1_fixed)
            val_writer.add_scalar('F1_fixed', mean_f1_fixed, model.total_steps)
            print(f"(Val @ epoch {epoch}) F1 fixed: {round(mean_f1_fixed, 4)}")
            
            mean_ap = sum(mean_ap)/len(mean_ap)
            val_writer.add_scalar('Mean AP', mean_ap, model.total_steps)
            print(f"(Val @ epoch {epoch}) Mean AP: {round(mean_ap, 4)}")

            # ADD: SimDet validation
            if opt.use_simdet:
                val_writer.add_scalar('Det F1', det_f1, model.total_steps)
                print(f"(Val @ epoch {epoch}) Det F1: {round(det_f1, 4)}")

                val_writer.add_scalar('Det AUC', det_auc, model.total_steps)
                print(f"(Val @ epoch {epoch}) Det AUC: {round(det_auc, 4)}")

                val_writer.add_scalar('Det MCC', det_mcc, model.total_steps)
                print(f"(Val @ epoch {epoch}) Det MCC: {round(det_mcc, 4)}")
            
            # save best model weights or those at save_epoch_freq 
            # if mean_iou > best_iou:
            #     print('saving best model at the end of epoch %d' % (epoch))
            #     model.save_networks( 'model_epoch_best.pth' )
            #     best_iou = mean_iou
            # ADD
            if mean_f1_best > best_f1 or mean_iou > best_iou:
                print('saving best model at the end of epoch %d' % (epoch))
                model.save_networks( f'model_epoch_best.pth' )  # _{epoch}
                best_f1 = mean_f1_best if mean_f1_best > best_f1 else best_f1
                best_iou = mean_iou if mean_iou > best_iou else best_iou

            early_stopping(mean_f1_best, model)
            # early_stopping(mean_iou, model)

        else:
            # ap, r_acc, f_acc, acc, = validate(model.model, val_loader)
            # val_writer.add_scalar('accuracy', acc, model.total_steps)
            # val_writer.add_scalar('ap', ap, model.total_steps)
            # print(f"(Val @ epoch {epoch}) ACC: {acc}; AP: {ap}")

            # ADD
            mean_ap, mean_acc, mean_acc_best_th, best_thres, f1, auc, mcc, all_img_paths = validate(model.model, val_loader)
            val_writer.add_scalar('accuracy', mean_acc, model.total_steps)
            val_writer.add_scalar('ap', mean_ap, model.total_steps)
            val_writer.add_scalar('f1', f1, model.total_steps)
            val_writer.add_scalar('auc', auc, model.total_steps)
            val_writer.add_scalar('mcc', mcc, model.total_steps)
            print(f"(Val @ epoch {epoch}) ACC: {round(mean_acc, 4)}")
            print(f"(Val @ epoch {epoch}) AP: {round(mean_ap, 4)}")
            print(f"(Val @ epoch {epoch}) F1: {round(f1, 4)}")
            print(f"(Val @ epoch {epoch}) AUC: {round(auc, 4)}")
            print(f"(Val @ epoch {epoch}) MCC: {round(mcc, 4)}")
            print(f"(Val @ epoch {epoch}) Best threshold: {round(float(best_thres), 4)}")
            acc = mean_acc

            # save best model weights or those at save_epoch_freq 
            if mean_acc > best_acc or mean_ap > best_ap or f1 > best_f1 or mcc > best_mcc:
                print('saving best model at the end of epoch %d' % (epoch))
                model.save_networks( f'model_epoch_best.pth' ) # _{epoch}
                best_acc = mean_acc if mean_acc > best_acc else best_acc
                best_ap = mean_ap if mean_ap > best_ap else best_ap
                best_f1 = f1 if f1 > best_f1 else best_f1
                best_mcc = mcc if mcc > best_mcc else best_mcc

            early_stopping(acc, model)
        
        # exit(0)
        
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                # early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
                early_stopping.reset()
            else:
                print("Early stopping.")
                break
        model.train()
        print()