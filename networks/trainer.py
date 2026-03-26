import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_model import BaseModel
from models import get_model
from utils.utils import compute_batch_iou, compute_batch_localization_f1, compute_batch_ap
from utils.area_loss import DiceLoss, DynamicLossScheduler

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.model = get_model(opt)

        # ADD
        if opt.pretrain_ckpt != None:
            print(f"Loading pretrained model from {opt.pretrain_ckpt}")
            state_dict = torch.load(opt.pretrain_ckpt, map_location='cpu')
            if not (opt.feature_layer == state_dict['feature_layer'] and \
            opt.decoder_type == state_dict['decoder_type']):
                print('feature_layer or decoder_type in the checkpoint state_dict not match args')
                exit(0)
            self.model.load_state_dict(state_dict['model'], strict=False)
            print("Pretrained model loaded successfully")
        
        # Initialize all possible parameters in the final layer
        for fc in self.model.fc:
            try:
                torch.nn.init.normal_(fc.weight.data, 0.0, opt.init_gain)
            except:
                pass

        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                # print(name)
                if "fc" in name and "resblock" not in name:
                    params.append(p) 
                    print(f"Training {name} in fc")
                # ADD
                elif self.model.use_noise_view and "DCTHighPass" in name:
                    params.append(p)
                    print(f"Training {name} with noise view")
                elif self.model.use_noise_view and "noise_fusion" in name:
                    params.append(p)
                    print(f"Training {name} with noise view")
                elif self.model.use_aspp and "aspp" in name:
                    params.append(p)
                    print(f"Training {name} with aspp")
                elif self.model.use_conprn and "conprn" in name:
                    params.append(p)
                    print(f"Training {name} with conprn")
                elif self.model.use_simdet and "detector" in name:
                    params.append(p)
                    print(f"Training {name} with simdet")
                else:
                    p.requires_grad = False
            # exit(0)
                    
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.grad_accum_steps = max(1, opt.grad_accum_steps)
        self._accum_step = 0
        self.optimizer.zero_grad(set_to_none=True)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.model.to(opt.gpu_ids[0])
        
        if opt.fully_supervised:
            self.ious = []
            self.F1_best = []
            self.F1_fixed = []
            self.ap = []
        else:
            self.logits = []
            self.labels = []
        
        # ADD
        if self.opt.use_area_loss:
            self.area_loss_fn = DiceLoss()
            self.DynamicLossScheduler = DynamicLossScheduler()
        
        if self.opt.use_simdet:
            self.det_f1 = []
            self.det_auc = []
            self.det_mcc = []

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        # ADD
        if self.opt.use_conprn or self.opt.use_simdet:  
            self.model.set_label(self.label)
        
        self.output = self.model(self.input)
        
        if self.opt.fully_supervised:
            # resize prediction to ground truth mask size
            if self.label.size()[1] != 256 * 256:
                label_size = (int(self.label.size()[1] ** 0.5), int(self.label.size()[1] ** 0.5))
                self.output = self.output.view(-1, 1, 256, 256)
                self.output = F.interpolate(self.output, size=label_size, mode='bilinear', align_corners=False)
                self.output = torch.flatten(self.output, start_dim=1).unsqueeze(1)

        if not self.opt.fully_supervised:
            self.output = torch.mean(self.output, dim=1)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self, epoch=0):
        self.forward()
        outputs = self.output
        
        if self.opt.fully_supervised:
            sigmoid_outputs = torch.sigmoid(outputs)
            
            # unflatten outputs and ground truth masks
            sigmoid_outputs = sigmoid_outputs.view(sigmoid_outputs.size(0), int(sigmoid_outputs.size(1)**0.5), int(sigmoid_outputs.size(1)**0.5))
            labels = self.label.view(self.label.size(0), int(self.label.size(1)**0.5), int(self.label.size(1)**0.5))
            
            iou = compute_batch_iou(sigmoid_outputs, labels)
            self.ious.extend(iou)

            F1_best, F1_fixed = compute_batch_localization_f1(sigmoid_outputs, labels)
            self.F1_best.extend(F1_best)
            self.F1_fixed.extend(F1_fixed)
            
            ap = compute_batch_ap(sigmoid_outputs, labels)
            self.ap.extend(ap)

        else:
            # print(f"detection: outputs shape: {outputs.shape}, label shape: {self.label.shape}")
            self.logits.append(outputs)
            self.labels.append(self.label)


        self.loss = self.loss_fn(outputs, self.label) 

        # ADD
        if self.opt.use_area_loss:
            self.DynamicLossScheduler.update(epoch)
            area_loss = self.area_loss_fn(sigmoid_outputs, self.label)
            # print(f"Effective IoU weight: {self.DynamicLossScheduler.effective_weight}, Area loss: {area_loss.item()}")
            # print(f"BCE_loss: {self.loss.item()}")
            self.loss += self.DynamicLossScheduler.effective_weight * area_loss
            # print(f"Total loss: {self.loss.item()}")

        if self.opt.use_conprn:
            # print(f"loss before ConPRN: {self.loss.item()}")
            # print(f"ConPRN loss: {self.model.con_loss.item()}")
            self.loss += self.model.con_loss
            # print(f"loss after ConPRN: {self.loss.item()}")
        
        if self.opt.use_simdet:
            image_labels = self.model.detector.get_image_labels(self.label)
            det_loss, det_metrics = self.model.detector.compute_loss_and_metrics(self.model.det_pred_probs, image_labels)
            # print(f"loss before SimDet: {self.loss.item()}")
            # print(f"SimDet loss: {det_loss.item()}")
            if epoch < 3:
                det_loss *= 0.0
            else:
                det_loss *= 0.1
            self.loss += det_loss
            # print(f"loss after SimDet: {self.loss.item()}")
            # detection metric
            # print(f"det_metrics: {det_metrics}")
            self.det_f1.extend([det_metrics['f1']])
            self.det_auc.extend([det_metrics['auc']])
            self.det_mcc.extend([det_metrics['mcc']])
            
        (self.loss / self.grad_accum_steps).backward()
        self._accum_step += 1

        if self._accum_step >= self.grad_accum_steps:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_step = 0

    def finalize_optimizer_step(self):
        if self._accum_step > 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_step = 0

    def format_output(self):
        if not self.opt.fully_supervised:
            self.logits = torch.cat(self.logits, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
