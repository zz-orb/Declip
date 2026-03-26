import torch
from .datasets import RealFakeDataset, RealFakeDetectionDataset
from .datasets import MyRealFakeDataset, RS_Data_RealFakeDataset, PSCCData_RealFakeDataset, Noise_RealFakeDataset
from .datasets import RS_Data_RealFakeDetectionDataset

def create_dataloader(opt):
    shuffle = True if opt.data_label == 'train' else False
    if opt.fully_supervised:
        # dataset = RealFakeDataset(opt)
        # ADD
        print(f"opt.train_dataset: {opt.train_dataset}")
        if 'DOTA' in opt.train_dataset:
            dataset = MyRealFakeDataset(opt)
        elif 'Hifi' in opt.train_dataset or 'PSCC' in opt.train_dataset:
            dataset = PSCCData_RealFakeDataset(opt)
        elif 'Noise' in opt.train_dataset:
            dataset = Noise_RealFakeDataset(opt)
        else: # RS-Data
            dataset = RS_Data_RealFakeDataset(opt)
    else:
        if 'SIOR' in opt.train_dataset or 'SOTA' in opt.train_dataset or 'FAST' in opt.train_dataset:
            dataset = RS_Data_RealFakeDetectionDataset(opt)
        else:
            dataset = RealFakeDetectionDataset(opt)
    
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              num_workers=int(opt.num_threads))
    return data_loader
