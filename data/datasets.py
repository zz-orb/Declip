import os
import random
import pickle
from io import BytesIO
from PIL import Image, ImageOps, ImageFile
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from random import shuffle

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants for mean and standard deviation
MEAN = [0.48145466, 0.4578275, 0.40821073]

STD = [0.26862954, 0.26130258, 0.27577711]

# Helper functions
def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", 'tif', 'tiff']):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if file.split('.')[-1] in exts and must_contain in os.path.join(r, file):
                out.append(os.path.join(r, file))
    return out

def get_list(path, must_contain=''):
    if path.endswith(".pickle"):
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        return [item for item in image_list if must_contain in item]
    return recursively_read(path, must_contain)

def randomJPEGcompression(image):
    qf = random.randint(30, 100)
    output_io_stream = BytesIO()
    image.save(output_io_stream, "JPEG", quality=qf, optimize=True)
    output_io_stream.seek(0)
    return Image.open(output_io_stream)

# Base Dataset class
class BaseDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self._init_data()

    def _init_data(self):
        pass

    def _get_data(self):
        pass

    def _get_transform(self):
        transform_list = [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]
        # ADD
        # if self.opt.data_label == 'train':
        if self.opt.data_aug == "blur":
            transform_list.insert(1, transforms.GaussianBlur(kernel_size=5, sigma=(0.4, 2.0)))
        elif self.opt.data_aug == "color_jitter":
            transform_list.insert(1, transforms.ColorJitter(0.3, 0.3, 0.3, 0.3))
        elif self.opt.data_aug == "jpeg_compression":
            transform_list.insert(1, transforms.Lambda(randomJPEGcompression))
        elif self.opt.data_aug == "all":
            transform_list.insert(1, transforms.ColorJitter(0.3, 0.3, 0.3, 0.3))
            transform_list.insert(2, transforms.Lambda(randomJPEGcompression))
            transform_list.insert(3, transforms.GaussianBlur(kernel_size=5, sigma=(0.4, 2.0)))

        return transforms.Compose(transform_list)

    def __len__(self):
        pass

class RealFakeDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.mask_transf = self._get_mask_transform()

    def _init_data(self):
        if self.opt.data_label == "train":
            self.input_path = self.opt.train_path
            self.masks_path = self.opt.train_masks_ground_truth_path
        elif self.opt.data_label == "valid":
            self.input_path = self.opt.valid_path
            self.masks_path = self.opt.valid_masks_ground_truth_path
        elif self.opt.data_label == "test":
            self.input_path = self.opt.test_path
            self.masks_path = self.opt.test_masks_ground_truth_path

        fake_list = self._get_data()
        
        self.labels_dict = self._set_labels(fake_list)
        self.fake_list = fake_list
        shuffle(self.fake_list)
        self.transform = self._get_transform()

    def _get_data(self):
        fake_list = get_list(self.input_path)
                
        return fake_list

    def _get_mask_transform(self):
        return transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        
    def get_mask_from_file(self, file_name):
        if "autosplice" in self.opt.train_dataset:
            file_name = file_name[:file_name.rfind('_')] + "_mask.png"
        self.mask_path = os.path.join(self.masks_path, file_name)
        mask = Image.open(self.mask_path).convert("L")
        if self.opt.train_dataset in ['pluralistic', 'lama', 'repaint-p2-9k', 'ldm', 'ldm_clean', 'ldm_real']:
            mask = ImageOps.invert(mask)
        return self.mask_transf(mask).view(-1)

    def _set_labels(self, fake_list):
        # masks images should be .png
        labels = {img: img.split("/")[-1].replace(".jpg", ".png") for img in fake_list}
        return labels
    
    def __len__(self):
        return len(self.fake_list)

    def __getitem__(self, idx):
        img_path = self.fake_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.get_mask_from_file(label)

        return img, label, img_path, self.mask_path

class RealFakeDetectionDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)

    def _get_data(self):
        fake_list = get_list(self.input_path)
        real_list = get_list(self.input_path_real)
                
        return real_list, fake_list

    def _init_data(self):
        if self.opt.data_label == "train":
            self.input_path = self.opt.train_path
            self.input_path_real = self.opt.train_real_list_path
            self.masks_path = self.opt.train_masks_ground_truth_path
        elif self.opt.data_label == "valid":
            self.input_path = self.opt.valid_path
            self.input_path_real = self.opt.valid_real_list_path
            self.masks_path = self.opt.valid_masks_ground_truth_path
        elif self.opt.data_label == "test":
            self.input_path = self.opt.test_path
            self.input_path_real = self.opt.test_real_list_path
            self.masks_path = self.opt.test_masks_ground_truth_path

        real_list, fake_list = self._get_data()
        self.labels_dict = self._set_labels(real_list, fake_list)
        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        self.transform = self._get_transform()

    def _set_labels(self, real_list, fake_list):
        labels = {img: 0 for img in real_list}
        labels.update({img: 1 for img in fake_list})
        return labels
    
    def __len__(self):
        return len(self.total_list)
    
    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label, img_path

# add 
class MyRealFakeDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.mask_transf = self._get_mask_transform()

    def _init_data(self):
        self.base_path = self.opt.data_root_path
        if self.opt.data_label == "train":
            self.input_path = self.opt.train_path
        elif self.opt.data_label == "valid":
            if '\\' in self.opt.valid_path or '/' in self.opt.valid_path:
                self.base_path, self.input_path = os.path.split(self.opt.valid_path)
            else:
                self.input_path = self.opt.valid_path
        elif self.opt.data_label == "test":
            self.input_path = self.opt.test_path

        self.fake_list = self._get_data()

        if self.opt.data_label == "train":
            shuffle(self.fake_list)
        self.transform = self._get_transform()

    def _get_data(self):
        # fake_list = get_list(self.input_path)
        fake_list = []
        with open(os.path.join(self.base_path, self.input_path)) as f:
            contents = f.readlines()
            for content in contents:
                image_name = content.strip()
                fake_list.append(os.path.join(self.base_path,image_name))
        # print(len(fake_list))
        return fake_list

    def _get_mask_transform(self):
        return transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    def __len__(self):
        return len(self.fake_list)

    def __getitem__(self, idx):
        img_path = self.fake_list[idx]
        img = Image.open(img_path).convert("RGB")
        img_size = img.size     # ADD 获取图片大小
        img = self.transform(img)
        # add 
        if 'copymove' in img_path:
            mask_path = img_path.replace('-fake.png', '-fakemask.png')
        else:
            mask_path = img_path.replace('-fake.png', '-mask.png')

        if 'real' in img_path:
            mask = Image.new("L", img_size, 0)
        else:    
            mask = Image.open(mask_path).convert("L")

        label = self.mask_transf(mask).view(-1)

        return img, label, img_path, mask_path
    

# add 
class RS_Data_RealFakeDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.mask_transf = self._get_mask_transform()

    def _init_data(self):
        self.base_path = self.opt.data_root_path
        if self.opt.data_label == "train":
            self.input_path = self.opt.train_path
        elif self.opt.data_label == "valid":
            if '\\' in self.opt.valid_path or '/' in self.opt.valid_path:
                self.base_path, self.input_path = os.path.split(self.opt.valid_path)
            else:
                self.input_path = self.opt.valid_path
        elif self.opt.data_label == "test":
            self.input_path = self.opt.test_path

        self.fake_list = self._get_data()

        if self.opt.data_label == "train" or self.opt.data_label == "valid":
            if 'order' not in self.opt.train_path:
                shuffle(self.fake_list)
        self.transform = self._get_transform()

    def _get_data(self):
        fake_list = []
        with open(os.path.join(self.base_path, self.input_path)) as f:
            contents = f.readlines()
            for content in contents:
                image_name = content.strip()
                fake_list.append(os.path.join(self.base_path,image_name))
        return fake_list

    def _get_mask_transform(self):
        return transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        

    def __len__(self):
        return len(self.fake_list)

    def __getitem__(self, idx):
        img_path = self.fake_list[idx]
        img = Image.open(img_path).convert("RGB")
        img_size = img.size     # ADD 获取图片大小
        img = self.transform(img)
        # ADD 
        if '/attacked_images/' in img_path:
            mask_path = img_path.replace('/attacked_images/', '/masks/')
        else:
            mask_path = img_path.replace('/images/', '/masks/')
        # ADD 有jpg图片
        if '.jpg' in mask_path:
            mask_path = mask_path.replace('.jpg', '.png')
        # ADD 带有真是图片
        if 'real' in img_path:
            mask = Image.new("L", img_size, 0)
        else:
            mask = Image.open(mask_path).convert("L")

        label = self.mask_transf(mask).view(-1)

        return img, label, img_path, mask_path


# add 
class PSCCData_RealFakeDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.mask_transf = self._get_mask_transform()

    def _init_data(self):
        self.base_path = self.opt.data_root_path
        if self.opt.data_label == "train":
            self.input_path = self.opt.train_path
        elif self.opt.data_label == "valid":
            if '\\' in self.opt.valid_path or '/' in self.opt.valid_path:
                self.base_path, self.input_path = os.path.split(self.opt.valid_path)
            else:
                self.input_path = self.opt.valid_path
        elif self.opt.data_label == "test":
            self.input_path = self.opt.test_path

        self.fake_list = self._get_data()

        if self.opt.data_label == "train":
            shuffle(self.fake_list)
        self.transform = self._get_transform()

    def _get_data(self):
        fake_list = []
        with open(os.path.join(self.base_path, self.input_path)) as f:
            contents = f.readlines()
            for content in contents:
                image_name = content.strip()
                fake_list.append(os.path.join(self.base_path,image_name))
        return fake_list

    def _get_mask_transform(self):
        return transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    def __len__(self):
        return len(self.fake_list)

    def __getitem__(self, idx):
        img_path = self.fake_list[idx]
        img = Image.open(img_path).convert("RGB")
        img_size = img.size     # ADD 获取图片大小
        img = self.transform(img)
        # add 
        mask_path = img_path.replace('/fake/', '/mask/')
        if '.jpg' in mask_path:
            mask_path = mask_path.replace('.jpg', '.png')
        # ADD: simdet 带有真是图片
        if 'authentic' in img_path:
            mask = Image.new("L", img_size, 0)
        else:
            mask = Image.open(mask_path).convert("L")
        # mask = Image.open(mask_path).convert("L")
        label = self.mask_transf(mask).view(-1)

        return img, label, img_path, mask_path

# ADD DOTA鲁棒性测试
class Noise_RealFakeDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.mask_transf = self._get_mask_transform()

    def _init_data(self):
        self.base_path = self.opt.data_root_path
        if self.opt.data_label == "train":
            self.input_path = self.opt.train_path
        elif self.opt.data_label == "valid":
            if '\\' in self.opt.valid_path or '/' in self.opt.valid_path:
                self.base_path, self.input_path = os.path.split(self.opt.valid_path)
            else:
                self.input_path = self.opt.valid_path
        elif self.opt.data_label == "test":
            self.input_path = self.opt.test_path

        self.fake_list = self._get_data()

        if self.opt.data_label == "train":
            shuffle(self.fake_list)
        self.transform = self._get_transform()

    def _get_data(self):
        fake_list = []
        with open(os.path.join(self.base_path, self.input_path)) as f:
            contents = f.readlines()
            for content in contents:
                image_name = content.strip()
                fake_list.append(os.path.join(self.base_path,image_name))
        return fake_list

    def _get_mask_transform(self):
        return transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    def __len__(self):
        return len(self.fake_list)

    def __getitem__(self, idx):
        img_path = self.fake_list[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        # print(f"Image path: {img_path}")
        if 'resize' not in img_path:
            mask_path = os.path.join(self.base_path, 'mask', os.path.basename(img_path.replace('fake', 'mask')))
        elif 'resize25' in img_path:
            mask_path = os.path.join(self.base_path, 'mask-resize25', os.path.basename(img_path.replace('fake', 'mask')))
        elif 'resize50' in img_path:
            mask_path = os.path.join(self.base_path, 'mask-resize50', os.path.basename(img_path.replace('fake', 'mask')))
        elif 'resize75' in img_path:
            mask_path = os.path.join(self.base_path, 'mask-resize75', os.path.basename(img_path.replace('fake', 'mask')))
        else:
            mask_path = os.path.join(self.base_path, 'mask', os.path.basename(img_path.replace('fake', 'mask')))
        if '.jpg' in mask_path:
            mask_path = mask_path.replace('.jpg', '.png')
        if 'real' in img_path:
            mask_path = img_path
        mask = Image.open(mask_path).convert("L")
        label = self.mask_transf(mask).view(-1)

        return img, label, img_path, mask_path


# ADD RS-DATA detiction训练
class RS_Data_RealFakeDetectionDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)

    def _get_data(self, data_path=""):
        file_list = []
        with open(data_path) as f:
            contents = f.readlines()
            for content in contents:
                image_name = content.strip()
                file_list.append(os.path.join(self.base_path,image_name))
        return file_list

    # def _get_data(self):
    #     fake_list = get_list(self.input_path)
    #     real_list = get_list(self.input_path_real)
        
    #     return real_list, fake_list

    def _init_data(self):
        self.base_path = self.opt.data_root_path
        if self.opt.data_label == "train":
            self.input_path = self.opt.train_path
            self.input_path_real = self.opt.train_real_list_path
            # self.masks_path = self.opt.train_masks_ground_truth_path
        elif self.opt.data_label == "valid":
            self.input_path = self.opt.valid_path
            self.input_path_real = self.opt.valid_real_list_path
            # self.masks_path = self.opt.valid_masks_ground_truth_path
        elif self.opt.data_label == "test":
            self.input_path = self.opt.test_path
            self.input_path_real = self.opt.test_real_list_path
            # self.masks_path = self.opt.test_masks_ground_truth_path

        # real_list, fake_list = self._get_data()
        fake_list = self._get_data(os.path.join(self.base_path, self.input_path))
        real_list = self._get_data(os.path.join(self.base_path, self.input_path_real))
        self.labels_dict = self._set_labels(real_list, fake_list)
        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        self.transform = self._get_transform()
        # print("len total_list: ", len(self.total_list))

    def _set_labels(self, real_list, fake_list):
        labels = {img: 0 for img in real_list}
        labels.update({img: 1 for img in fake_list})
        return labels
    
    def __len__(self):
        return len(self.total_list)
    
    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label, img_path

