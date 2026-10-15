import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels, create_exponential_distribution, create_dual_exp_distribution, unjson, get_img_from_path
from collections import Counter
#rho = 0.01
# 'lt': [21, 36, 13, 344, 171, 10, 7, 24, 15, 14, 77, 7, 434, 6, 38, 328, 149, 12, 67, 85, 33, 19, 13, 477, 
# 9, 206, 226, 48, 135, 42, 273, 11, 61, 11, 378, 32, 10, 237, 248, 64, 7, 74, 17, 30, 12, 44, 197, 
# 314, 118, 40, 89, 6, 260, 18, 5, 5, 5, 455, 25, 23, 70, 179, 98, 9, 163, 102, 8, 188, 5, 500, 8, 
# 142, 216, 6, 299, 286, 56, 156, 123, 58, 27, 20, 93, 29, 361, 26, 15, 396, 112, 415, 46, 53, 16, 
#  6, 81, 22, 129, 51, 35, 107],


 # #rho = 0.03
 #'lt': [46.0, 17.0, 363.0, 66.0, 68.0, 283.0, 179.0, 326.0, 246.0, 53.0, 304.0, 85.0, 16.0, 273.0, 264.0, 139.0, 79.0, 21.0, 28.0, 23.0, 29.0, 43.0, 221.0, 160.0, 229.0, 57.0, 199.0, 82.0, 22.0, 255.0, 500.0, 88.0, 71.0, 166.0, 31.0, 376.0, 433.0, 192.0, 112.0, 32.0, 16.0, 482.0, 48.0, 36.0, 91.0, 18.0, 22.0, 172.0, 76.0, 125.0, 144.0, 293.0, 315.0, 155.0, 117.0, 44.0, 121.0, 130.0, 105.0, 55.0, 25.0, 73.0, 449.0, 15.0, 41.0, 50.0, 24.0, 59.0, 17.0, 64.0, 404.0, 149.0, 15.0, 19.0, 26.0, 94.0, 39.0, 134.0, 27.0, 61.0, 37.0, 101.0, 338.0, 109.0, 30.0, 19.0, 35.0, 418.0, 40.0, 390.0, 98.0, 465.0, 20.0, 237.0, 185.0, 33.0, 206.0, 350.0, 51.0, 213.0],          

# #rho = 0.07
#'lt': [102.0, 92.0, 145.0, 235.0, 87.0, 255.0, 223.0, 43.0, 35.0, 35.0, 80.0, 97.0, 284.0, 48.0, 325.0, 108.0, 372.0, 382.0, 206.0, 50.0, 36.0, 217.0, 334.0, 414.0, 161.0, 403.0, 343.0, 425.0, 248.0, 74.0, 76.0, 461.0, 134.0, 53.0, 292.0, 63.0, 137.0, 123.0, 500.0, 175.0, 352.0, 180.0, 437.0, 262.0, 56.0, 44.0, 166.0, 41.0, 362.0, 300.0, 449.0, 38.0, 316.0, 72.0, 127.0, 70.0, 185.0, 66.0, 61.0, 99.0, 117.0, 94.0, 195.0, 473.0, 130.0, 269.0, 111.0, 64.0, 153.0, 308.0, 211.0, 170.0, 120.0, 52.0, 190.0, 89.0, 200.0, 84.0, 45.0, 59.0, 114.0, 276.0, 78.0, 55.0, 49.0, 149.0, 392.0, 58.0, 68.0, 229.0, 37.0, 40.0, 42.0, 82.0, 486.0, 105.0, 141.0, 47.0, 157.0, 242.0],

# #rho = 0.1
# 'lt': [54.0, 201.0, 52.0, 79.0, 405.0, 254.0, 66.0, 415.0, 61.0, 102.0, 336.0, 139.0, 53.0, 135.0, 237.0, 183.0, 56.0, 74.0, 488.0, 179.0, 306.0, 344.0, 260.0, 159.0, 387.0, 132.0, 321.0, 85.0, 369.0, 112.0, 87.0, 51.0, 63.0, 232.0, 58.0, 156.0, 75.0, 105.0, 110.0, 70.0, 93.0, 455.0, 221.0, 64.0, 69.0, 98.0, 396.0, 67.0, 286.0, 500.0, 273.0, 149.0, 72.0, 142.0, 192.0, 95.0, 126.0, 445.0, 60.0, 152.0, 197.0, 115.0, 378.0, 226.0, 361.0, 248.0, 57.0, 118.0, 188.0, 77.0, 299.0, 89.0, 211.0, 424.0, 91.0, 292.0, 206.0, 163.0, 279.0, 167.0, 352.0, 216.0, 145.0, 328.0, 466.0, 50.0, 171.0, 107.0, 243.0, 175.0, 477.0, 266.0, 83.0, 123.0, 434.0, 314.0, 81.0, 129.0, 100.0, 121.0]

#rho = 0.005
# 'lt': [38.0, 277.0, 5.0, 3.0, 24.0, 72.0, 26.0, 27.0, 100.0, 69.0, 425.0, 362.0, 29.0, 52.0, 30.0, 
# 4.0, 13.0, 162.0, 32.0, 2.0, 34.0, 403.0, 19.0, 58.0, 2.0, 65.0, 308.0, 3.0, 5.0, 138.0, 47.0, 17.0, 
# 20.0, 12.0, 18.0, 5.0, 146.0, 4.0, 154.0, 21.0, 343.0, 6.0, 190.0, 10.0, 224.0, 7.0, 111.0, 3.0, 8.0, 
# 201.0, 16.0, 42.0, 131.0, 124.0, 22.0, 236.0, 14.0, 23.0, 2.0, 212.0, 81.0, 62.0, 90.0, 3.0, 263.0, 3.0, 
# 13.0, 8.0, 50.0, 325.0, 473.0, 449.0, 180.0, 95.0, 382.0, 44.0, 11.0, 292.0, 500.0, 85.0, 6.0, 11.0, 105.0, 
# 6.0, 40.0, 117.0, 4.0, 171.0, 4.0, 76.0, 249.0, 9.0, 9.0, 55.0, 36.0, 5.0, 2.0, 7.0, 10.0, 15.0],

lt_class_distribution = [21, 36, 13, 344, 171, 10, 7, 24, 15, 14, 77, 7, 434, 6, 38, 328, 149, 12, 67, 85, 33, 19, 13, 477, 
9, 206, 226, 48, 135, 42, 273, 11, 61, 11, 378, 32, 10, 237, 248, 64, 7, 74, 17, 30, 12, 44, 197, 
314, 118, 40, 89, 6, 260, 18, 5, 5, 5, 455, 25, 23, 70, 179, 98, 9, 163, 102, 8, 188, 5, 500, 8, 
142, 216, 6, 299, 286, 56, 156, 123, 58, 27, 20, 93, 29, 361, 26, 15, 396, 112, 415, 46, 53, 16, 
 6, 81, 22, 129, 51, 35, 107]

lt_class_order = [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]

cifar100_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

#-----------------------------------------------------------------------------------------------

def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

def build_transform_patch_align(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    # if is_train:
    #     # Instead of RandomResizedCrop, use a simple Resize to the input size.
    #     transform = [
    #         transforms.Resize(input_size, interpolation=3),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.ToTensor(),
    #     ]
    #     return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # Maintain similar ratio as in the original function
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # You can optionally compose the transforms if needed:
    # return transforms.Compose(t)
    return t

    #-----------------------------------------------------------------------------------------------

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

# Modified CIFAR100 for LT data
class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    # class_order = np.arange(100).tolist()
    class_order = lt_class_order
    class_distribution = lt_class_distribution

    # def download_data(self):
    #     train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
    #     test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        # self.train_data, self.train_targets = train_dataset.data, np.array(
        #     train_dataset.targets
        # )
        # self.test_data, self.test_targets = test_dataset.data, np.array(
        #     test_dataset.targets
        # )

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        train_data, train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

        selected_data = []
        selected_targets = []
        for i, class_index in enumerate(class_order):
            num_samples = class_distribution[i]
            class_indices = np.where(self.train_targets == class_index)[0]
            np.random.shuffle(class_indices)
            selected_indices = class_indices[:num_samples]

            # Add selected data and targets to lists
            selected_data.append(train_data[selected_indices])
            selected_targets.append(train_targets[selected_indices])

        self.train_data = np.concatenate(selected_data, axis = 0)
        self.train_targets = np.concatenate(selected_targets, axis=0)


# Modified CIFAR224 for LT data
class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False
        self.dataset = 'cifar224'
        self.num_classes = 100
        self.initial = args["init_cls"]
        self.inc = args["increment"]
        

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        # self.class_order  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 91, 92, 93, 94, 95, 96, 97, 98, 99, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        # self.class_order = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        self.class_order = lt_class_order
        # rho = 0.01, 0.05
        # self.class_distribution = create_exponential_distribution(100, 500, 0.01) #Num of classes, max samples, rho
        self.class_distribution = create_dual_exp_distribution(100, 500, 0.01, 0.01, 3, self.inc)
        
    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("/home/abhishek/SVP/local_datasets/", train=True, download=False)
        test_dataset = datasets.cifar.CIFAR100("/home/abhishek/SVP/local_datasets/", train=False, download=False)
        train_data, train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)
        self.train_label_names ,self.test_label_names = [], []
        for i in self.test_targets:
            self.test_label_names.append(cifar100_labels[i])
        self.test_label_names = np.array(self.test_label_names)
        selected_data = []
        selected_targets = []
        for i, class_index in enumerate(self.class_order):
            num_samples = self.class_distribution[i]
            class_indices = np.where(train_targets == class_index)[0]
            # np.random.shuffle(class_indices)
            selected_indices = class_indices[:num_samples]

            # Add selected data and targets to lists
            selected_data.append(train_data[selected_indices])
            selected_targets.append(train_targets[selected_indices])
        
        for i in selected_targets:
            self.train_label_names.append(cifar100_labels[int(i[0])])
            
        self.train_data = np.concatenate(selected_data, axis = 0)
        self.train_targets = np.concatenate(selected_targets, axis=0)
        self.train_label_names = np.array(self.train_label_names)
        print(f"class counts: {Counter(self.train_targets)}")


class iNaturalist(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.dataset = 'inaturalist'

        self.datafile = "data/iNaturalist/train_100.json"
        self.test_datafile = "data/iNaturalist/val_100.json"

        self.train_trsf = build_transform_patch_align(True, None)
        self.test_trsf = build_transform(False, None)
        self.common_trsf = []

        # if counts == 100
        self.class_order = lt_class_order
        #If counts = 300
        # self.class_order = [169, 221, 146, 296, 92, 229, 49, 70, 126, 77, 243, 57, 120, 223, 27, 217, 154, 8, 101, 182, 54, 50, 97, 162, 28, 193, 79, 6, 151, 0, 173, 187, 44, 277, 157, 16, 257, 122, 35, 9, 171, 176, 134, 4, 87, 255, 112, 153, 246, 286, 147, 186, 107, 98, 10, 7, 266, 214, 36, 165, 139, 254, 103, 269, 59, 23, 298, 39, 73, 208, 135, 299, 102, 20, 12, 268, 11, 209, 161, 210, 74, 42, 56, 99, 235, 133, 159, 189, 145, 238, 215, 241, 138, 195, 233, 279, 88, 199, 64, 280, 26, 203, 17, 245, 174, 236, 137, 252, 80, 68, 181, 150, 118, 94, 75, 2, 109, 119, 232, 15, 69, 184, 201, 261, 55, 287, 273, 66, 175, 32, 142, 136, 116, 76, 179, 1, 91, 248, 275, 240, 290, 170, 132, 106, 265, 270, 85, 19, 284, 43, 259, 293, 228, 200, 104, 227, 51, 224, 163, 297, 216, 260, 93, 46, 100, 272, 105, 72, 131, 3, 156, 256, 249, 48, 113, 82, 172, 213, 62, 53, 271, 168, 253, 226, 183, 180, 276, 283, 78, 61, 143, 278, 30, 141, 196, 264, 231, 52, 222, 40, 125, 219, 65, 166, 192, 289, 45, 205, 89, 267, 90, 71, 167, 262, 281, 212, 96, 84, 127, 67, 294, 178, 288, 197, 21, 18, 111, 108, 58, 230, 188, 31, 110, 291, 129, 24, 204, 83, 81, 251, 60, 13, 128, 14, 86, 63, 117, 198, 274, 5, 22, 144, 258, 121, 38, 207, 148, 292, 41, 234, 237, 114, 202, 37, 149, 206, 155, 29, 130, 177, 194, 152, 34, 33, 239, 160, 140, 47, 244, 25, 282, 158, 211, 123, 190, 218, 164, 247, 263, 191, 220, 115, 242, 250, 124, 295, 95, 285, 225, 185]
        # self.class_order = [299, 298, 297, 296, 295, 169, 221, 146, 92, 229, 49, 70, 126, 77, 243, 57, 120, 223, 27, 217, 154, 8, 101, 182, 54, 50, 97, 162, 28, 193, 79, 6, 151, 0, 173, 187, 44, 277, 157, 16, 257, 122, 35, 9, 171, 176, 134, 4, 87, 255, 112, 153, 246, 286, 147, 186, 107, 98, 10, 7, 266, 214, 36, 165, 139, 254, 103, 269, 59, 23, 39, 73, 208, 135, 102, 20, 12, 268, 11, 209, 161, 210, 74, 42, 56, 99, 235, 133, 159, 189, 145, 238, 215, 241, 138, 195, 233, 279, 88, 199, 64, 280, 26, 203, 17, 245, 174, 236, 137, 252, 80, 68, 181, 150, 118, 94, 75, 2, 109, 119, 232, 15, 69, 184, 201, 261, 55, 287, 273, 66, 175, 32, 142, 136, 116, 76, 179, 1, 91, 248, 275, 240, 290, 170, 132, 106, 265, 270, 85, 19, 284, 43, 259, 293, 228, 200, 104, 227, 51, 224, 163, 216, 260, 93, 46, 100, 272, 105, 72, 131, 3, 156, 256, 249, 48, 113, 82, 172, 213, 62, 53, 271, 168, 253, 226, 183, 180, 276, 283, 78, 61, 143, 278, 30, 141, 196, 264, 231, 52, 222, 40, 125, 219, 65, 166, 192, 289, 45, 205, 89, 267, 90, 71, 167, 262, 281, 212, 96, 84, 127, 67, 294, 178, 288, 197, 21, 18, 111, 108, 58, 230, 188, 31, 110, 291, 129, 24, 204, 83, 81, 251, 60, 13, 128, 14, 86, 63, 117, 198, 274, 5, 22, 144, 258, 121, 38, 207, 148, 292, 41, 234, 237, 114, 202, 37, 149, 206, 155, 29, 130, 177, 194, 152, 34, 33, 239, 160, 140, 47, 244, 25, 282, 158, 211, 123, 190, 218, 164, 247, 263, 191, 220, 115, 242, 250, 124, 95, 285, 225, 185]
        #If counts = 500
        # self.class_order = [275, 330, 132, 492, 118, 499, 50, 77, 356, 282, 101, 487, 110, 378, 479, 322, 367, 112, 210, 28, 252, 176, 36, 410, 351, 289, 428, 425, 403, 2, 294, 230, 399, 192, 312, 382, 78, 306, 478, 69, 138, 228, 156, 120, 283, 142, 401, 93, 56, 11, 122, 133, 126, 184, 243, 353, 424, 429, 70, 73, 246, 88, 272, 7, 472, 145, 441, 489, 281, 407, 163, 274, 337, 57, 236, 3, 240, 105, 171, 79, 44, 474, 181, 408, 99, 491, 440, 436, 63, 368, 448, 182, 22, 363, 26, 293, 193, 291, 374, 23, 154, 299, 386, 456, 245, 8, 103, 360, 89, 85, 347, 68, 321, 166, 153, 279, 438, 147, 340, 400, 379, 343, 108, 179, 238, 223, 268, 227, 443, 169, 355, 266, 201, 14, 119, 296, 350, 341, 137, 357, 157, 213, 256, 454, 418, 267, 6, 49, 128, 430, 136, 59, 335, 264, 393, 348, 146, 298, 139, 482, 269, 27, 159, 52, 241, 292, 97, 318, 30, 55, 80, 475, 195, 214, 75, 450, 300, 455, 253, 354, 94, 446, 10, 262, 260, 32, 170, 273, 372, 174, 470, 54, 15, 320, 437, 271, 344, 74, 464, 325, 336, 359, 396, 476, 313, 150, 0, 462, 284, 369, 161, 224, 217, 326, 117, 16, 76, 219, 35, 9, 496, 413, 98, 4, 421, 392, 483, 397, 165, 490, 375, 480, 135, 233, 226, 257, 447, 12, 327, 473, 162, 394, 102, 463, 87, 187, 151, 107, 229, 390, 39, 258, 265, 104, 493, 129, 186, 431, 20, 305, 255, 488, 196, 92, 106, 42, 231, 221, 232, 442, 204, 189, 31, 209, 200, 315, 439, 324, 109, 64, 459, 17, 311, 208, 297, 303, 121, 222, 409, 352, 406, 183, 494, 66, 346, 288, 205, 116, 316, 398, 134, 53, 1, 91, 310, 381, 461, 427, 199, 307, 45, 362, 19, 333, 143, 43, 175, 235, 414, 314, 125, 173, 328, 215, 216, 51, 21, 46, 485, 419, 444, 83, 498, 465, 248, 469, 100, 203, 290, 72, 131, 259, 420, 261, 48, 113, 82, 172, 370, 62, 497, 168, 391, 180, 460, 468, 334, 61, 254, 86, 286, 141, 365, 486, 249, 458, 308, 387, 40, 323, 65, 477, 301, 345, 495, 90, 71, 167, 466, 212, 412, 96, 84, 127, 67, 380, 178, 434, 197, 277, 18, 111, 364, 58, 188, 432, 366, 371, 385, 24, 339, 81, 422, 60, 251, 13, 471, 270, 445, 319, 373, 198, 452, 5, 278, 144, 457, 377, 38, 207, 148, 484, 415, 41, 338, 234, 383, 237, 404, 114, 276, 202, 287, 309, 37, 149, 206, 155, 29, 130, 177, 194, 152, 449, 426, 342, 332, 280, 34, 481, 405, 33, 239, 384, 160, 411, 402, 423, 395, 376, 140, 358, 47, 244, 25, 467, 158, 211, 123, 302, 190, 453, 218, 164, 317, 247, 263, 191, 220, 115, 331, 389, 416, 361, 242, 388, 435, 250, 124, 304, 329, 349, 295, 95, 451, 285, 225, 433, 185, 417]
        #if count = 800
        # self.class_order = [799, 798, 797, 796, 795, 794, 793, 792, 791, 790, 575, 550, 106, 257, 710, 653, 659, 580, 114, 504, 34, 699, 250, 263, 137, 641, 89, 113, 775, 355, 9, 680, 230, 327, 629, 191, 588, 525, 389, 82, 677, 519, 366, 751, 233, 245, 784, 669, 44, 464, 393, 451, 396, 549, 614, 774, 651, 275, 708, 657, 340, 324, 83, 161, 648, 307, 363, 753, 359, 217, 242, 713, 194, 375, 465, 43, 526, 685, 69, 745, 370, 291, 636, 627, 276, 54, 508, 541, 631, 593, 292, 220, 771, 201, 354, 400, 617, 570, 647, 132, 777, 573, 415, 429, 493, 164, 502, 261, 269, 345, 512, 709, 482, 737, 413, 444, 20, 278, 178, 681, 543, 719, 55, 749, 783, 785, 347, 22, 382, 392, 529, 406, 47, 472, 70, 10, 748, 337, 612, 282, 98, 199, 24, 236, 754, 123, 495, 353, 147, 32, 440, 30, 764, 187, 625, 144, 532, 590, 770, 148, 646, 14, 111, 48, 243, 584, 136, 597, 414, 675, 140, 586, 319, 411, 331, 59, 551, 620, 567, 769, 503, 206, 42, 25, 299, 635, 129, 213, 731, 513, 361, 547, 352, 520, 223, 128, 740, 1, 239, 468, 121, 390, 613, 768, 215, 587, 622, 649, 78, 728, 348, 720, 562, 521, 362, 351, 746, 264, 149, 404, 135, 478, 171, 758, 596, 744, 208, 79, 694, 265, 705, 391, 325, 35, 368, 599, 759, 180, 544, 618, 605, 372, 738, 533, 240, 650, 23, 715, 394, 315, 107, 179, 119, 488, 108, 395, 732, 603, 314, 31, 126, 336, 39, 663, 45, 254, 722, 356, 365, 207, 426, 273, 85, 322, 138, 294, 120, 66, 422, 185, 601, 232, 434, 4, 610, 528, 371, 330, 408, 457, 369, 781, 734, 349, 522, 752, 511, 296, 776, 49, 68, 301, 339, 181, 235, 679, 212, 640, 17, 226, 167, 28, 231, 564, 486, 420, 424, 664, 172, 344, 742, 718, 36, 672, 480, 293, 170, 716, 552, 165, 689, 741, 189, 156, 407, 760, 698, 615, 280, 118, 572, 295, 485, 553, 346, 342, 761, 306, 328, 317, 150, 216, 463, 630, 557, 88, 249, 515, 193, 518, 335, 600, 788, 298, 8, 157, 459, 224, 103, 380, 219, 376, 285, 374, 305, 555, 6, 271, 534, 701, 703, 712, 665, 611, 527, 381, 332, 427, 92, 367, 510, 634, 574, 566, 288, 247, 321, 484, 535, 559, 184, 333, 441, 492, 773, 639, 539, 455, 152, 252, 499, 497, 159, 578, 5, 56, 479, 717, 432, 691, 569, 658, 558, 405, 16, 621, 430, 763, 77, 683, 174, 642, 739, 401, 772, 421, 727, 475, 687, 318, 117, 517, 289, 542, 211, 439, 589, 91, 726, 787, 490, 598, 786, 19, 654, 538, 81, 287, 779, 204, 501, 673, 645, 104, 364, 38, 267, 323, 423, 643, 134, 454, 452, 258, 96, 624, 214, 175, 693, 443, 418, 151, 388, 246, 302, 762, 591, 531, 154, 628, 13, 633, 378, 75, 262, 190, 592, 676, 326, 221, 133, 182, 41, 530, 141, 303, 297, 453, 266, 58, 274, 671, 284, 29, 377, 51, 18, 419, 101, 50, 476, 281, 251, 329, 15, 696, 684, 197, 583, 660, 417, 702, 456, 177, 445, 125, 577, 514, 385, 780, 3, 686, 256, 470, 397, 604, 458, 450, 127, 241, 644, 283, 244, 666, 581, 237, 461, 122, 442, 576, 563, 253, 72, 386, 652, 537, 619, 188, 162, 384, 255, 176, 448, 100, 202, 765, 227, 163, 546, 168, 416, 304, 674, 425, 755, 725, 670, 579, 270, 690, 471, 379, 277, 481, 433, 536, 286, 697, 524, 469, 692, 102, 338, 767, 523, 21, 173, 560, 595, 491, 186, 467, 259, 545, 505, 711, 607, 153, 142, 71, 46, 205, 373, 341, 540, 500, 516, 565, 7, 704, 158, 714, 568, 594, 357, 76, 409, 97, 300, 585, 93, 637, 2, 84, 94, 80, 623, 733, 198, 554, 707, 229, 730, 248, 750, 309, 87, 112, 757, 507, 234, 37, 435, 169, 766, 183, 735, 74, 616, 506, 313, 403, 195, 131, 40, 203, 383, 52, 279, 228, 209, 410, 222, 160, 387, 688, 110, 145, 447, 466, 64, 436, 494, 358, 311, 272, 210, 196, 449, 756, 343, 428, 431, 402, 655, 723, 192, 109, 61, 218, 438, 724, 609, 606, 116, 166, 706, 0, 661, 789, 667, 105, 473, 360, 474, 437, 115, 124, 496, 477, 626, 632, 782, 462, 26, 238, 63, 65, 483, 736, 200, 33, 12, 290, 489, 729, 90, 350, 308, 146, 662, 316, 656, 602, 62, 498, 412, 95, 548, 398, 73, 561, 139, 747, 556, 695, 446, 700, 67, 99, 608, 743, 268, 668, 638, 678, 310, 399, 53, 86, 312, 721, 487, 320, 778, 155, 682, 60, 571, 130, 582, 27, 260, 509, 143, 11, 334, 460, 225, 57]
        


    def download_data(self):

        data_dict_train = unjson(self.datafile)
        data_dict_test = unjson(self.test_datafile)

        datapath_train = data_dict_train['data_path']
        train_data = get_img_from_path(datapath_train, self.dataset)
        train_targets = np.asarray(data_dict_train['labels'])
        train_label_names = np.asarray(data_dict_train['label_names'])

        datapath_test = data_dict_test['data_path']
        self.test_data = get_img_from_path(datapath_test, self.dataset)
        self.test_targets = np.asarray(data_dict_test['labels'])
        self.test_label_names = np.asarray(data_dict_test['label_names'])

        selected_data = []
        selected_targets = []
        selected_label_names = []
        for i, class_index in enumerate(self.class_order):
            class_indices = np.where(train_targets == class_index)[0]
            np.random.shuffle(class_indices)
            
            # Add selected data and targets to lists
            selected_data.append(train_data[class_indices])
            selected_targets.append(train_targets[class_indices])
            selected_label_names.append(train_label_names[class_indices])
            
        self.train_data = np.concatenate(selected_data, axis = 0)
        self.train_targets = np.concatenate(selected_targets, axis=0)
        self.train_label_names = np.concatenate(selected_label_names, axis=0)

        

class food101(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.dataset = 'food101'

        self.datafile = "data/food101/train_file_upd.json"
        self.test_datafile = "data/food101/test_file_upd.json"

        self.train_trsf = build_transform(True, None)
        self.test_trsf = build_transform(False, None)
        self.common_trsf = []
        self.initial = args["init_cls"]
        self.inc = args["increment"]

        #exp_dist
        self.class_order = [35, 36, 27, 55, 58, 37, 9, 88, 75, 10, 57, 54, 86, 90, 97, 72, 78, 6, 40, 49, 19, 23, 14, 45, 43, 96, 87, 12, 66, 16, 5, 62, 56, 31, 25, 65, 61, 84, 4, 53, 41, 18, 28, 80, 81, 69, 52, 39, 11, 82, 1, 0, 94, 29, 74, 71, 59, 73, 42, 51, 34, 85, 98, 44, 15, 33, 20, 89, 60, 95, 38, 68, 79, 93, 67, 99, 83, 2, 32, 3, 47, 50, 24, 48, 46, 76, 63, 26, 91, 8, 100, 92, 70, 30, 21, 64, 7, 77, 22, 17, 13]
        #exp_dual_dist
        # self.class_order  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        # self.class_order = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

        # rho = 0.01, 0.05
        # self.class_distribution = create_exponential_distribution(101, 750, 0.01) #Num of classes, max samples, rho
        self.class_distribution = create_dual_exp_distribution(101, 750, 0.01, 0.05, self.inc, self.initial)


    def download_data(self):

        data_dict_train = unjson(self.datafile)
        data_dict_test = unjson(self.test_datafile)

        datapath_train = data_dict_train['datapath']
        train_data = get_img_from_path(datapath_train, self.dataset)
        train_targets = np.asarray(data_dict_train['labels'])
        train_label_names = np.asarray(data_dict_train['class_name'])

        datapath_test = data_dict_test['datapath']
        self.test_data = get_img_from_path(datapath_test, self.dataset)
        self.test_targets = np.asarray(data_dict_test['labels'])
        self.test_label_names = np.asarray(data_dict_test['class_name'])

        selected_data = []
        selected_targets = []
        selected_label_names = []
        for i, class_index in enumerate(self.class_order):
            num_samples = self.class_distribution[i]
            class_indices = np.where(train_targets == class_index)[0]
            np.random.shuffle(class_indices)
            selected_indices = class_indices[:num_samples]

            # Add selected data and targets to lists
            selected_data.append(train_data[selected_indices])
            selected_targets.append(train_targets[selected_indices])
            selected_label_names.append(train_label_names[selected_indices])

        self.train_data = np.concatenate(selected_data, axis = 0)
        self.train_targets = np.concatenate(selected_targets, axis=0)
        self.train_label_names = np.concatenate(selected_label_names, axis=0)

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

"""
class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        #assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenets/train/"
        test_dir = "./data/imagenets/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet100(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.num_classes = 100

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        # 1. Define Class Order
        # Random permutation for 200 classes, or use a fixed seed if reproducibility is needed
        # self.class_order = np.arange(200).tolist() 
        np.random.seed(1993) # Optional: Ensure consistent order across runs
        self.class_order = np.random.permutation(self.num_classes).tolist()

        # 2. Define Class Distribution (Long-Tailed)
        # Assuming a max sample count per class (e.g., 500) and an imbalance ratio (rho)
        # You can use create_exponential_distribution or create_dual_exp_distribution from utils
        # Example: 200 classes, max 500 samples, rho=0.01 (Imbalance Factor 100)
        self.class_distribution = create_exponential_distribution(self.num_classes, 100, 0.01)

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-s/train/"
        test_dir = "./data/imagenet-s/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        # Get full data initially
        train_data, train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # Convert to numpy arrays for indexing
        train_data = np.array(train_data)
        train_targets = np.array(train_targets)

        # 3. Apply LT Selection Logic
        selected_data = []
        selected_targets = []
        
        # Verify we have enough samples, otherwise clip to available amount
        real_counts = Counter(train_targets)

        for i, class_index in enumerate(self.class_order):
            target_count = self.class_distribution[i]
            
            # Find indices for this class
            class_indices = np.where(train_targets == class_index)[0]
            
            # Shuffle indices to pick random samples
            np.random.shuffle(class_indices)
            
            # Handle case where real data has fewer samples than distribution requests
            available_count = len(class_indices)
            final_count = min(target_count, available_count)
            
            # Select samples
            selected_indices = class_indices[:final_count]

            selected_data.append(train_data[selected_indices])
            selected_targets.append(train_targets[selected_indices])

        # Concatenate and assign back
        self.train_data = np.concatenate(selected_data, axis=0)
        self.train_targets = np.concatenate(selected_targets, axis=0)
        
        print(f"ImageNet-R LT generated. Total samples: {len(self.train_data)}")
"""
import os
class iImageNet100(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.dataset = 'imagenet100'
        self.num_classes = 100
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]
        
        # --- Transforms: Standard ImageNet Protocol ---
        # "Original" ImageNet100 requirements usually mandate normalization 
        # and standard cropping, which build_transform() in your file might miss.
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            # Explicitly define standard ImageNet transforms
            self.train_trsf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
            self.test_trsf = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
            
        self.common_trsf = [] # Normalization is already inside train/test_trsf

        # --- Class Order & Distribution ---
        # Use the provided lt_class_order (permutation of 0-99)
        self.class_order = lt_class_order
        
        # Create Exponential Distribution (Long-Tail)
        # Max samples approx 1300 for ImageNet. 
        # Rho and other params matched to your iCIFAR224 example.
        self.class_distribution = create_dual_exp_distribution(100, 1200, 0.01, 0.01, 3, self.inc)
        print(self.class_distribution)

    def download_data(self):
        # Assumes folder structure: data/imagenet100/train and data/imagenet100/val
        # containing 100 subfolders each.
        root_train = os.path.join("./data", "imagenet-s", "train")
        root_test = os.path.join("./data", "imagenet-s", "test")

        train_dataset = datasets.ImageFolder(root_train)
        test_dataset = datasets.ImageFolder(root_test)

        # Get paths and targets
        # train_dataset.samples is a list of (path, label_idx)
        train_samples = np.array(train_dataset.samples)
        train_paths = train_samples[:, 0]
        train_targets = train_samples[:, 1].astype(int)

        test_samples = np.array(test_dataset.samples)
        self.test_data = test_samples[:, 0]
        self.test_targets = test_samples[:, 1].astype(int)

        # --- Apply Exponential Distribution Subsampling ---
        selected_paths = []
        selected_targets = []

        for i, class_index in enumerate(self.class_order):
            # Target count for this class from the distribution
            num_samples = self.class_distribution[i]
            
            # Find indices for this class
            class_indices = np.where(train_targets == class_index)[0]
            
            # Subsample if we have more images than the distribution requires
            if len(class_indices) > num_samples:
                # np.random.shuffle(class_indices) # Shuffle if you want random selection
                selected_indices = class_indices[:num_samples]
            else:
                selected_indices = class_indices

            selected_paths.append(train_paths[selected_indices])
            selected_targets.append(train_targets[selected_indices])

        # Flatten list of arrays
        self.train_data = np.concatenate(selected_paths, axis=0)
        self.train_targets = np.concatenate(selected_targets, axis=0)

        print(f"ImageNet100 (LT) Initialized. Total samples: {len(self.train_data)}")
        # print(f"Class counts: {Counter(self.train_targets)}")

"""
class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-r/train/"
        test_dir = "./data/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
"""

class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.num_classes = 200

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        # 1. Define Class Order
        # Random permutation for 200 classes, or use a fixed seed if reproducibility is needed
        # self.class_order = np.arange(200).tolist() 
        np.random.seed(1993) # Optional: Ensure consistent order across runs
        self.class_order = np.random.permutation(self.num_classes).tolist()

        # 2. Define Class Distribution (Long-Tailed)
        # Assuming a max sample count per class (e.g., 500) and an imbalance ratio (rho)
        # You can use create_exponential_distribution or create_dual_exp_distribution from utils
        # Example: 200 classes, max 500 samples, rho=0.01 (Imbalance Factor 100)
        self.class_distribution = create_exponential_distribution(self.num_classes, 500, 0.01)

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-r/train/"
        test_dir = "./data/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        # Get full data initially
        train_data, train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # Convert to numpy arrays for indexing
        train_data = np.array(train_data)
        train_targets = np.array(train_targets)

        # 3. Apply LT Selection Logic
        selected_data = []
        selected_targets = []
        
        # Verify we have enough samples, otherwise clip to available amount
        real_counts = Counter(train_targets)

        for i, class_index in enumerate(self.class_order):
            target_count = self.class_distribution[i]
            
            # Find indices for this class
            class_indices = np.where(train_targets == class_index)[0]
            
            # Shuffle indices to pick random samples
            np.random.shuffle(class_indices)
            
            # Handle case where real data has fewer samples than distribution requests
            available_count = len(class_indices)
            final_count = min(target_count, available_count)
            
            # Select samples
            selected_indices = class_indices[:final_count]

            selected_data.append(train_data[selected_indices])
            selected_targets.append(train_targets[selected_indices])

        # Concatenate and assign back
        self.train_data = np.concatenate(selected_data, axis=0)
        self.train_targets = np.concatenate(selected_targets, axis=0)
        
        print(f"ImageNet-R LT generated. Total samples: {len(self.train_data)}")
"""
class iImageNetA(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-a/train/"
        test_dir = "./data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
"""
class iImageNetA(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.num_classes = 200

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        # 1. Define Class Order
        # Random permutation for 200 classes, or use a fixed seed if reproducibility is needed
        # self.class_order = np.arange(200).tolist() 
        np.random.seed(1993) # Optional: Ensure consistent order across runs
        self.class_order = np.random.permutation(self.num_classes).tolist()

        # 2. Define Class Distribution (Long-Tailed)
        # Assuming a max sample count per class (e.g., 500) and an imbalance ratio (rho)
        # You can use create_exponential_distribution or create_dual_exp_distribution from utils
        # Example: 200 classes, max 500 samples, rho=0.01 (Imbalance Factor 100)
        self.class_distribution = create_exponential_distribution(self.num_classes, 500, 0.01)

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-a/train/"
        test_dir = "./data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        # Get full data initially
        train_data, train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # Convert to numpy arrays for indexing
        train_data = np.array(train_data)
        train_targets = np.array(train_targets)

        # 3. Apply LT Selection Logic
        selected_data = []
        selected_targets = []
        
        # Verify we have enough samples, otherwise clip to available amount
        real_counts = Counter(train_targets)

        for i, class_index in enumerate(self.class_order):
            target_count = self.class_distribution[i]
            
            # Find indices for this class
            class_indices = np.where(train_targets == class_index)[0]
            
            # Shuffle indices to pick random samples
            np.random.shuffle(class_indices)
            
            # Handle case where real data has fewer samples than distribution requests
            available_count = len(class_indices)
            final_count = min(target_count, available_count)
            
            # Select samples
            selected_indices = class_indices[:final_count]

            selected_data.append(train_data[selected_indices])
            selected_targets.append(train_targets[selected_indices])

        # Concatenate and assign back
        self.train_data = np.concatenate(selected_data, axis=0)
        self.train_targets = np.concatenate(selected_targets, axis=0)
        
        print(f"ImageNet-R LT generated. Total samples: {len(self.train_data)}")

"""
class CUB(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/cub/train/"
        test_dir = "./data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

"""
class CUB(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.num_classes = 200

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        # 1. Define Class Order
        # Random permutation for 200 classes, or use a fixed seed if reproducibility is needed
        # self.class_order = np.arange(200).tolist() 
        np.random.seed(1993) # Optional: Ensure consistent order across runs
        self.class_order = np.random.permutation(self.num_classes).tolist()

        # 2. Define Class Distribution (Long-Tailed)
        # Assuming a max sample count per class (e.g., 500) and an imbalance ratio (rho)
        # You can use create_exponential_distribution or create_dual_exp_distribution from utils
        # Example: 200 classes, max 500 samples, rho=0.01 (Imbalance Factor 100)
        self.class_distribution = create_exponential_distribution(self.num_classes, 500, 0.01)

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/cub/train/"
        test_dir = "./data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        # Get full data initially
        train_data, train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # Convert to numpy arrays for indexing
        train_data = np.array(train_data)
        train_targets = np.array(train_targets)

        # 3. Apply LT Selection Logic
        selected_data = []
        selected_targets = []
        
        # Verify we have enough samples, otherwise clip to available amount
        real_counts = Counter(train_targets)

        for i, class_index in enumerate(self.class_order):
            target_count = self.class_distribution[i]
            
            # Find indices for this class
            class_indices = np.where(train_targets == class_index)[0]
            
            # Shuffle indices to pick random samples
            np.random.shuffle(class_indices)
            
            # Handle case where real data has fewer samples than distribution requests
            available_count = len(class_indices)
            final_count = min(target_count, available_count)
            
            # Select samples
            selected_indices = class_indices[:final_count]

            selected_data.append(train_data[selected_indices])
            selected_targets.append(train_targets[selected_indices])

        # Concatenate and assign back
        self.train_data = np.concatenate(selected_data, axis=0)
        self.train_targets = np.concatenate(selected_targets, axis=0)
        
        
        
class objectnet(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/objectnet/train/"
        test_dir = "./data/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/omnibenchmark/train/"
        test_dir = "./data/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class vtab(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/vtab-cil/vtab/train/"
        test_dir = "./data/vtab-cil/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)




