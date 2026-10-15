import torch

def test_dataloader(dataloader, num_batches_to_check = 5):
    """
        Function to test the working of pytorch dataloader
    """
    cuda_device_index = 1
    device = torch.device(f"cuda:{cuda_device_index}" if torch.cuda.is_available() else "cpu")
    print(f"Testing Dataloader with {num_batches_to_check} batches ... \n")
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches_to_check:
            break

        images, labels = images.to(device), labels.to(device)
        print(f"Batch {i+1} \n")
        print(f" - Image batch shape: {images.shape}\n") #should be (batch_size, channels, height, width)
        print(f" - Labels batch shape: {labels.shape}, {len(images)} \n")

        #check if the images are tensors and have the right shape
        assert isinstance(images, torch.Tensor), f"Labels should be of type Tensor \n"
        assert len(images.shape) == 4, f"Expected 4D shape for images, but got {images.shape}\n"
        assert images.shape[1] == 3, f"Expected 3 channel (RGB), but got {images.shape[1]}\n"
        #check if the labels are tensors and of right shape
        assert isinstance(labels, torch.Tensor), f"Labels should be of type Tensor \n"
        assert len(labels.shape) == 1, f"Expected 1D shape for labels, but got {labels.shape}\n"
        assert images.shape[0] == labels.shape[0], f"Number of images must match number of labels \n\n"

    print(f"Dataloader testing completed successfully")