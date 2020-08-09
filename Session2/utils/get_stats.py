import torch
from torch.utils.data import Dataset, DataLoader

def get_avg_mean_std(dataset:Dataset, batch_size:int=50):
    """
    batchwise avg
    """
    print(len(dataset))
    loader = DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=True)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print(mean, std)
    return mean, std


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for images, _ in tqdm(dataloader):
        for i in range(3):
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

# Calculate mean first and then calculate variance and std using the mean - more accurate and time consumsing

def get_mean(dataset:Dataset, batch_size:int=50):
    mean = 0.0
    loader = DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=True)
    for images in loader:
        batch_size = images.size(0) 
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)
    return mean

def get_std(dataset:Dataset, mean:torch.Tensor, batch_size:int=50):
    var = 0.0
    loader = DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=True)
    for images in loader:
        batch_samples = images.size(0)
        # convert into 3 flattened channels
        images = images.view(batch_samples, images.size(1), -1)
        # take mean for each of these channels, substract from image channels
        # square them and add across channels to get variance
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    # square root over total pixels
    std = torch.sqrt(var / (len(dataset)*dataset[0].shape[1]*dataset[0].shape[2]))
    return std
