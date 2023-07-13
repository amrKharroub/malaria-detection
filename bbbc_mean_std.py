
import requests
import zipfile
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm


url = 'https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip'
r = requests.get(url, allow_redirects=True)

open('malaria.zip', 'wb').write(r.content)

with zipfile.ZipFile("/content/malaria.zip", 'r') as zObject:
  zObject.extractall(path="/content/malaria")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = datasets.ImageFolder(
    root="/malaria/Malaria/", transform=transforms.Compose([
                                            transforms.Resize((1200, 1600)),
                                            transforms.ToTensor(),
                                              ]),
)
train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean**2) ** 0.5

    return mean, std


mean, std = get_mean_std(train_loader)
print(mean)
print(std)

