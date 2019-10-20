import torchvision
import os
from pathlib import Path

transformations = torchvision.transforms.Compose([
    # you can add other transformations in this list
    torchvision.transforms.Resize(size=(800, 800)),
    torchvision.transforms.ToTensor()
])

test_dataset = torchvision.datasets.ImageFolder(
    root="/media/disk1/STUFF/eddible_ones/data/datasets/dataset-test/",
    transform=transformations)

# get the current folder
path = os.getcwd()
#print("The current working directory is %s" % path)

for data in test_dataset:
    image, label = data
    image_path = print(os.path.join(path, str(label)))
    #print(image_path)
    dirName = Path(image_path)
    #print(dirName.is_dir())
    '''
    if dirName.is_dir():
        try:
            os.mkdir(image_path)
            print ("Successfully created the directory %s " % image_path)
        except OSError:
            print("Creation of the directory %s failed" % image_path)
    '''
i = 0
#path = os.path.join(path, str(i))
