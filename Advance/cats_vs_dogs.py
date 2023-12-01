import torchvision
from torchvision import transforms
from torch import nn
import torchsummary
import torch
from multiprocessing import Process, freeze_support

#use this dataset for later: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

print("GPU Available:", torch.cuda.is_available())

img_dimensions = 224
batch_size = 512

# Define the transformations to be applied to the images
img_transform = transforms.Compose([
    transforms.Resize((img_dimensions, img_dimensions)),
    transforms.ToTensor()
    ])

# Load the dataset
train_dataset = torchvision.datasets.ImageFolder(
    root="Animals/train/train_data",
    transform=img_transform)

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [20000, 5000])

# Define the dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=True
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=True
)

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(64 * 28 * 28, 512),
    nn.ReLU(),
    nn.Linear(512, 2)
)

def my_function():

    torchsummary.summary(model, (3, img_dimensions, img_dimensions))
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(2):
        for batch in train_dataloader:
            x, y = batch
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f'Epoch {epoch} loss: {loss}')
        correct = 0
        total = 0
        for batch in val_dataloader:
            x, y = batch
            y_hat = model(x)
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print(f'Epoch {epoch} accuracy: {correct / total}')

    # Your multiprocessing logic here
    pass

if __name__ == '__main__':
    freeze_support()  # Call freeze_support() for Windows executable
    # Spawn your processes here
    p = Process(target=my_function)
    p.start()
    p.join()  # Wait for the process to finish
