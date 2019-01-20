import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import agtree2dot
import subprocess

# getting the data from torchvision FashionMNIST, the train_set are an iterable object.
train_set = torchvision.datasets.FashionMNIST(
root = './data/fashionMNIST',
download= True,
train = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

batch_size = 2048
train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size)

def show_batch(batch):

    grid = torchvision.utils.make_grid(batch.detach(),nrow=25)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()


class Autoencoder (nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(

            nn.Linear(in_features=28*28,out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200,out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100,out_features=50)
        )

        self.decoder = nn.Sequential(

            nn.Linear(in_features=50,out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100,out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200,out_features=28*28),
            nn.Sigmoid()
        )

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)

        return decoded


autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=.01)

no_of_epochs = 1

for epoch in range (no_of_epochs):
    for batch, batch_train_data in enumerate (train_loader):
        images ,labels = batch_train_data
        images = images.view(-1,28*28)

        output = autoencoder(images)
        loss = criterion(output, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'epoch, {epoch}, loss = {loss.item()}')

counter =0
for i in range (600):
    counter +=1
print(counter)

sample = next(iter(train_loader))
sample_images ,_ = sample
samples = sample_images[:8]

output = autoencoder(samples.view(8,28*28))
print(samples.shape)
print(output.view(8,1,28,28).shape)


show_batch(output.view(8,1,28,28))

agtree2dot.save_dot(loss,
                    {
                        input: 'input',
                        output: 'output',
                        loss: 'loss',
                    },
                    open('./mlp.dot', 'w'))

print('Generated mlp.dot')

try:
    subprocess.check_call(['dot', 'mlp.dot', '-Lg', '-T', 'pdf', '-o', 'mlp.pdf' ])

except subprocess.CalledProcessError:
    print('Calling the dot command failed. Is Graphviz installed?')
    sys.exit(1)

print('Generated mlp.pdf')