import torch
import torch.nn as nn
import torch.optim as optim
from data.download_dataset import download_cifar10
from src.utils.data_loader import get_dataloader
from src.utils.visualization import show_generated_images
from src.models.dcgan import Generator,Discriminator

# Config..

z_dim = 100
batch_size = 64
lr = 0.0002
num_epochs = 5
img_channels = 3
features_g = 64
features_d = 64


# Load dataset
dataset = download_cifar10()
data_loader = get_dataloader(dataset,batch_size)

# Init the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator(z_dim, img_channels, features_g).to(device)
disc = Discriminator(img_channels, features_d).to(device)
optim_g = optim.Adam(gen.parameters(),lr,(0.5,0.999))
optim_d = optim.Adam(disc.parameters(),lr,(0.5,0.999))
criterion = nn.BCELoss()

# Training the model
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(data_loader):
        real = real.to(device)
        batch_size = real.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)
        disc_real = disc(real).reshape(-1)
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_d = criterion(disc_real, torch.ones_like(disc_real)) + criterion(disc_fake, torch.zeros_like(disc_fake))

        optim_d.zero_grad()
        loss_d.backward()
        optim_d.step()

        # Train Generator
        output = disc(fake).reshape(-1)
        loss_g = criterion(output, torch.ones_like(output))

        optim_g.zero_grad()
        loss_g.backward()
        optim_g.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(data_loader)}], Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")

    # Show some generated images at the end of each epoch
    with torch.no_grad():
        fake = gen(torch.randn(batch_size, z_dim, 1, 1).to(device))
        show_generated_images(fake)