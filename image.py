import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_dim = 100
batch_size = 64
epochs = 5000
lr = 0.0002

# MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize between -1 and 1
])
data_loader = DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Function to show generated image
def show_image(img):
    img = img.view(28, 28).detach().cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

# Training loop
for epoch in range(epochs):
    for real_imgs, _ in data_loader:
        real_imgs = real_imgs.view(-1, 784).to(device)
        valid = torch.ones((real_imgs.size(0), 1), device=device)
        fake = torch.zeros((real_imgs.size(0), 1), device=device)

        # Train Generator
        noise = torch.randn(real_imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(noise)
        loss_G = criterion(discriminator(gen_imgs), valid)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator
        loss_real = criterion(discriminator(real_imgs), valid)
        loss_fake = criterion(discriminator(gen_imgs.detach()), fake)
        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")
        show_image(gen_imgs[0])
