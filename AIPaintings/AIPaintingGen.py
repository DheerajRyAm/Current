import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import glob
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')
# Path to the dataset
data_path = '/content/drive/MyDrive/images'

# Define transform before using it
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label, sub_dir in enumerate(os.listdir(root_dir)):
            sub_dir_path = os.path.join(root_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                for img_path in glob.glob(os.path.join(sub_dir_path, '*.jpg')) + glob.glob(os.path.join(sub_dir_path, '*.png')):
                    self.image_paths.append(img_path)
                    self.labels.append(label)
        print(f"Total images: {len(self.image_paths)}")  # Debug print
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# Create the dataset and dataloader after defining the transform
dataset = CustomDataset(root_dir=data_path, transform=transform)
print(f"Dataset length: {len(dataset)}")  # Debug print

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn)

# Check if DataLoader is providing batches
for batch_idx, (real, labels) in enumerate(loader):
    print(f"Batch {batch_idx}: Real images shape: {real.shape}, Labels shape: {labels.shape}")
    break

class Discriminator(nn.Module):
    def __init__(self, img_channels, num_classes):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, 64 * 64)
        self.model = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0),
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels).view(labels.size(0), 1, 64, 64)
        x = torch.cat([x, c], dim=1)
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, num_classes):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, z_dim)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + z_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, c], dim=1)
        return self.model(x)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def gradient_penalty(disc, real, fake, labels, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = disc(interpolated_images, labels)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def find_latest_checkpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, '*.pth'))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 100
image_dim = 64 * 64
num_classes = len(os.listdir(data_path))  # This will automatically set num_classes
batch_size = 64
epochs = 2001

disc = Discriminator(img_channels=3, num_classes=num_classes).to(device)
gen = Generator(z_dim, img_channels=3, num_classes=num_classes).to(device)

disc.apply(weights_init)
gen.apply(weights_init)

fixed_noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
fixed_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

opt_disc = optim.RMSprop(disc.parameters(), lr=5e-5)
opt_gen = optim.RMSprop(gen.parameters(), lr=5e-5)

writer_fake = SummaryWriter(f"runs/WGAN_GP_Artwork/fake")
writer_real = SummaryWriter(f"runs/WGAN_GP_Artwork/real")
writer_loss = SummaryWriter(f"runs/WGAN_GP_Artwork/loss")
step = 0
lambda_gp = 10

start_epoch = 0

model_dir = "/content/drive/MyDrive/models art"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

latest_checkpoint = find_latest_checkpoint(model_dir)
if latest_checkpoint:
    print(f"Loading checkpoint {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=torch.device(device))
    gen.load_state_dict(checkpoint['generator_state_dict'])
    disc.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
    opt_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    step = checkpoint.get('step', 0)

save_interval = 3

for epoch in range(start_epoch, epochs):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, labels) in enumerate(loop):
        real = real.to(device)
        labels = labels.to(device)
        cur_batch_size = real.size(0)


        for _ in range(5):
            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise, labels)

            disc_real = disc(real, labels).view(-1)
            disc_fake = disc(fake, labels).view(-1)
            gp = gradient_penalty(disc, real, fake, labels, device=device)
            lossD = -(torch.mean(disc_real) - torch.mean(disc_fake)) + lambda_gp * gp

            opt_disc.zero_grad()
            lossD.backward()
            opt_disc.step()

        noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise, labels)

        disc_fake = disc(fake, labels).view(-1)
        lossG = -torch.mean(disc_fake)

        opt_gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        loop.set_postfix(lossD=lossD.item(), lossG=lossG.item())

    if epoch % save_interval == 0:
       torch.save({
                  'epoch': epoch,
                  'step': step,
                  'generator_state_dict': gen.state_dict(),
                  'discriminator_state_dict': disc.state_dict(),
                  'optimizer_gen_state_dict': opt_gen.state_dict(),
                  'optimizer_disc_state_dict': opt_disc.state_dict(),
                  'lossG': lossG.item(),
                  'lossD': lossD.item(),
                  }, os.path.join(model_dir, f'epoch_{epoch}_model.pth'))


    with torch.no_grad():
        fake = gen(fixed_noise, fixed_labels).detach().cpu()
        real_images = real.detach().cpu()

        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(real_images, normalize=True)

        writer_fake.add_image("Artwork Fake Images", img_grid_fake, global_step=step)
        writer_real.add_image("Artwork Real Images", img_grid_real, global_step=step)
        writer_loss.add_scalars('Loss', {'Generator': lossG, 'Discriminator': lossD}, epoch)
        step += 1

        # Display the images
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img_grid_fake.permute(1, 2, 0))
        ax[0].set_title("Fake Images")
        ax[0].axis("off")

        ax[1].imshow(img_grid_real.permute(1, 2, 0))
        ax[1].set_title("Real Images")
        ax[1].axis("off")

        plt.show()

total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")

writer_fake.close()
writer_real.close()
writer_loss.close()
