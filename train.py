import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from dataset import MTDataset
from generator import Generator
from discriminators import MultiscaleDiscriminator, NLayerDiscriminator
from losses import LSGAN, VGGLoss, MakeUpLoss, CycleConsistencyLoss

exp_name = 'try1'
save_dir = os.path.join('./checkpoints', exp_name)
os.makedirs(save_dir, exist_ok=True)

# Configs
A_dir = './data/no_makeup'
B_dir = './data/makeup'
A_seg_dir = './data/no_makeup_segs'
B_seg_dir = './data/makeup_segs'

epochs = 200
batch_size = 1
lr = 2e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 1
beta = 10
gamma = 0.005
lambda_lips = 1
lambda_skin = 1
lambda_face = 0.1
cycle_loss_metric = 'l1'
disc = 'patchgan'
csv_dataroot=r'E:\codes\GAN-i-do-makeup\splits'
image_dataroot=r"E:\datasets\MT\all"
phase='train'

# Save training config
config_str = f"""Training Configuration:
-----------------------
A_dir: {A_dir}
B_dir: {B_dir}
A_seg_dir: {A_seg_dir}
B_seg_dir: {B_seg_dir}
save_dir: {save_dir}
csv_dataroot: {csv_dataroot}
image_dataroot: {image_dataroot}
phase: Train
epochs: {epochs}
batch_size: {batch_size}
learning_rate: {lr}
device: {device}
discriminator: {disc}
cycle_loss_metric: {cycle_loss_metric}

Loss Weights:
alpha (GAN loss): {alpha}
beta (Cycle consistency): {beta}
gamma (Perceptual/VGG): {gamma}
lambda_lips: {lambda_lips}
lambda_skin: {lambda_skin}
lambda_face: {lambda_face}
"""

with open(os.path.join(save_dir, 'training_config.txt'), 'w') as f:
    f.write(config_str)

# Dataset
dataset = MTDataset(csv_dataroot, image_dataroot, phase)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Models
netG = Generator().to(device)
if disc == 'patchgan':
    netDA = NLayerDiscriminator().to(device)
    netDB = NLayerDiscriminator().to(device)
elif disc == 'MSD':
    netDA = MultiscaleDiscriminator().to(device)
    netDB = MultiscaleDiscriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_DA = torch.optim.Adam(netDA.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_DB = torch.optim.Adam(netDB.parameters(), lr=lr, betas=(0.5, 0.999))

# Losses
adv_loss = LSGAN().to(device)
vgg_loss = VGGLoss().to(device)
make_up_loss = MakeUpLoss().to(device)
cycle_loss = CycleConsistencyLoss().to(device)

# Training
for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        real_no_makeup = data['non_makeup'].to(device)
        real_makeup = data['makeup'].to(device)

        # Generator forward
        gen1 = netG(no_makeup=real_no_makeup, makeup=real_makeup)
        gen1_no_makeup = gen1['No_makeup_output']
        gen1_makeup = gen1['makeup_output']
        
        gen2 = netG(no_makeup=gen1_no_makeup, makeup=gen1_makeup)
        gen2_no_makeup = gen2['No_makeup_output']
        gen2_makeup = gen2['makeup_output']
        
        # Discriminator A forward
        loss_DA_real = adv_loss(netDA(real_no_makeup), True)
        loss_DA_fake = adv_loss(netDA(gen1_no_makeup.detach()), False)
        loss_DA = (loss_DA_real + loss_DA_fake) * 0.5

        # Discriminator B forward
        loss_DB_real = adv_loss(netDB(real_makeup), True)
        loss_DB_fake = adv_loss(netDB(gen1_makeup.detach()), False)
        loss_DB = (loss_DB_real + loss_DB_fake) * 0.5

        optimizer_DA.zero_grad()
        loss_DA.backward()
        optimizer_DA.step()

        optimizer_DB.zero_grad()
        loss_DB.backward()
        optimizer_DB.step()

        L_adv = loss_DA + loss_DB
        L_per = vgg_loss(gen_non_makeup=gen1_no_makeup, non_makeup=real_no_makeup, gen_makeup=gen1_makeup, makeup=real_makeup)
        L_cycle = cycle_loss(rec_non_makeup = gen2_no_makeup, non_makeup=real_no_makeup, rec_makeup=gen2_makeup, makeup=real_makeup)
        L_makeup = make_up_loss(gen_makeup=gen1_makeup, real_makeup=real_makeup)

        loss = alpha * L_adv + beta * L_cycle + gamma * L_per + L_makeup

        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()

        if i % 10 == 0:
            print(f"[{epoch}/{epochs}] [{i}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")

    if epoch % 10 == 0:
        save_path = os.path.join(save_dir, f'netG_epoch{epoch}.pth')
        torch.save(netG.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")
