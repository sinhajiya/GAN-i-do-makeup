import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import MTDataset
from generator import Generator
from discriminators import MultiscaleDiscriminator, NLayerDiscriminator
from losses import LSGAN, VGGLoss, MakeUpLoss, CycleConsistencyLoss

# Experiment Setup
exp_name = 'try1'
save_dir = os.path.join('./checkpoints', exp_name)
os.makedirs(save_dir, exist_ok=True)

# Configurations
epochs = 200
batch_size = 1
lr = 2e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss Weights
alpha = 1
beta = 10
gamma = 0.005
lambda_lips = 1
lambda_skin = 1
lambda_face = 0.1
cycle_loss_metric = 'l1'

# Paths
disc = 'patchgan'
csv_dataroot = r'/content/GAN-i-do-makeup/splits'
image_dataroot = r'/content/GAN-i-do-makeup/dataset/all/images'
phase = 'train'

# Save config
with open(os.path.join(save_dir, 'training_config.txt'), 'w') as f:
    f.write(f"""Training Configuration:
-----------------------
save_dir: {save_dir}
csv_dataroot: {csv_dataroot}
image_dataroot: {image_dataroot}
phase: {phase}
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
""")

# Dataset & DataLoader
dataset = MTDataset(csv_dataroot, image_dataroot, phase)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Models
netG = Generator().to(device)
netDA = NLayerDiscriminator().to(device) if disc == 'patchgan' else MultiscaleDiscriminator().to(device)
netDB = NLayerDiscriminator().to(device) if disc == 'patchgan' else MultiscaleDiscriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_DA = torch.optim.Adam(netDA.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_DB = torch.optim.Adam(netDB.parameters(), lr=lr, betas=(0.5, 0.999))

# Losses
adv_loss = LSGAN().to(device)
vgg_loss = VGGLoss().to(device)
make_up_loss = MakeUpLoss().to(device)
cycle_loss = CycleConsistencyLoss().to(device)

# Training Loop
start_time = time.time()
for epoch in range(epochs):
    print(f"\nEpoch [{epoch + 1}/{epochs}]")
    
    for i, data in enumerate(dataloader):
        real_no_makeup = data['non_makeup'].to(device)
        real_makeup = data['makeup'].to(device)

        # gen
        gen1 = netG(no_makeup=real_no_makeup, makeup=real_makeup)
        gen1_no_makeup = gen1['No_makeup_output']
        gen1_makeup = gen1['makeup_output']

        with torch.no_grad():
            gen2 = netG(no_makeup=gen1_no_makeup.detach(), makeup=gen1_makeup.detach())
            gen2_no_makeup = gen2['No_makeup_output']
            gen2_makeup = gen2['makeup_output']

        # disc A
        optimizer_DA.zero_grad()
        loss_DA_real = adv_loss(netDA(real_no_makeup), True)
        loss_DA_fake = adv_loss(netDA(gen1_no_makeup.detach()), False)
        loss_DA = 0.5 * (loss_DA_real + loss_DA_fake)
        loss_DA.backward()
        optimizer_DA.step()

        # disc B
        optimizer_DB.zero_grad()
        loss_DB_real = adv_loss(netDB(real_makeup), True)
        loss_DB_fake = adv_loss(netDB(gen1_makeup.detach()), False)
        loss_DB = 0.5 * (loss_DB_real + loss_DB_fake)
        loss_DB.backward()
        optimizer_DB.step()

        # loss disc
        L_adv = adv_loss(netDA(gen1_no_makeup), True) + adv_loss(netDB(gen1_makeup), True)
        L_per = vgg_loss(
            gen_non_makeup=gen1_no_makeup,
            non_makeup=real_no_makeup,
            gen_makeup=gen1_makeup,
            makeup=real_makeup
        )
        L_cycle = cycle_loss(
            rec_non_makeup=gen2_no_makeup,
            non_makeup=real_no_makeup,
            rec_makeup=gen2_makeup,
            makeup=real_makeup
        )
        L_makeup = make_up_loss(
            gen_makeup=gen1_makeup,
            real_makeup=real_makeup
        )

        total_loss_G = alpha * L_adv + beta * L_cycle + gamma * L_per + L_makeup

        optimizer_G.zero_grad()
        total_loss_G.backward()  
        optimizer_G.step()

        # log
        if i % 10 == 0:
            print(f"[{i}/{len(dataloader)}] "
                  f"Loss_G: {total_loss_G.item():.4f} | "
                  f"Loss_DA: {loss_DA.item():.4f} | Loss_DB: {loss_DB.item():.4f}")

    # save every 10 models
    if (epoch + 1) % 10 == 0:
        ckpt_path = os.path.join(save_dir, f'netG_epoch{epoch+1}.pth')
        torch.save(netG.state_dict(), ckpt_path)
        print(f"Saved checkpoint to: {ckpt_path}")


elapsed = time.time() - start_time
print(f"\nTraining complete! Total time: {elapsed/60:.2f} minutes")
