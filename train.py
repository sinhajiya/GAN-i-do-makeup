import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from visdom import Visdom
from dataset import MTDataset
from generator import Generator
from discriminators import MultiscaleDiscriminator, NLayerDiscriminator
from losses import LSGAN, VGGLoss, MakeUpLoss, CycleConsistencyLoss
import torch.nn as nn

def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train():
    exp_name = 'experiment1'
    save_dir = os.path.join('./checkpoints', exp_name)
    os.makedirs(save_dir, exist_ok=True)
    max_len_dataset = 400
    
    viz = Visdom()
    assert viz.check_connection(), "Start Visdom server: python -m visdom.server"

    win_G = viz.line(X=[0], Y=[0], opts=dict(title='Generator Loss'))
    win_DA = viz.line(X=[0], Y=[0], opts=dict(title='Discriminator A Loss'))
    win_DB = viz.line(X=[0], Y=[0], opts=dict(title='Discriminator B Loss'))
    win_img = None

    epochs = 200
    batch_size = 8
    lr = 2e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alpha, beta, gamma = 1, 10, 0.005

    disc = 'patchgan'
    csv_dataroot = r'E:\codes\GAN-i-do-makeup\splits'
    image_dataroot = r"E:\datasets\MT\all\images"
    phase = 'train'

    dataset = MTDataset(csv_dataroot, image_dataroot, phase,max_len_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    netG = Generator().to(device)
    netDA = NLayerDiscriminator().to(device) if disc == 'patchgan' else MultiscaleDiscriminator().to(device)
    netDB = NLayerDiscriminator().to(device) if disc == 'patchgan' else MultiscaleDiscriminator().to(device)

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_DA = torch.optim.Adam(netDA.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_DB = torch.optim.Adam(netDB.parameters(), lr=lr, betas=(0.5, 0.999))

    start_epoch = 0
    checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        netG.load_state_dict(checkpoint['netG'])
        netDA.load_state_dict(checkpoint['netDA'])
        netDB.load_state_dict(checkpoint['netDB'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_DA.load_state_dict(checkpoint['optimizer_DA'])
        optimizer_DB.load_state_dict(checkpoint['optimizer_DB'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting from scratch.")
        netG.apply(init_weights_xavier)
        netDA.apply(init_weights_xavier)
        netDB.apply(init_weights_xavier)
        with open(os.path.join(save_dir, 'training_config.txt'), 'w') as f:
            f.write(f"Training config:\nEpochs: {epochs}, LR: {lr}, Batch: {batch_size}, Device: {device}")

    adv_loss = LSGAN().to(device)
    vgg_loss = VGGLoss().to(device)
    make_up_loss = MakeUpLoss().to(device)
    cycle_loss = CycleConsistencyLoss().to(device)

    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")

        for i, data in enumerate(dataloader):
            real_no_makeup = data['non_makeup'].to(device)
            real_makeup = data['makeup'].to(device)

            gen1 = netG(no_makeup=real_no_makeup, makeup=real_makeup)
            gen1_no_makeup = gen1['No_makeup_output']
            gen1_makeup = gen1['makeup_output']

            with torch.no_grad():
                gen2 = netG(no_makeup=gen1_no_makeup, makeup=gen1_makeup)
                gen2_no_makeup = gen2['No_makeup_output']
                gen2_makeup = gen2['makeup_output']

            optimizer_DA.zero_grad()
            loss_DA_real = adv_loss(netDA(real_no_makeup), True)
            loss_DA_fake = adv_loss(netDA(gen1_no_makeup.detach()), False)
            loss_DA = 0.5 * (loss_DA_real + loss_DA_fake)
            loss_DA.backward()
            optimizer_DA.step()

            optimizer_DB.zero_grad()
            loss_DB_real = adv_loss(netDB(real_makeup), True)
            loss_DB_fake = adv_loss(netDB(gen1_makeup.detach()), False)
            loss_DB = 0.5 * (loss_DB_real + loss_DB_fake)
            loss_DB.backward()
            optimizer_DB.step()

            L_adv = adv_loss(netDA(gen1_no_makeup), True) + adv_loss(netDB(gen1_makeup), True)
            L_per = vgg_loss(gen1_no_makeup, real_no_makeup, gen1_makeup, real_makeup)
            L_cycle = cycle_loss(gen2_no_makeup, real_no_makeup, gen2_makeup, real_makeup)
            L_makeup = make_up_loss(gen1_makeup, real_makeup)

            total_loss_G = alpha * L_adv + beta * L_cycle + gamma * L_per + L_makeup

            optimizer_G.zero_grad()
            total_loss_G.backward()
            optimizer_G.step()

            step = epoch * len(dataloader) + i + 1
            viz.line(X=[step], Y=[total_loss_G.item()], win=win_G, update='append')
            viz.line(X=[step], Y=[loss_DA.item()], win=win_DA, update='append')
            viz.line(X=[step], Y=[loss_DB.item()], win=win_DB, update='append')

            if i % 10 == 0:
                print(f"[{i}/{len(dataloader)}] "
                      f"Loss_G: {total_loss_G.item():.4f} | "
                      f"L_adv: {L_adv.item():.4f} | "
                      f"L_cycle: {L_cycle.item():.4f} | "
                      f"L_per: {L_per.item():.4f} | "
                      f"L_makeup: {L_makeup:.4f} | "
                      f"Loss_DA: {loss_DA.item():.4f} | "
                      f"Loss_DB: {loss_DB.item():.4f}")

                grid = make_grid(torch.cat([
                    real_no_makeup, real_makeup, gen1_no_makeup, gen1_makeup
                ], dim=0), nrow=4, normalize=True, scale_each=True)
                win_img = viz.image(grid, win=win_img, opts=dict(title='[Real NoMakeup, Real Makeup, Fake NoMakeup, Fake Makeup]'))

        # Save checkpoint only once per even-numbered epoch
        if (epoch + 1) % 2 == 0:
            ckpt = {
                'epoch': epoch + 1,
                'netG': netG.state_dict(),
                'netDA': netDA.state_dict(),
                'netDB': netDB.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_DA': optimizer_DA.state_dict(),
                'optimizer_DB': optimizer_DB.state_dict(),
            }
            torch.save(ckpt, checkpoint_path)
            print(f"Saved checkpoint to: {checkpoint_path} at epoch {epoch + 1}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete! Total time: {elapsed / 60:.2f} minutes")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train()
