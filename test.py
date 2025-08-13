
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MTDataset
from generator import Generator
from PIL import Image

exp_name = 'experiment11'
checkpoint_path = os.path.join('./checkpoints', exp_name, 'latest_checkpoint.pth')
checkpoint_path = r"E:\codes\gan-makeuo-models\checkpoints\try2_dataset_top500\latest_checkpoint.pth"
csv_dataroot = r'E:\codes\GAN-i-do-makeup\splits'
image_dataroot = r"E:\datasets\MT\all\images"
output_dir = f'./results_{exp_name}'
os.makedirs(output_dir, exist_ok=True)
max_len_dataset=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = Generator().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
netG.load_state_dict(checkpoint['netG'])
netG.eval()

dataset = MTDataset(csv_dataroot, image_dataroot, phase='test', max_len_dataset=max_len_dataset)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def save_triplet(non_makeup_path, makeup_path, generated_tensor, save_path):
    # Convert generated tensor to image
    gen_img = generated_tensor.clone().detach().cpu()
    gen_img = (gen_img + 1) / 2  # assuming output in [-1, 1]
    gen_img = gen_img.squeeze(0).permute(1, 2, 0)  # CHW -> HWC
    gen_img = (gen_img.numpy() * 255).astype("uint8")
    gen_img_pil = Image.fromarray(gen_img)

    # Load original images
    no_makeup_img = Image.open(non_makeup_path).convert("RGB")
    makeup_img = Image.open(makeup_path).convert("RGB")

    # Resize all to the same size
    w, h = no_makeup_img.size
    makeup_img = makeup_img.resize((w, h))
    gen_img_pil = gen_img_pil.resize((w, h))

    # Create a side-by-side image
    triplet_width = w * 3
    triplet_img = Image.new("RGB", (triplet_width, h))
    triplet_img.paste(no_makeup_img, (0, 0))
    triplet_img.paste(makeup_img, (w, 0))
    triplet_img.paste(gen_img_pil, (2 * w, 0))

    triplet_img.save(save_path)
    print(f"Saved triplet to {save_path}")

with torch.no_grad():
    for i, batch in enumerate(dataloader):
        real_no_makeup = batch['non_makeup'].to(device)
        real_makeup = batch['makeup'].to(device)

        # Single forward pass
        output = netG(no_makeup=real_no_makeup, makeup=real_makeup)
        fake_makeup = output['makeup_output']
        fake_no_makeup = output['No_makeup_output']

        # Paths & names
        no_makeup_path = batch['non_makeup_path'][0]
        makeup_path = batch['makeup_path'][0]
        nm_name = os.path.splitext(os.path.basename(no_makeup_path))[0]
        mk_name = os.path.splitext(os.path.basename(makeup_path))[0]

        # -------------------------
        # Save No-makeup → Makeup
        # -------------------------
        triplet_name_nm2mk = f"{nm_name}_X_{mk_name}_triplet.jpg"
        triplet_path_nm2mk = os.path.join(output_dir, triplet_name_nm2mk)
        save_triplet(no_makeup_path, makeup_path, fake_makeup, triplet_path_nm2mk)

        result_pil_nm2mk = transforms.ToPILImage()(denormalize(fake_makeup.squeeze().cpu()))
        result_name_nm2mk = f"{nm_name}_X_{mk_name}_generated.jpg"
        result_pil_nm2mk.save(os.path.join(output_dir, result_name_nm2mk))

        # -------------------------
        # Save Makeup → No-makeup
        # -------------------------
        triplet_name_mk2nm = f"{mk_name}_X_{nm_name}_triplet.jpg"
        triplet_path_mk2nm = os.path.join(output_dir, triplet_name_mk2nm)
        save_triplet(makeup_path, no_makeup_path, fake_no_makeup, triplet_path_mk2nm)

        result_pil_mk2nm = transforms.ToPILImage()(denormalize(fake_no_makeup.squeeze().cpu()))
        result_name_mk2nm = f"{mk_name}_X_{nm_name}_generated.jpg"
        result_pil_mk2nm.save(os.path.join(output_dir, result_name_mk2nm))

        print(f"[{i+1}/{len(dataloader)}] Saved:")
        print(f"  - No→Makeup: {triplet_name_nm2mk}, {result_name_nm2mk}")
        print(f"  - Makeup→No: {triplet_name_mk2nm}, {result_name_mk2nm}")
