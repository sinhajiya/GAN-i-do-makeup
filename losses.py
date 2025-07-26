import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import torchvision.models as models
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torchvision.transforms as transforms

class LSGAN(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label).float())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).float())

        self.loss = nn.MSELoss()

    def get_target_tensot(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)
    
    def forward(self, predictions, target_is_real):
        if isinstance(predictions,list):
            loss = 0
            for pred in predictions:
                if isinstance(pred, list):
                    pred = pred[-1]

                target_tensor = self.get_target_tensot(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
        else:
            target_tensor = self.get_target_tensot(predictions, target_is_real)
            return self.loss(predictions,target_tensor)

class CycleConsistencyLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'charbonnier':
            self.loss_fn = lambda x, y: torch.mean(torch.sqrt((x - y) ** 2 + 1e-6))
        else:
            raise ValueError("Invalid loss_type. Use 'l1', 'l2', or 'charbonnier'.")

    def forward(self, rec_non_makeup, non_makeup, rec_makeup, makeup):
        loss_non_makeup = self.loss_fn(rec_non_makeup, non_makeup)
        loss_makeup = self.loss_fn(rec_makeup, makeup)
        return loss_non_makeup + loss_makeup

class VGGLoss(nn.Module):
    def __init__(self, vgg_normal_correct=True):
        super().__init__()
        self.vgg_normal_correct = vgg_normal_correct
        if vgg_normal_correct:
            self.vgg = VGG19_feature(vgg_normal_correct=True)
        else:
            self.vgg = VGG19_feature()
        self.loss = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, gen_non_makeup, non_makeup, gen_makeup, makeup):
   
        gen_non_makeup_vgg = self.vgg(gen_non_makeup, ['r11', 'r21', 'r31', 'r41', 'r51'], preprocess=True) 
        non_makeup_vgg = self.vgg(non_makeup, ['r11', 'r21','r31', 'r41','r51'], preprocess=True)
        gen_makeup_vgg = self.vgg(gen_makeup, ['r11', 'r21', 'r31', 'r41', 'r51'], preprocess=True) 
        makeup_vgg = self.vgg(makeup, ['r11', 'r21','r31', 'r41','r51'], preprocess=True)                                                                                      
      
        loss_value = 0
        for i in range(len(makeup_vgg)):
            loss_value += self.weights[i] * (
                self.loss(gen_non_makeup_vgg[i], makeup_vgg[i].detach()) +
                self.loss(gen_makeup_vgg[i], non_makeup_vgg[i].detach())
            )

        return loss_value

class VGG19_feature(nn.Module):
    def __init__(self, vgg_normal_correct=True):
        super().__init__()
        vgg_pretrained = models.vgg19(pretrained=True).features
        self.vgg_slices = nn.ModuleList([
            vgg_pretrained[:2],   # r11
            vgg_pretrained[2:7],  # r21
            vgg_pretrained[7:12], # r31
            vgg_pretrained[12:21],# r41
            vgg_pretrained[21:30] # r51
        ])
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()
        self.vgg_normal_correct = vgg_normal_correct
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, layers=None, preprocess=False):
        if preprocess:
            x = (x + 1) / 2  # convert from [-1,1] to [0,1]
            x = (x - self.mean) / self.std  # normalize to ImageNet

        features = []
        for i, slice in enumerate(self.vgg_slices):
            x = slice(x)
            features.append(x)

        if layers:
            layer_idx = {'r11': 0, 'r21': 1, 'r31': 2, 'r41': 3, 'r51': 4}
            return [features[layer_idx[l]] for l in layers]
        return features

class MakeUpLoss(nn.Module):
    def __init__(self, cluster_number=32):
        super().__init__()
        self.cluster_number = cluster_number
        self.loss = nn.L1Loss()
        self.spacing = 2.0 / cluster_number  

        self.processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.parser = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.parser.eval()
        self.parser.to("cuda" if torch.cuda.is_available() else "cpu")
        self.to_pil = transforms.Compose([
            transforms.Normalize([-1, -1, -1], [2, 2, 2]),  # [-1,1] → [0,1]
            transforms.ToPILImage()
        ])

    @torch.no_grad()
    def get_parsing_mask(self, img_tensor): 
        B = img_tensor.shape[0]
        masks = []

        for i in range(B):
            img = img_tensor[i].detach().cpu()
            img_pil = self.to_pil(img)
            inputs = self.processor(images=img_pil, return_tensors="pt").to(img_tensor.device)
            with torch.no_grad():
                out = self.parser(**inputs)
            logits = out.logits  # (1, 19, H/4, W/4)
            upsampled = F.interpolate(logits, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)
            labels = upsampled.argmax(dim=1)  # (1, H, W)
            masks.append(labels)
        return torch.cat(masks, dim=0)  # (B, H, W)

    def calc_hist(self, data):
    
        N = data.size(0)
        grid = torch.linspace(-1, 1, self.cluster_number + 1, device=data.device).view(-1, 1)  
        data = data.view(1, -1)  
        hist = torch.clamp(self.spacing - torch.abs(grid - data), min=0.0) * 10
        hist = hist / hist.sum()
        return hist.mean(dim=1) 

    def generate_masks(self, seg_mask_tensor):
        B, H, W = seg_mask_tensor.shape
        lips_mask = (seg_mask_tensor == 11) | (seg_mask_tensor == 12)
        face_mask = seg_mask_tensor == 1
        eyes_mask = (seg_mask_tensor == 4) | (seg_mask_tensor == 5)
        eye_shadow_mask = torch.zeros_like(seg_mask_tensor, dtype=torch.float32)

        for i in range(B):
            eye_coords = (eyes_mask[i]).nonzero(as_tuple=False)
            if eye_coords.numel() > 0:
                y_min = eye_coords[:, 0].min().item()
                y_max = eye_coords[:, 0].max().item()
                x_min = eye_coords[:, 1].min().item()
                x_max = eye_coords[:, 1].max().item()

                pad_y = int(0.15 * (y_max - y_min))
                y_min = max(0, y_min - pad_y)
                y_max = min(H, y_max + pad_y)

                pad_x = int(0.15 * (x_max - x_min))
                x_min = max(0, x_min - pad_x)
                x_max = min(W, x_max + pad_x)
                y_max = min(H - 1, y_max + pad_y)
                x_max = min(W - 1, x_max + pad_x)
                eye_shadow_mask[i, y_min:y_max, x_min:x_max] = 1.0

        return (
            lips_mask.unsqueeze(1).float(),
            face_mask.unsqueeze(1).float(),
            eye_shadow_mask.unsqueeze(1)
        )

    def forward(self, gen_makeup, real_makeup):

        gen_makeup_mask = self.get_parsing_mask(gen_makeup)
        real_makeup_mask = self.get_parsing_mask(real_makeup)
        lips_gen, face_gen, eyes_gen = self.generate_masks(gen_makeup_mask)
        lips_ref, face_ref, eyes_ref = self.generate_masks(real_makeup_mask)
        

        region_names = ['lips', 'face', 'eyes']
        gen_masks = [lips_gen, face_gen, eyes_gen]
        ref_masks = [lips_ref, face_ref, eyes_ref]

        B, C, H, W = gen_makeup.shape
        total_loss = 0.0

        for name, mask_gen, mask_ref in zip(region_names, gen_masks, ref_masks):
            for b in range(B):
                for c in range(C):
                    region_gen = gen_makeup[b, c][mask_gen[b, 0] > 0.5]
                    region_ref = real_makeup[b, c][mask_ref[b, 0] > 0.5]
                    if region_gen.numel() == 0 or region_ref.numel() == 0:
                        continue
                    hist_gen = self.calc_hist(region_gen)
                    hist_ref = self.calc_hist(region_ref)
                    
                    if name == 'face':
                        total_loss += 0.1 * self.loss(hist_gen, hist_ref)  
                    else:
                        total_loss += self.loss(hist_gen, hist_ref)

        return total_loss / (B * C * len(region_names))
