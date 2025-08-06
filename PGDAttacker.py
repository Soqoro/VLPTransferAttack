import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityTextAttacker:
    """
    Dummy text attacker that returns the input texts unchanged.
    Used for image-only attacks like PGD.
    """
    def attack(self, model, texts):
        return texts


class PGDAttacker:
    def __init__(self, model, normalization, eps=8 / 255, steps=10, step_size=2 / 255):
        self.model = model
        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size

    def attack(self, images, device):
        images = images.detach().to(device)
        adv_images = images.clone() + torch.empty_like(images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(self.steps):
            adv_images.requires_grad_(True)
            normed = self.normalization(adv_images)
            output = self.model.inference_image(normed)
            features = output['image_feat']

            # Simple loss: maximize self-contrastiveness (to diverge from clean features)
            loss = -features.norm(dim=-1).mean()
            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data
            adv_images = adv_images + self.step_size * grad.sign()
            adv_images = torch.max(torch.min(adv_images, images + self.eps), images - self.eps)
            adv_images = torch.clamp(adv_images, 0.0, 1.0)

        return adv_images


class Attacker:
    """
    PGD image attacker wrapper compatible with eval.py.
    """
    def __init__(self, model, img_attacker, txt_attacker=None):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker if txt_attacker is not None else IdentityTextAttacker()

    def attack(self, imgs, txts, txt2img, device='cuda', **kwargs):
        adv_imgs = self.img_attacker.attack(imgs, device)
        adv_txts = self.txt_attacker.attack(self.model, txts)
        return adv_imgs, adv_txts, 0.0
