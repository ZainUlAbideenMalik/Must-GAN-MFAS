import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

class StyleSwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7):
        super(StyleSwinTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.msa1 = nn.MultiheadAttention(dim, num_heads)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.msa2 = nn.MultiheadAttention(dim, num_heads)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x, _ = self.msa1(x, x, x)
        x = x_res + x  

        x_res = x
        x = self.norm2(x)
        x = self.mlp1(x)
        x = x_res + x 

        x_res = x
        x = self.norm1(x)
        x, _ = self.msa2(x, x, x)
        x = x_res + x  

        x_res = x
        x = self.norm2(x)
        x = self.mlp2(x)
        x = x_res + x  
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(SwinTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.msa = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x, _ = self.msa(x, x, x)
        x = x_res + x  

        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x_res + x 
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, dim, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dim * 8, 1, kernel_size=4, stride=1, padding=0))
        )

    def forward(self, x):
        return self.layers(x)

def r1_gradient_penalty(real_images, real_scores):
    grad_real = torch.autograd.grad(outputs=real_scores.sum(), inputs=real_images, create_graph=True)[0]
    grad_penalty = torch.sum(grad_real.view(grad_real.size(0), -1) ** 2, dim=1).mean()
    return grad_penalty

class ModalityEncoder(nn.Module):
    def __init__(self, in_channels, num_blocks=6, dim=256):
        super(ModalityEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            StyleSwinTransformerBlock(dim) for _ in range(num_blocks)
        ])

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

class ModalityDecoder(nn.Module):
    def __init__(self, out_channels, num_blocks=6, dim=256):
        super(ModalityDecoder, self).__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim) for _ in range(num_blocks)
        ])
        self.final_layer = nn.Linear(dim, out_channels)

    def forward(self, x, features):
        for block, feature in zip(self.blocks, features):
            x = block(x + feature)
        x = self.final_layer(x)
        return x

class CrossFeatureFusion(nn.Module):
    def __init__(self, num_modalities, dim=256):
        super(CrossFeatureFusion, self).__init__()
        self.num_modalities = num_modalities
        self.dim = dim
        self.fusion_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_modalities)])
            for _ in range(num_modalities)
        ])

    def forward(self, features):
        fused_features = []
        for modality_features in zip(*features):
            modality_fusions = []
            for i in range(self.num_modalities):
                fusion = modality_features[i]
                for j in range(self.num_modalities):
                    if i != j:
                        fusion += self.fusion_layers[i][j](modality_features[j])
                modality_fusions.append(fusion)
            fused_features.append(modality_fusions)
        return list(zip(*fused_features))

class Classifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=2):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class MuSTGAN_MFAS(nn.Module):
    def __init__(self, num_modalities=3, num_blocks=6, dim=256, out_channels=3):
        super(MuSTGAN_MFAS, self).__init__()
        self.encoders = nn.ModuleList([ModalityEncoder(in_channels=3, num_blocks=num_blocks, dim=dim) for _ in range(num_modalities)])
        self.decoders = nn.ModuleList([ModalityDecoder(out_channels=out_channels, num_blocks=num_blocks, dim=dim) for _ in range(num_modalities)])
        self.cross_feature_fusion = CrossFeatureFusion(num_modalities, dim)
        self.classifier = Classifier(input_dim=dim * num_modalities, num_classes=2)

        # Hyperparameters
        self.lambda_t = 1e-4
        self.alpha_params = {
            'alpha1': 0.25,
            'alpha2': 100,
            'alpha3': 1,
            'alpha4': 100,
            'alpha5': 1,
            'alpha6': 10,
            'alpha7': 1,
            'alpha8': 1
        }

        # Loss functions
        self.criterion_identity = nn.L1Loss()
        self.criterion_intensity = nn.MSELoss()
        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion_center = nn.MSELoss()

        self.discriminator = Discriminator()

    def forward(self, inputs):
        encoded_features = [encoder(input) for encoder, input in zip(self.encoders, inputs)]
        fused_features = self.cross_feature_fusion(encoded_features)
        reconstructed_outputs = [decoder(features[-1], features) for decoder, features in zip(self.decoders, fused_features)]
        classification_output = self.classifier(torch.cat([features[-1] for features in fused_features], dim=-1))
        return reconstructed_outputs, classification_output

    def calculate_loss(self, outputs, targets, classification_output, labels, real_images):
        reconstruction_loss = sum(self.criterion_identity(output, target) for output, target in zip(outputs, targets))
        intensity_loss = sum(self.criterion_intensity(output, target) for output, target in zip(outputs, targets))
        classification_loss = self.criterion_classification(classification_output, labels)
        center_loss = sum(self.criterion_center(output, target) for output, target in zip(outputs, targets))

        fake_scores = self.discriminator(outputs[0])
        real_scores = self.discriminator(real_images)
        adversarial_loss = F.softplus(fake_scores).mean() + F.softplus(-real_scores).mean()
        r1_penalty = r1_gradient_penalty(real_images, real_scores)

        total_loss = (self.alpha_params['alpha1'] * reconstruction_loss + 
                      self.alpha_params['alpha2'] * intensity_loss +
                      self.alpha_params['alpha3'] * classification_loss +
                      self.alpha_params['alpha4'] * center_loss +
                      self.alpha_params['alpha5'] * adversarial_loss +
                      self.lambda_t * r1_penalty)

        return total_loss

def get_model_optimizer():
    model = MuSTGAN_MFAS(num_modalities=3)
    generator_optimizer = optim.Adam(model.parameters(), lr=5e-5)
    discriminator_optimizer = optim.Adam(model.discriminator.parameters(), lr=2e-4)
    return model, generator_optimizer, discriminator_optimizer

def train_step(model, inputs, targets, labels, real_images, gen_optimizer, disc_optimizer):
    model.train()
    gen_optimizer.zero_grad()
    disc_optimizer.zero_grad()

    outputs, classification_output = model(inputs)
    loss = model.calculate_loss(outputs, targets, classification_output, labels, real_images)
    loss.backward()
    gen_optimizer.step()
    disc_optimizer.step()

    return loss.item()
if __name__ == "__main__":
    model = MuSTGAN_MFAS(num_modalities=3)
    print(model)
