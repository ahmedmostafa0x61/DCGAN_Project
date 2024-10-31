import torch
from src.models.dcgan import Generator
from src.utils.visualization import show_generated_images

# Load trained model
z_dim = 100
model_path = "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator(z_dim, img_channels=3, features_g=64).to(device)
gen.load_state_dict(torch.load(model_path, map_location=device))
gen.eval()

# Generate images
with torch.no_grad():
    noise = torch.randn(16, z_dim, 1, 1).to(device)
    fake_images = gen(noise)
    show_generated_images(fake_images)