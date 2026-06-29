import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2

from models.srresnet.model import SRResNet
from models.colorizer.model import ColorizationUNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_tensor(name, tensor):
    print(f"[{name}]")
    if isinstance(tensor, torch.Tensor):
        print(f"  Shape  : {tuple(tensor.shape)}")
        print(f"  DType  : {tensor.dtype}")
        print(f"  Min    : {tensor.min().item():.4f}")
        print(f"  Max    : {tensor.max().item():.4f}")
        print(f"  Mean   : {tensor.mean().item():.4f}")
        print(f"  Device : {tensor.device}")
    elif isinstance(tensor, np.ndarray):
        print(f"  Shape  : {tensor.shape}")
        print(f"  DType  : {tensor.dtype}")
        print(f"  Min    : {tensor.min():.4f}")
        print(f"  Max    : {tensor.max():.4f}")
        print(f"  Mean   : {tensor.mean():.4f}")
        print(f"  Device : CPU (NumPy)")
    print()

class IRPipeline:

    def __init__(self, sr_model_path, color_model_path):
        print("=" * 60)
        print("Loading Models...")
        print("=" * 60)

        try:
            self.sr_model = SRResNet().to(DEVICE)
            self.sr_model.load_state_dict(torch.load(sr_model_path, map_location=DEVICE))
            self.sr_model.eval()
            print("✓ SRResNet Loaded")
        except Exception as e:
            print(f"❌ Failed to load SRResNet: {e}")
            raise e

        try:
            self.color_model = ColorizationUNet().to(DEVICE)
            self.color_model.load_state_dict(torch.load(color_model_path, map_location=DEVICE))
            self.color_model.eval()
            print("✓ Colorization Model Loaded")
        except Exception as e:
            print(f"❌ Failed to load ColorizationUNet: {e}")
            raise e

        print("=" * 60)

    def load_image(self, sample_folder):
        sample_folder = Path(sample_folder)
        try:
            thermal = np.load(sample_folder / "tir_200m.npy")
            thermal = thermal.astype(np.float32)
            log_tensor("Thermal Input (NumPy)", thermal)
            return thermal
        except Exception as e:
            print(f"❌ Error loading thermal image: {e}")
            raise e

    def super_resolve(self, thermal):
        try:
            tensor = torch.from_numpy(thermal).unsqueeze(0).unsqueeze(0).to(DEVICE)
            log_tensor("Thermal Tensor", tensor)
            with torch.no_grad():
                sr = self.sr_model(tensor)
            log_tensor("SR Output Tensor", sr)
            return sr
        except Exception as e:
            print(f"❌ Error during super resolution: {e}")
            raise e

    def colorize(self, sr_tensor):
        try:
            with torch.no_grad():
                rgb = self.color_model(sr_tensor)
            log_tensor("Colorization Output Tensor", rgb)
            return rgb
        except Exception as e:
            print(f"❌ Error during colorization: {e}")
            raise e

    def tensor_to_numpy(self, tensor):
        image = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return image

    def normalize(self, image):
        return np.clip(image, 0.0, 1.0)
        
    def to_uint8(self, img):
        return (self.normalize(img) * 255).astype(np.uint8)

    def save_result(self, rgb_tensor, output_path, sample_folder, brightness=1.0):
        print("\n" + "=" * 60)
        print("Saving Prediction & Generating Outputs")
        print("=" * 60)

        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            rgb_np = self.tensor_to_numpy(rgb_tensor) * brightness
            pred_uint8 = self.to_uint8(rgb_np)
            Image.fromarray(pred_uint8).save(output_path)
            print(f"✅ Saved : {output_path}")

            # Also generate other images
            folder = Path(sample_folder)
            
            # Thermal input
            thermal = np.load(folder / "tir_200m.npy")
            thermal_cmapped = cv2.applyColorMap(self.to_uint8(thermal), cv2.COLORMAP_INFERNO)
            thermal_cmapped = cv2.cvtColor(thermal_cmapped, cv2.COLOR_BGR2RGB)
            thermal_cmapped = cv2.resize(thermal_cmapped, (pred_uint8.shape[1], pred_uint8.shape[0]))
            Image.fromarray(thermal_cmapped).save(out_dir / "thermal_input.png")
            print(f"✅ Saved : {out_dir / 'thermal_input.png'}")
            
            # SR Output preview
            sr = np.load(folder / "tir_100m.npy")
            sr_uint8 = self.to_uint8(sr)
            sr_uint8 = cv2.cvtColor(sr_uint8, cv2.COLOR_GRAY2RGB)
            Image.fromarray(sr_uint8).save(out_dir / "sr_output.png")
            print(f"✅ Saved : {out_dir / 'sr_output.png'}")
            
            # Ground Truth
            try:
                gt = np.load(folder / "rgb.npy")
                gt = gt * brightness
                gt_uint8 = self.to_uint8(gt)
                Image.fromarray(gt_uint8).save(out_dir / "ground_truth.png")
                print(f"✅ Saved : {out_dir / 'ground_truth.png'}")
            except Exception:
                gt_uint8 = np.zeros_like(pred_uint8)
            
            # Comparison
            # Create a large image concatenating them horizontally
            comparison = np.concatenate([thermal_cmapped, sr_uint8, pred_uint8, gt_uint8], axis=1)
            comp_img = Image.fromarray(comparison)
            
            # Add labels if possible
            try:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(comp_img)
                labels = ["Thermal Input", "Super Resolution", "Predicted RGB", "Ground Truth"]
                w = pred_uint8.shape[1]
                for i, label in enumerate(labels):
                    draw.text((i * w + 10, 10), label, fill=(255, 255, 255))
            except Exception:
                pass
                
            comp_path = out_dir / "comparison.png"
            comp_img.save(comp_path)
            print(f"✅ Saved : {comp_path}")
            
        except Exception as e:
            print(f"❌ Error during saving output: {e}")
            raise e

    def run(self, sample_folder, output_image, brightness=1.0):
        print()
        print("Reading Thermal Image...")
        thermal = self.load_image(sample_folder)

        print("Running SRResNet...")
        sr = self.super_resolve(thermal)

        print("Running Colorization...")
        rgb = self.colorize(sr)

        print("Saving Result...")
        self.save_result(rgb, output_image, sample_folder, brightness)

        print()
        print("=" * 60)
        print("Pipeline Finished Successfully!")
        print("=" * 60)