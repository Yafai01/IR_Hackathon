import os
import cv2
import numpy as np
import tifffile
from pathlib import Path


class DatasetGenerator:

    def __init__(
        self,
        input_path,
        output_dir="datasets/processed",
        patch_size=512,
        stride=256,
        scale_factor=2
    ):

        self.input_path = input_path
        self.output_dir = output_dir

        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = scale_factor

        self.image = None
        self.rgb = None
        self.thermal = None

    ########################################################

    def load(self):

        print("=" * 50)
        print("Loading TIFF...")

        self.image = tifffile.imread(self.input_path)

        print("Shape :", self.image.shape)
        print("Type  :", self.image.dtype)

    ########################################################

    def split_bands(self):

        print("=" * 50)
        print("Splitting Bands...")

        blue = self.image[:, :, 0]
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]
        thermal = self.image[:, :, 3]

        self.rgb = np.stack([red, green, blue], axis=-1)

        self.thermal = thermal

        print("RGB Shape :", self.rgb.shape)
        print("Thermal Shape :", self.thermal.shape)

    ########################################################

    def normalize(self):

        print("=" * 50)
        print("Normalizing...")

        rgb = self.rgb.astype(np.float32)

        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        thermal = self.thermal.astype(np.float32)

        thermal = (
            thermal - thermal.min()
        ) / (
            thermal.max() - thermal.min() + 1e-8
        )

        self.rgb = rgb
        self.thermal = thermal

    ########################################################

    def save_preview(self):

        print("=" * 50)
        print("Saving Preview Images...")

        os.makedirs("outputs", exist_ok=True)

        rgb = (self.rgb * 255).astype(np.uint8)

        cv2.imwrite(
            "outputs/rgb_preview.png",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )

        thermal = (self.thermal * 255).astype(np.uint8)

        cv2.imwrite(
            "outputs/thermal_preview.png",
            thermal
        )

        print("Preview images saved.")

    ########################################################

    def create_low_resolution(self):

        print("=" * 50)
        print("Creating Low Resolution Thermal...")

        h, w = self.thermal.shape

        self.lr_thermal = cv2.resize(

            self.thermal,

            (w // self.scale_factor,
             h // self.scale_factor),

            interpolation=cv2.INTER_AREA

        )

        print("LR Shape :", self.lr_thermal.shape)

    ########################################################

    def create_patches(self):

        print("=" * 50)
        print("Generating patches...")

        Path(self.output_dir).mkdir(
            parents=True,
            exist_ok=True
        )

        sample = 0

        h_lr, w_lr = self.lr_thermal.shape

        for y in range(
                0,
                h_lr - self.patch_size + 1,
                self.stride
        ):

            for x in range(
                    0,
                    w_lr - self.patch_size + 1,
                    self.stride
            ):

                lr = self.lr_thermal[
                    y:y + self.patch_size,
                    x:x + self.patch_size
                ]

                y2 = y * self.scale_factor
                x2 = x * self.scale_factor

                hr = self.thermal[
                    y2:y2 + self.patch_size * self.scale_factor,
                    x2:x2 + self.patch_size * self.scale_factor
                ]

                rgb = self.rgb[
                    y2:y2 + self.patch_size * self.scale_factor,
                    x2:x2 + self.patch_size * self.scale_factor,
                    :
                ]

                if hr.shape != (
                        self.patch_size * self.scale_factor,
                        self.patch_size * self.scale_factor
                ):
                    continue

                if rgb.shape != (
                        self.patch_size * self.scale_factor,
                        self.patch_size * self.scale_factor,
                        3
                ):
                    continue

                folder = os.path.join(
                    self.output_dir,
                    f"sample_{sample:05d}"
                )

                os.makedirs(folder, exist_ok=True)

                np.save(
                    os.path.join(folder, "tir_200m.npy"),
                    lr
                )

                np.save(
                    os.path.join(folder, "tir_100m.npy"),
                    hr
                )

                np.save(
                    os.path.join(folder, "rgb.npy"),
                    rgb
                )

                sample += 1

        print(f"\nTotal Samples Generated : {sample}")

    ########################################################

    def run(self):

        self.load()

        self.split_bands()

        self.normalize()

        self.save_preview()

        self.create_low_resolution()

        self.create_patches()


############################################################

if __name__ == "__main__":

    generator = DatasetGenerator(

        input_path="datasets/raw/landsat9_auto.tif",

        output_dir="datasets/processed",

        patch_size=512,

        stride=256,

        scale_factor=2

    )

    generator.run()