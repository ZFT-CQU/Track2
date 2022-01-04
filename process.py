import SimpleITK
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from utils import convert_to_range_0_1, contrast_matching, poisson_blend, nodule_diameter
import json
from pathlib import Path
import time

import torch
from torch import autograd
from WGAN import Generator
import scipy.ndimage as ndi
from skimage import morphology

# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
execute_in_docker = True


class Nodulegeneration(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(validators=dict(input_image=(
            UniqueImagesValidator(),
            UniquePathIndicesValidator(),
        )),
            input_path=Path("/input/")
            if execute_in_docker else Path("./test/"),
            output_path=Path("/output/")
            if execute_in_docker else Path("./output/"),
            output_file=Path("/output/results.json") if
            execute_in_docker else Path("./output/results.json"))

        # load nodules.json for location
        with open("/input/nodules.json"
                  if execute_in_docker else "test/nodules.json") as f:
            self.data = json.load(f)

        self.generator = Generator()
        self.generator.load_state_dict(
            torch.load('checkpoint/netG_last.pth')['model'])

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        input_image = SimpleITK.GetArrayFromImage(input_image)
        print(input_image.shape)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            gpu = 0
            self.generator = self.generator.cuda(gpu)
        self.generator.eval()
        total_time = time.time()
        if len(input_image.shape) == 2:
            input_image = np.expand_dims(input_image, 0)

        nodule_images = np.zeros(input_image.shape)
        for j in range(len(input_image)):
            cxr_img_scaled = input_image[j, :, :]
            nodule_data = [
                i for i in self.data['boxes'] if i['corners'][0][2] == j
            ]

            for nodule in nodule_data:
                cxr_img_scaled = convert_to_range_0_1(cxr_img_scaled)
                boxes = nodule['corners']
                # print(boxes)
                y_min, x_min, y_max, x_max = boxes[2][0], boxes[2][1], boxes[
                    0][0], boxes[0][1]

                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(
                    x_max), int(y_max)

                required_diameter = max(x_max - x_min, y_max - y_min)
                times = 2.0
                flag = False
                while True:
                    for _ in range(10):
                        # use WGAN to generate samples
                        fixed_noise_128 = torch.randn(1, 128)
                        if use_cuda:
                            fixed_noise_128 = fixed_noise_128.cuda(gpu)
                        noisev = autograd.Variable(fixed_noise_128)
                        samples = self.generator(noisev)
                        nodule = samples.cpu().data.numpy()[0, 0]
                        # denoise
                        mask = morphology.remove_small_objects(nodule > 10.0, min_size=50, connectivity=1)
                        nodule[mask == False] = 0.0
                        diameter = nodule_diameter(nodule)
                        if diameter >= (required_diameter / times):
                            flag = True
                            break
                    times = times * 2.0
                    if flag:
                        break
                scaling_factor = required_diameter / diameter
                nodule = ndi.interpolation.zoom(nodule,
                                                scaling_factor,
                                                mode='nearest',
                                                order=1)
                crop = cxr_img_scaled[x_min:x_max, y_min:y_max].copy()
                nodule = convert_to_range_0_1(nodule)

                # contrast matching:
                c = contrast_matching(nodule, cxr_img_scaled[x_min:x_max,
                                                             y_min:y_max])
                nodule = nodule * c

                result = poisson_blend(nodule, cxr_img_scaled, y_min, y_max,
                                       x_min, x_max)
                result[x_min:x_max, y_min:y_max] = np.mean(np.array(
                    [crop * 255, result[x_min:x_max, y_min:y_max]]),
                    axis=0)
                cxr_img_scaled = result.copy()

            nodule_images[j, :, :] = result
        print('total time took ', time.time() - total_time)
        return SimpleITK.GetImageFromArray(nodule_images)


if __name__ == "__main__":
    Nodulegeneration().process()
