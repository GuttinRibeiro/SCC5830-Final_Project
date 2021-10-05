from utils import constants
from utils.tools import *
import os
from models import DenseNN
from scipy.ndimage import distance_transform_edt

# import matplotlib.pyplot as plt
# import cv2
# from skimage.util import img_as_uint

# import time

def vector_to_rgb(vector):
    vector = min_max_norm(vector.numpy())
    return vector.astype(np.uint8)


if __name__ == '__main__':
    # Move model to GPU if available (preferred mode)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Selected device: ', device)

    model = DenseNN().to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.load_state_dict(torch.load(os.path.join(constants.WEIGHTS_FOLDER, 'torch_weights_final_obs2.pth')))
    model.eval()
    # Global constants
    BATCH_SIZE = 4

    # Configure paths and folders to store trained weights and training statistics
    DATASET_PATH = os.path.join(os.getcwd(), constants.DATASET_FOLDER)
    dataset = DepthCompletionDataset(DATASET_PATH, 'color', 'depth', 'rawDepth', '.png')
    _, test_loader = train_test_split(dataset, validation_split=0.1, batch_size=BATCH_SIZE)

    for _, data in enumerate(test_loader):
        colors, inputs, labels = data
        colors, inputs, labels = colors.to(device), inputs.to(device), labels.to(device)
        inputs_concat = torch.cat([colors, inputs], dim=1)

        outputs = model(inputs_concat)

        mask = torch.eq(inputs, 0)
        weights = torch.from_numpy((distance_transform_edt(mask.to('cpu')))*10+1).to(device)

        outputs_w = torch.mul(outputs, weights).float()
        labels_w = torch.mul(labels, weights).float()

        s1 = ssim(outputs, labels, 11, 7500.0).mean(1).mean(1).mean(1).cpu()
        # s2 = ssim(outputs_w, labels, 11, 7500.0).mean(1).mean(1).mean(1).cpu()
        s3 = ssim(outputs_w, labels_w, 11, 7500.0).mean(1).mean(1).mean(1).cpu()

        print('SSIM inicial: ', s1)
        # print('SSIM ponderando entrada: ', s2)
        print('SSIM ponderando ambas: ', s3)