from utils import constants
from utils.tools import *
import os

import matplotlib.pyplot as plt
import cv2
from skimage.util import img_as_uint

import time

def vector_to_rgb(vector):
    vector = min_max_norm(vector.numpy())
    return vector.astype(np.uint8)


if __name__ == '__main__':
    # Global constants
    BATCH_SIZE = 1

    # Configure paths and folders to store trained weights and training statistics
    DATASET_PATH = os.path.join(os.getcwd(), constants.DATASET_FOLDER)

    # Load dataset and split it into two disjoint sets (train and test)
    dataset = DepthCompletionDataset(DATASET_PATH, 'color', 'depth', 'rawDepth', '.png')
    train_loader, test_loader = train_test_split(dataset, validation_split=0.0, batch_size=BATCH_SIZE)

    # Move model to GPU if available (preferred mode)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Selected device: ', device)

    for data in train_loader:
        color, raw, depth = data
        
        # plt.subplot(221)
        # plt.imshow(raw[0, 0].numpy())

        # n = normals(raw)
        # plt.subplot(222)
        # plt.imshow(np.transpose(vector_to_rgb(n[0]), axes=(1, 2, 0)))
        # # cv2.imwrite('raw.tif', cv2.cvtColor(img_as_uint(np.transpose(n[0].numpy(), axes=(1, 2, 0))), cv2.COLOR_RGB2BGR))

        # plt.subplot(223)
        # plt.imshow(depth[0, 0].numpy())

        # plt.subplot(224)
        # plt.imshow(np.transpose(vector_to_rgb(normals(depth)[0]), axes=(1, 2, 0)))
        # # cv2.imwrite('depth.tif', cv2.cvtColor(img_as_uint(np.transpose(normals(depth)[0].numpy(), axes=(1, 2, 0))), cv2.COLOR_RGB2BGR))

        # plt.show()

        # tic = time.time()
        edges = canny_edge_detector(color)[0].numpy()
        # toc = time.time()
        # print(toc-tic)
        plt.imshow(edges, cmap='gray')
        plt.show()