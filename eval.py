from utils import constants
from utils.tools import *
import os
import time
from models import DenseNN
# import matplotlib.pyplot as plt

def eval(dataloader, weight_final_token, output_file, device='cuda', rec_param=7):
    # Create model and load weights
    model = DenseNN(repetitions=rec_param).to(device)
    model.load_state_dict(torch.load(os.path.join(constants.WEIGHTS_FOLDER, weight_final_token)))
    model.eval()

    # Initialize variables to calculate statistics over the dataset
    mae = 0.0000
    rmse = 0.0000
    rel = 0.0000
    delta = 0.0000
    delta_sq = 0.0000
    delta_cb = 0.0000
    count = 0
    mae_mask = 0.0000
    rmse_mask = 0.0000
    rel_mask = 0.0000
    delta_mask = 0.0000
    delta_sq_mask = 0.0000
    delta_cb_mask = 0.0000
    time_count = 0.0

    # Min and max MAE over the test set
    min_mae = 10.0000
    max_mae = 0.0000

    for _, data in enumerate(dataloader):
        colors, inputs, labels = data
        colors, inputs, labels = colors.to(device), inputs.to(device), labels.to(device)
        inputs_nn = torch.cat([colors, inputs], dim=1)

        tic = time.time()
        outputs, depth = model(inputs_nn)
        toc = time.time()
        time_count += toc-tic

        # Write each image in batch
        for j in range(outputs.shape[0]):
            mask = torch.eq(inputs[j], 0)
            rel_mask += RelativeError(outputs[j], labels[j], mask)
            delta_mask += threshold(outputs[j], labels[j], 1.25, mask)
            delta_sq_mask += threshold(outputs[j], labels[j], 1.25**2, mask)
            delta_cb_mask += threshold(outputs[j], labels[j], 1.25**3, mask)

            rel += RelativeError(outputs[j], labels[j])
            delta += threshold(outputs[j], labels[j], 1.25)
            delta_sq += threshold(outputs[j], labels[j], 1.25**2)
            delta_cb += threshold(outputs[j], labels[j], 1.25**3)
            outputs[j] = (outputs[j]-torch.min(outputs[j]))/(torch.max(outputs[j])-torch.min(outputs[j]))
            labels[j] = (labels[j]-torch.min(labels[j]))/(torch.max(labels[j])-torch.min(labels[j]))
            depth[j] = (depth[j]-torch.min(depth[j]))/(torch.max(depth[j])-torch.min(depth[j]))

            mae_aux = MAE(outputs[j], labels[j], mask)
            if mae_aux < min_mae:
                min_mae = mae_aux

            if mae_aux > max_mae:
                max_mae = mae_aux

            mae_mask += mae_aux
            rmse_mask += RMSE(outputs[j], labels[j], mask)
            mae += MAE(outputs[j], labels[j])
            rmse += RMSE(outputs[j], labels[j])
            count += 1

    with open(output_file, "w") as f:
        f.write('0) Inference time: ' + str(time_count/count)+'\n')
        f.write('1) MAE: ' + str(mae/count)+'\n')
        f.write('2) RMSE: ' + str(rmse/count)+'\n')
        f.write('3) Rel: ' + str(rel/count)+'\n')
        f.write('4) Delta < 1.25' + str(delta/count)+'\n')
        f.write('5) Delta < 1.25^2' + str(delta_sq/count)+'\n')
        f.write('6) Delta < 1.25^3' + str(delta_cb/count)+'\n')
        f.write('Statistics considering only valid pixels:'+'\n')
        f.write('1) MAE: ' + str(mae_mask/count)+'\n')
        f.write('2) RMSE: ' + str(rmse_mask/count)+'\n')
        f.write('3) Rel: ' + str(rel_mask/count)+'\n')
        f.write('4) Delta < 1.25' + str(delta_mask/count)+'\n')
        f.write('5) Delta < 1.25^2' + str(delta_sq_mask/count)+'\n')
        f.write('6) Delta < 1.25^3' + str(delta_cb_mask/count)+'\n')
        f.write('7) Min MAE: ' + str(min_mae)+'\n')
        f.write('8) Max MAE: ' + str(max_mae)+'\n')
        f.close()

    print('Metrics:')
    print('0) Inference time: ', time_count/count)
    print('1) MAE: ', mae/count)
    print('2) RMSE: ', rmse/count)
    print('3) Rel: ', rel/count)
    print('4) Delta < 1.25', delta/count)
    print('5) Delta < 1.25^2', delta_sq/count)
    print('6) Delta < 1.25^3', delta_cb/count)

def histogram(dataloader, weight_final_token, device='cuda', rec_param=7):
    # Create model and load weights
    model = DenseNN(repetitions=rec_param).to(device)
    model.load_state_dict(torch.load(os.path.join(constants.WEIGHTS_FOLDER, weight_final_token)))
    model.eval()

    # Initialize variables to calculate statistics over the dataset
    mae = []

    for _, data in enumerate(dataloader):
        colors, inputs, labels = data
        colors, inputs, labels = colors.to(device), inputs.to(device), labels.to(device)
        inputs_nn = torch.cat([colors, inputs], dim=1)

        outputs, _ = model(inputs_nn)

        # Write each image in batch
        for j in range(outputs.shape[0]):
            mask = torch.eq(inputs[j], 0)
            outputs[j] = (outputs[j]-torch.min(outputs[j]))/(torch.max(outputs[j])-torch.min(outputs[j]))
            labels[j] = (labels[j]-torch.min(labels[j]))/(torch.max(labels[j])-torch.min(labels[j]))

            mae.append(MAE(outputs[j], labels[j], mask)) 

    mae = np.array(mae)
    with open('histogram.txt', 'w') as f:
        for measurement in mae:
            f.write(str(measurement)+'\n')

        f.close()
    # plt.hist(mae, bins=15, range=(0.02, 0.34))
    # plt.show()
    # hist, edges = np.histogram(mae, bins=15, range=(0.02, 0.34))


if __name__ == '__main__':
    # Configure paths and folders to store trained weights and training statistics
    DATASET_PATH = os.path.join(os.getcwd(), constants.DATASET_FOLDER)

    # Load image lists and split them into training and validation
    img_l, raw_l, depth_l = get_file_lists(DATASET_PATH)
    train_idx, val_idx = split_set_indices(img_l, split_ratio=0.1685)

    # Select the enabled device to move my model to it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Selected device: ', device)

    test_dataset = RGBDCompletionDataset(img_l[val_idx], raw_l[val_idx], depth_l[val_idx], DATASET_PATH, apply_augmentation=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    # eval(test_loader, 'torch_weights_final_5rep.pth', 'table_statistics.txt', device=device, rec_param=5)
    histogram(test_loader, 'torch_weights_final_5rep.pth', device=device, rec_param=5)
