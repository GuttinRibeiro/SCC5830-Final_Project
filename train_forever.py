from utils import constants
from utils.tools import *
import os
import time
from models import DenseNN
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import distance_transform_edt
import argparse

def train(dataloader, weight_partial_token, weight_final_token, verbose=False, epochs=30, device='cuda', rec_param=7, use_edt=True):
    # Create model
    model = DenseNN(repetitions=rec_param).to(device)
    if verbose == True:
        print(model)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Depth loss: L1Loss
    # depth_loss = torch.nn.L1Loss(reduction='none')
    depth_loss = torch.nn.L1Loss()

    # Optimizer: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create a summary writer to register statistics in tensorboard
    writer = SummaryWriter()

    # Training loop
    step = 0
    print('The training loop started!')
    for epoch in range(epochs):
    
        # Inicializar as variaveis de perda e as componentes
        running_loss = 0.0
        running_depth_loss = 0.0
        running_grad_loss = 0.0
        running_ssim_loss = 0.0

        # Habilita o treinamento dos pesos da rede
        model.train()

        # Inicializa o contador de tempo
        tic = time.time()

        # Iteracao para cada lote
        for i, data in enumerate(dataloader):
            step += 1
            # Pegar as imagens colorida, profundidade medida (inputs) e profundidade completa (labels)
            colors, inputs, labels = data

            # Mover as imagens para o mesmo dispositivo que o modelo esta (CPU ou GPU)
            colors, inputs, labels = colors.to(device), inputs.to(device), labels.to(device)

            # Concatenar as componentes de cor e de profundidade para formar a entrada da rede
            inputs_concat = torch.cat([colors, inputs], dim=1)

            # Resetar os gradientes do otimizador
            optimizer.zero_grad()

            # Calcular a saida do modelo
            outputs, _ = model(inputs_concat)
            # outputs = model(colors)

            # Calcula os valores da entrada que tem valor nulo (referentes aos buracos)
            mask = torch.eq(inputs, 0)
            # valid_pixels_bacth = mask.sum(dim=(1, 2, 3))

            # Calcula a transformada da distancia para a mascara com os buracos para ponderar
            # o MAE e priviliegiar o preenchimento dos buracos durante a regressao
            if use_edt:
                weights = torch.from_numpy((distance_transform_edt(mask.to('cpu')))*10+1).to(device)
                outputs_w = torch.mul(outputs, weights).float()
                labels_w = torch.mul(labels, weights).float()
            else:
                outputs_w = outputs
                labels_w = labels

            # Calcula a soma das diferencas dos gradientes em cada direcao
            loss_grad = gradient_loss(outputs_w, labels_w)

            # Calcula a dissimilaridade estrutural
            loss_ssim = structural_disparity_loss(outputs_w, labels_w)

            # Calcula o MAE considerando os valores ponderados
            loss_depth = depth_loss(outputs_w, labels_w)
            # # Update: considerar apenas os valores com pixels medidos
            # depth_sum = torch.sum(mask*loss_depth, dim=(1, 2, 3))
            # loss_depth = torch.sum(depth_sum/valid_pixels_bacth)
            
            # Calcula o valor final da funcao de perda a partir das componentes anteriores
            loss = loss_depth+0.5*loss_grad+1000*loss_ssim

            # Etapa de propagacao para tras e passo do otimizador para ajustar os pesos do
            # modelo em treinamento
            loss.backward()
            optimizer.step()

            # Acumular os valores das componentes da funcao de perda
            running_loss += loss.item()
            running_depth_loss += loss_depth.item()
            running_grad_loss += loss_grad.item()
            running_ssim_loss += loss_ssim.item()
            if i % 10 == 9:    # Mostrar andamento a cada 10 lotes
                toc = time.time()

                # Calcular estatisticas durante o treinamento (MAE, RMSE e erro relativo absoluto) para uma imagem
                depth_n = (labels[0]-torch.min(labels[0]))/(torch.max(labels[0])-torch.min(labels[0]))
                pred_n = (outputs[0]-torch.min(outputs[0]))/(torch.max(outputs[0])-torch.min(outputs[0]))
                mae_mask = MAE(pred_n, depth_n, mask[0])
                rmse_mask =  RMSE(pred_n, depth_n, mask[0])
                rel_mask = RelativeError(outputs[0], labels[0], mask[0])
                mae = MAE(pred_n, depth_n)
                rmse =  RMSE(pred_n, depth_n)
                rel = RelativeError(outputs[0], labels[0])

                if verbose:
                    print('[%d, %5d] Training loss: %.6f' %
                        (epoch + 1, i + 1, running_loss / 10))
                    print('[%d, %5d] Depth loss: %.6f' %
                        (epoch + 1, i + 1, running_depth_loss / 10))
                    print('[%d, %5d] Grad loss: %.6f' %
                        (epoch + 1, i + 1, running_grad_loss / 10))
                    print('[%d, %5d] SSIM loss: %.6f' %
                        (epoch + 1, i + 1, running_ssim_loss / 10))
                    print('[%d, %5d] MAE for one sample: %.6f' %
                        (epoch + 1, i + 1, mae))
                    print('[%d, %5d] RMSE for one sample: %.6f' %
                        (epoch + 1, i + 1, rmse))
                    print('[%d, %5d] Relative error for one sample: %.6f' %
                        (epoch + 1, i + 1, rel))
                    print('Elapsed time: ', toc-tic)

                # Salvar valores para posterior visualizacao no tensorboard
                writer.add_scalar('Loss/train/loss_depth', running_depth_loss / 10, step)
                writer.add_scalar('Loss/train/loss_grad', running_grad_loss / 10, step)
                writer.add_scalar('Loss/train/loss_ssim', running_ssim_loss / 10, step)
                writer.add_scalar('Loss/train/loss', running_loss / 10, step)
                writer.add_scalar('Metrics/train/mae', mae, step)
                writer.add_scalar('Metrics/train/rmse', rmse, step)
                writer.add_scalar('Metrics/train/rel', rel, step)
                writer.add_scalar('Metrics/train/mae_mask', mae_mask, step)
                writer.add_scalar('Metrics/train/rmse_mask', rmse_mask, step)
                writer.add_scalar('Metrics/train/rel_mask', rel_mask, step)

                # Resetar acumuladores
                running_loss = 0.0
                running_depth_loss = 0.0
                running_grad_loss = 0.0
                running_ssim_loss = 0.0
                tic = time.time()

        # Salvar pesos ao fim da epoca
        torch.save(model.state_dict(), os.path.join(constants.WEIGHTS_FOLDER, weight_partial_token+str(epoch)+'.pth'))

    print('Finished Training')
    # Salvar pesos do modelo final
    torch.save(model.state_dict(), os.path.join(constants.WEIGHTS_FOLDER, weight_final_token))

def eval(dataloader, weight_final_token, output_folder, output_file, device='cuda', rec_param=7):
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

            mae_mask += MAE(outputs[j], labels[j], mask)
            rmse_mask += RMSE(outputs[j], labels[j], mask)
            mae += MAE(outputs[j], labels[j])
            rmse += RMSE(outputs[j], labels[j])
            count += 1

            imageio.imwrite(os.path.join(output_folder, str(int(count))+'.jpg'), tensor_to_rgb(outputs[j]))
            imageio.imwrite(os.path.join(output_folder, str(int(count))+'raw.jpg'), tensor_to_rgb((inputs[j]-torch.min(inputs[j]))/(torch.max(inputs[j])-torch.min(inputs[j]))))
            imageio.imwrite(os.path.join(output_folder, str(int(count))+'gt.jpg'), tensor_to_rgb(labels[j]))
            imageio.imwrite(os.path.join(output_folder, str(int(count))+'color.jpg'), np.transpose(colors[j].cpu().detach().numpy(), axes=(1, 2, 0)).astype(np.uint8))
            imageio.imwrite(os.path.join(output_folder, str(int(count))+'unrefined.jpg'), tensor_to_rgb(depth[j]))

    with open(output_file, "w") as f:
        f.write('0) Inference time: ' + str(time_count/count))
        f.write('1) MAE: ' + str(mae/count))
        f.write('2) RMSE: ' + str(rmse/count))
        f.write('3) Rel: ' + str(rel/count))
        f.write('4) Delta < 1.25' + str(delta/count))
        f.write('5) Delta < 1.25^2' + str(delta_sq/count))
        f.write('6) Delta < 1.25^3' + str(delta_cb/count))
        f.write('Statistics considering only valid pixels:')
        f.write('1) MAE: ' + str(mae_mask/count))
        f.write('2) RMSE: ' + str(rmse_mask/count))
        f.write('3) Rel: ' + str(rel_mask/count))
        f.write('4) Delta < 1.25' + str(delta_mask/count))
        f.write('5) Delta < 1.25^2' + str(delta_sq_mask/count))
        f.write('6) Delta < 1.25^3' + str(delta_cb_mask/count))
        f.close()

    print('Metrics:')
    print('0) Inference time: ', time_count/count)
    print('1) MAE: ', mae/count)
    print('2) RMSE: ', rmse/count)
    print('3) Rel: ', rel/count)
    print('4) Delta < 1.25', delta/count)
    print('5) Delta < 1.25^2', delta_sq/count)
    print('6) Delta < 1.25^3', delta_cb/count)

if __name__ == '__main__':
    # Training configuration
    recursive_prop_param_list = [5, 9, 3, 5, 7, 9]
    weights_partial_token_list = ['torch_weights_5rep', 'torch_weights_9rep', 'torch_weights_3rep_noedt', 'torch_weights_5rep_noedt', 'torch_weights_7rep_noedt', 'torch_weights_9rep_noedt']
    weights_final_token_list = ['torch_weights_final_5rep.pth', 'torch_weights_final_9rep.pth', 'torch_weights_final_3rep_noedt.pth', 'torch_weights_final_5rep_noedt.pth', 'torch_weights_final_7rep_noedt.pth', 'torch_weights_final_9rep_noedt.pth']
    edt_control_list = [True, True, False, False, False, False]
    outfolders = ['outputs100_5rep', 'outputs100_9rep', 'outputs100_3rep_noedt', 'outputs100_5rep_noedt', 'outputs100_7rep_noedt', 'outputs100_9rep_noedt']
    outfiles = ['outputs100_5rep.txt', 'outputs100_9rep.txt', 'outputs100_3rep_noedt.txt', 'outputs100_5rep_noedt.txt', 'outputs100_7rep_noedt.txt', 'outputs100_9rep_noedt.txt']

    assert len(recursive_prop_param_list) == len(weights_partial_token_list) == len(weights_final_token_list) == len(edt_control_list) == len(outfolders)

    # Global constants
    BATCH_SIZE = 4
    EPOCHS = 30

    parser = argparse.ArgumentParser(description='Training script for DepthCompletionNN.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Display statistics during training')
    parser.add_argument('-t', '--train', action='store_true', default=False, help='Train my CNN using training set')
    parser.add_argument('-r', '--run', action='store_true', default=False, help='Run my CNN using test set')
    parser.add_argument('-e', '--epochs', action='store', type=int, default=EPOCHS, help='Number of epochs to train')
    parser.add_argument('-b', '--batch_size', action='store', type=int, default=BATCH_SIZE, help='Batch size')
 
    args = parser.parse_args()

    # Configure paths and folders to store trained weights and training statistics
    DATASET_PATH = os.path.join(os.getcwd(), constants.DATASET_FOLDER)

    if os.path.isdir(constants.WEIGHTS_FOLDER) == False:
        os.mkdir(constants.WEIGHTS_FOLDER)

    # Load image lists and split them into training and validation
    img_l, raw_l, depth_l = get_file_lists(DATASET_PATH)
    train_idx, val_idx = split_set_indices(img_l, split_ratio=0.1685)

    if args.verbose:
        print('Number of images in test set: ', len(val_idx))

    # Select the enabled device to move my model to it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Selected device: ', device)

    for i in range(len(recursive_prop_param_list)):
        print('Training ', i, ' has started!')
        if os.path.isdir(outfolders[i]) == False:
            os.mkdir(outfolders[i])

        if args.train == True:
            # Create dataset and train loader applying data augmentation and its respective dataloader
            train_dataset = RGBDCompletionDataset(img_l[train_idx], raw_l[train_idx], depth_l[train_idx], DATASET_PATH, apply_augmentation=True)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            # Perform training loop
            train(train_loader, weight_partial_token=weights_partial_token_list[i], weight_final_token=weights_final_token_list[i], verbose=args.verbose, epochs=args.epochs, device=device, rec_param=recursive_prop_param_list[i], use_edt=edt_control_list[i])

        if args.run == True:
            # Create dataset and train loader applying data augmentation and its respective dataloader
            test_dataset = RGBDCompletionDataset(img_l[val_idx], raw_l[val_idx], depth_l[val_idx], DATASET_PATH, apply_augmentation=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

            print('Evaluating results using test set...')
            eval(test_loader, weight_final_token=weights_final_token_list[i], output_folder=outfolders[i], output_file=outfiles[i], device=device)