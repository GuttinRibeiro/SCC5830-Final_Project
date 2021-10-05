# coding=utf-8
from utils import constants
from utils.tools import *
import os
import time
from models import DenseNN
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import distance_transform_edt
import argparse

def train(dataloader, verbose=False, epochs=30, device='cuda'):
    # Create model
    model = DenseNN().to(device)
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
            weights = torch.from_numpy((distance_transform_edt(mask.to('cpu')))*10+1).to(device)
            outputs_w = torch.mul(outputs, weights).float()
            labels_w = torch.mul(labels, weights).float()

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
                # mae = MAE(pred_n, depth_n, mask[0])
                # rmse =  RMSE(pred_n, depth_n, mask[0])
                # rel = RelativeError(outputs[0], labels[0], mask[0])
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

                # Resetar acumuladores
                running_loss = 0.0
                running_depth_loss = 0.0
                running_grad_loss = 0.0
                running_ssim_loss = 0.0
                tic = time.time()

        # Salvar pesos ao fim da epoca
        torch.save(model.state_dict(), os.path.join(constants.WEIGHTS_FOLDER, 'torch_weights_7rep_correct'+str(epoch)+'.pth'))

    print('Finished Training')
    # Salvar pesos do modelo final
    torch.save(model.state_dict(), os.path.join(constants.WEIGHTS_FOLDER, 'torch_weights_final_7rep_correct.pth'))

def eval(dataloader, device='cuda'):
    # Create model and load weights
    model = DenseNN().to(device)
    model.load_state_dict(torch.load(os.path.join(constants.WEIGHTS_FOLDER, 'torch_weights_final_7rep_correct.pth')))
    model.eval()

    # Initialize variables to calculate statistics over the dataset
    mae = 0.0000
    rmse = 0.0000
    rel = 0.0000
    delta = 0.0000
    delta_sq = 0.0000
    delta_cb = 0.0000
    count = 0.0000

    for _, data in enumerate(dataloader):
        colors, inputs, labels = data
        colors, inputs, labels = colors.to(device), inputs.to(device), labels.to(device)
        inputs_nn = torch.cat([colors, inputs], dim=1)

        outputs, depth = model(inputs_nn)

        # Write each image in batch
        for j in range(outputs.shape[0]):
            # mask = torch.eq(inputs[j], 0)
            # rel += RelativeError(outputs[j], labels[j], mask)
            # delta += threshold(outputs[j], labels[j], 1.25, mask)
            # delta_sq += threshold(outputs[j], labels[j], 1.25**2, mask)
            # delta_cb += threshold(outputs[j], labels[j], 1.25**3, mask)

            rel += RelativeError(outputs[j], labels[j])
            delta += threshold(outputs[j], labels[j], 1.25)
            delta_sq += threshold(outputs[j], labels[j], 1.25**2)
            delta_cb += threshold(outputs[j], labels[j], 1.25**3)
            outputs[j] = (outputs[j]-torch.min(outputs[j]))/(torch.max(outputs[j])-torch.min(outputs[j]))
            labels[j] = (labels[j]-torch.min(labels[j]))/(torch.max(labels[j])-torch.min(labels[j]))
            depth[j] = (depth[j]-torch.min(depth[j]))/(torch.max(depth[j])-torch.min(depth[j]))

            # mae += MAE(outputs[j], labels[j], mask)
            # rmse += RMSE(outputs[j], labels[j], mask)
            mae += MAE(outputs[j], labels[j])
            rmse += RMSE(outputs[j], labels[j])
            count += 1

            imageio.imwrite(os.path.join(constants.OUTPUT_FOLDER, str(int(count))+'.jpg'), tensor_to_rgb(outputs[j]))
            imageio.imwrite(os.path.join(constants.OUTPUT_FOLDER, str(int(count))+'raw.jpg'), tensor_to_rgb((inputs[j]-torch.min(inputs[j]))/(torch.max(inputs[j])-torch.min(inputs[j]))))
            imageio.imwrite(os.path.join(constants.OUTPUT_FOLDER, str(int(count))+'gt.jpg'), tensor_to_rgb(labels[j]))
            imageio.imwrite(os.path.join(constants.OUTPUT_FOLDER, str(int(count))+'color.jpg'), np.transpose(colors[j].cpu().detach().numpy(), axes=(1, 2, 0)).astype(np.uint8))
            imageio.imwrite(os.path.join(constants.OUTPUT_FOLDER, str(int(count))+'unrefined.jpg'), tensor_to_rgb(depth[j]))

                
    print('Metrics:')
    print('1) MAE: ', mae/count)
    print('2) RMSE: ', rmse/count)
    print('3) Rel: ', rel/count)
    print('4) Delta < 1.25', delta/count)
    print('5) Delta < 1.25^2', delta_sq/count)
    print('6) Delta < 1.25^3', delta_cb/count)

if __name__ == '__main__':
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

    if os.path.isdir(constants.OUTPUT_FOLDER) == False:
        os.mkdir(constants.OUTPUT_FOLDER)

    # Load image lists and split them into training and validation
    img_l, raw_l, depth_l = get_file_lists(DATASET_PATH)
    train_idx, val_idx = split_set_indices(img_l, split_ratio=0.1685)

    if args.verbose:
        print('Number of images in test set: ', len(val_idx))

    # Select the enabled device to move my model to it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Selected device: ', device)

    if args.train == True:
        # Create dataset and train loader applying data augmentation and its respective dataloader
        train_dataset = RGBDCompletionDataset(img_l[train_idx], raw_l[train_idx], depth_l[train_idx], DATASET_PATH, apply_augmentation=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Perform training loop
        train(train_loader, verbose=args.verbose, epochs=args.epochs, device=device)

    if args.run == True:
        # Create dataset and train loader applying data augmentation and its respective dataloader
        test_dataset = RGBDCompletionDataset(img_l[val_idx], raw_l[val_idx], depth_l[val_idx], DATASET_PATH, apply_augmentation=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        print('Evaluating results using test set...')
        eval(test_loader, device=device)