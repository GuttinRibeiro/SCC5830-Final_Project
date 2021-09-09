from utils import constants
from utils.tools import *
import os
import time
from models import DepthCompletionNN
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import distance_transform_edt
import argparse

def train(dataloader, verbose=False, epochs=30, device='cuda'):
    # Create model
    model = DepthCompletionNN()
    if verbose == True:
        print(model)

    # Depth loss: L1Loss
    depth_loss = torch.nn.L1Loss()

    # Optimizer: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create a summary writer to register statistics in tensorboard
    writer = SummaryWriter()

    # Training loop
    step = 0
    print('The training loop started!')
    for epoch in range(epochs):
    
        # Inicializar as variáveis de perda e as componentes
        running_loss = 0.0
        running_depth_loss = 0.0
        running_grad_loss = 0.0
        running_ssim_loss = 0.0

        # Habilita o treinamento dos pesos da rede
        model.train()

        # Inicializa o contador de tempo
        tic = time.time()

        # Iteração para cada lote
        for i, data in enumerate(dataloader):
            step += 1
            # Pegar as imagens colorida, profundidade medida (inputs) e profundidade completa (labels)
            colors, inputs, labels = data

            # Mover as imagens para o mesmo dispositivo que o modelo está (CPU ou GPU)
            colors, inputs, labels = colors.to(device), inputs.to(device), labels.to(device)

            # Concatenar as componentes de cor e de profundidade para formar a entrada da rede
            inputs_concat = torch.cat([colors, inputs], dim=1)

            # Resetar os gradientes do otimizador
            optimizer.zero_grad()

            # Calcular a saída do modelo
            outputs = model(inputs_concat)
            # outputs = model(colors)

            # Calcula os valores da entrada que têm valor nulo (referentes aos buracos)
            holes_mask = torch.eq(inputs, 0)

            # Calcula a transformada da distância para a máscara com os buracos para ponderar
            # o MAE e priviliegiar o preenchimento dos buracos durante a regressão
            weights = torch.from_numpy((distance_transform_edt(holes_mask.to('cpu')))*10+1).to(device)

            # Calcula a soma das diferenças dos gradientes em cada direção
            loss_grad = gradient_loss(outputs, labels)

            # Calcula a dissimilaridade estrutural
            loss_ssim = structural_disparity_loss(outputs, labels)

            # Multiplica a saída obtida da rede e os rótulos de profundidade pelos pesos
            # obtidos pela transformada da distância
            outputs_w = torch.mul(outputs, weights).float()
            labels_w = torch.mul(labels, weights).float()
            # outputs_w = outputs
            # labels_w = labels

            # Calcula o MAE considerando os valores ponderados
            loss_depth = depth_loss(outputs_w, labels_w)

            # Calcula o valor final da função de perda a partir das componentes anteriores
            loss = loss_depth+5*loss_grad+1000*loss_ssim
            # loss = loss_depth+10000*loss_ssim

            # Etapa de propagação para trás e passo do otimizador para ajustar os pesos do
            # modelo em treinamento
            loss.backward()
            optimizer.step()

            # Acumular os valores das componentes da função de perda
            running_loss += loss.item()
            running_depth_loss += loss_depth.item()
            running_grad_loss += loss_grad.item()
            running_ssim_loss += loss_ssim.item()
            if i % 10 == 9:    # Mostrar andamento a cada 10 lotes
                toc = time.time()

                # Calcular estatísticas durante o treinamento (MAE, RMSE e erro relativo absoluto) para uma imagem
                depth_n = min_max_norm(labels[0], scale=1.0)
                pred_n = min_max_norm(outputs[0], scale=1.0)
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

                # Salvar valores para posterior visualização no tensorboard
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

        # Salvar pesos ao fim da época
        torch.save(model.state_dict(), os.path.join(constants.WEIGHTS_FOLDER, 'torch_weights'+str(epoch)+'.pth'))

    print('Finished Training')
    # Salvar pesos do modelo final
    torch.save(model.state_dict(), os.path.join(constants.WEIGHTS_FOLDER, 'torch_weights_final.pth'))

def eval(dataloader, device='cuda'):
    # Create model and load weights
    model = DepthCompletionNN().to(device)
    model.load_state_dict(torch.load(os.path.join(constants.WEIGHTS_FOLDER, 'torch_weights_final.pth')))
    model.eval()

    # Initialize variables to calculate statistics over the dataset
    mae = 0.0000
    rmse = 0.0000
    rel = 0.0000
    count = 0.0000

    for _, data in enumerate(dataloader):
        colors, inputs, labels = data
        colors, inputs, labels = colors.to(device), inputs.to(device), labels.to(device)
        inputs_nn = torch.cat([colors, inputs], dim=1)

        outputs = model(inputs_nn)

        # Write each image in batch
        for j in range(outputs.shape[0]):
            rel += RelativeError(outputs[j], labels[j])
            outputs[j] = min_max_norm(outputs[j], scale=1.0)
            labels[j] = min_max_norm(labels[j], scale=1.0)

            mae += MAE(outputs[j], labels[j])
            rmse += RMSE(outputs[j], labels[j])
            count += 1

            imageio.imwrite(os.path.join(constants.OUTPUT_FOLDER, str(int(count))+'.jpg'), tensor_to_rgb(outputs[j]))
            imageio.imwrite(os.path.join(constants.OUTPUT_FOLDER, str(int(count))+'raw.jpg'), tensor_to_rgb(min_max_norm(inputs[j], scale=1.0)))
            imageio.imwrite(os.path.join(constants.OUTPUT_FOLDER, str(int(count))+'gt.jpg'), tensor_to_rgb(labels[j]))
            imageio.imwrite(os.path.join(constants.OUTPUT_FOLDER, str(int(count))+'color.jpg'), np.transpose(colors[j].cpu().detach().numpy(), axes=(1, 2, 0)).astype(np.uint8))
                
    print('Metrics:')
    print('1) MAE: ', mae/count)
    print('2) RMSE: ', rmse/count)
    print('3) Rel: ', rel/count)

if __name__ == '__main__':
    # Global constants
    BATCH_SIZE = 16
    EPOCHS = 30

    parser = argparse.ArgumentParser(description='Training script for DepthCompletionNN.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Display statistics during training')
    parser.add_argument('-r', '--run', action='store_true', default=False, help='Run the neural network after training')
    parser.add_argument('-e', '--epochs', action='store', type=int, default=EPOCHS, help='Number of epochs to train')
    parser.add_argument('-b', '--batch_size', action='store', type=int, default=BATCH_SIZE, help='Batch size')
 
    args = parser.parse_args()

    # Configure paths and folders to store trained weights and training statistics
    DATASET_PATH = os.path.join(os.getcwd(), constants.DATASET_FOLDER)

    if os.path.isdir(constants.WEIGHTS_FOLDER) == False:
        os.mkdir(constants.WEIGHTS_FOLDER)

    if os.path.isdir(constants.OUTPUT_FOLDER) == False:
        os.mkdir(constants.OUTPUT_FOLDER)

    # Load dataset and split it into two disjoint sets (train and test)
    dataset = DepthCompletionDataset(DATASET_PATH, 'color', 'depth', 'rawDepth', '.png')
    train_loader, test_loader = train_test_split(dataset, validation_split=0.1, batch_size=args.batch_size)

    # Move model to GPU if available (preferred mode)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Selected device: ', device)

    train(train_loader, verbose=args.verbose, epochs=args.epochs, device=device)

    if args.run == True:
        print('Evaluating results in training...')
        eval(test_loader, device=device)