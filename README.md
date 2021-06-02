# Preenchimento de profundidade em imagens RGB-D
Este repositório contém o trabalho final da disciplina SCC5830 - Processamento de Imagens cursada no primeiro semestre de 2021 no ICMC USP.
#### Autor: Augusto Ribeiro Castro - 9771421

## Abstract

Imagens RGB-D são uma combinação de uma imagem RGB e a correspondente imagem de profundidade, que guarda, para cada pixel, a distância entre o objeto no ambiente e o plano da imagem. Além da opção de utilizar técnicas computacionais para o cálculo da profundidade da imagem a partir de imagens RGB (visão estéreo ou estimativa monocular utilizando aprendizado profundo, por exemplo), é possível também obter essa informação do ambiente por uma combinação de sensores, como ocorre no Kinect, em que dados da câmera e de um sensor de profundidade são combinados. 

No caso da obtenção por meio de um hardware, apesar da capacidade de obtenção de imagens em tempo real e a um custo relativamente baixo, os dados obtidos apresentam falhas como ruídos de medição que prejudicam a definição dos limites físicos dos objetos e, muitas vezes, a ausência da medição de profundidade para algumas regiões, devido à reflexão em superfícies irregulares ou espelhadas. Sendo assim, como etapa anterior à utilização da profundidade crua de uma imagem RGB-D obtida por um hardware específico, é necessária a aplicação de técninas de "image completion" para preencher as informações faltantes. Juntamente, podem ser aplicados procedimentos que buscam aprimorar, filtrar e aumentar a resolução dessas imagens. Concluída a etapa prévia, as imagens RGB-D se tornam valiosas informações para tarefas em robótica, detectção de esqueleto, detecção de objetos e mapeamento de ambientes.

No contexto do pré-processamento da informação de profundidade de imagens RGB-D, este trabalho busca então implementar uma maneira de completar regiões em que não foi possível obter a medição de profundidade, além de possíveis etapas posteriores de melhoramento da imagem. Como fonte de dados e referência teórica é possível citar o trabalho de XIAN, Chuhua, et al. 2020 [1], cujos dados podem ser encontrados em https://github.com/chuhuaxian/HF-RGBD. Apesar de existirem outros conjuntos de dados para a tarefa, como o KITTI e o Matterplot3D, o conjunto de dados do artigo citado será o abordado neste trabalho.

## Imagens de exemplo

As imagens de exemplo foram obtidas do repositório https://github.com/chuhuaxian/HF-RGBD. No caso das imagens de profundidade, os valores foram normalizados para serem inteiros e estarem no intervalo [0, 255] apenas para fins de visualização.

#### Imagem RGB (entrada 1):

![00000-color](https://user-images.githubusercontent.com/31515305/120489256-e1247500-c38d-11eb-82b2-a9606e9d86ef.png)

#### Imagem de profundidade medida (entrada 2):

![rawDepthNorm](https://user-images.githubusercontent.com/31515305/120489239-dd90ee00-c38d-11eb-9cb9-c106c01c5c1a.png)

 #### Imagem de profundidade processada (exemplo de saída):

![depthNorm](https://user-images.githubusercontent.com/31515305/120489234-dc5fc100-c38d-11eb-9d61-808ae1ffbcc6.png)


## Referência 
[1] Xian, C., Zhang, D., Dai, C., & Wang, C. C. (2020). Fast Generation of High-Fidelity RGB-D Images by Deep Learning With Adaptive Convolution. IEEE Transactions on Automation Science and Engineering. Disponível em https://arxiv.org/abs/2002.05067 
