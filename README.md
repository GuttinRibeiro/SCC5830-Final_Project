# Preenchimento de profundidade em imagens RGB-D
Este repositório contém o trabalho final da disciplina SCC5830 - Processamento de Imagens cursada no primeiro semestre de 2021 no ICMC USP.
#### Autor: Augusto Ribeiro Castro - 9771421

## Abstract

Imagens RGB-D são uma combinação de uma imagem RGB e a correspondente imagem de profundidade, que guarda, para cada pixel, a distância entre o objeto no ambiente e o plano da imagem. Além da opção de utilizar técnicas computacionais para o cálculo da profundidade da imagem a partir de imagens RGB (visão estéreo ou estimativa monocular utilizando aprendizado profundo, por exemplo), é possível também obter essa informação do ambiente por uma combinação de sensores, como ocorre no Kinect, em que dados da câmera e de um sensor de profundidade são combinados. 

No caso da obtenção por meio de um hardware, apesar da capacidade de obtenção de imagens em tempo real e a um custo relativamente baixo, os dados obtidos apresentam falhas como ruídos de medição que prejudicam a definição dos limites físicos dos objetos e, muitas vezes, a ausência da medição de profundidade para algumas regiões, devido à reflexão em superfícies irregulares ou espelhadas. Sendo assim, como etapa anterior à utilização da profundidade crua de uma imagem RGB-D obtida por um hardware específico, é necessária a aplicação de técninas de "image completion" para preencher as informações faltantes. Concluída a etapa prévia, as imagens RGB-D se tornam valiosas informações para tarefas em robótica, detectção de esqueleto, detecção de objetos e mapeamento de ambientes.

No contexto do pré-processamento da informação de profundidade de imagens RGB-D, este trabalho busca então implementar uma maneira de completar regiões em que não foi possível obter a medição de profundidade, além de possíveis etapas posteriores de melhoramento da imagem. Como fonte de dados e referência teórica é possível citar o trabalho de XIAN, Chuhua, et al. 2020 [1], cujos dados podem ser encontrados em https://github.com/chuhuaxian/HF-RGBD. Apesar de existirem outros conjuntos de dados para a tarefa, como o KITTI e o Matterplot3D, o conjunto de dados do artigo citado será o abordado neste trabalho.

## Objetivo do projeto

Este projeto tem como finalidade produzir imagens de profundidade de melhor qualidade a partir de imagens de ambientes internos capturadas por um sensor Kinect no formado RGB-D. Para isso, será utilizado o dataset do artigo [1], que contém o par de imagem RGB e de profundidade para 1302 cenas, bem como a imagem de profundidade de saída de alta qualidade produzida pelo estudo e assumida como verdadeira. Os principais desafios são completar as regiões que não puderam ser medidas pelo sensor e buscar gerar contornos bem definidos.

## Descrição das imagens do conjunto de dados

As imagens foram obtidas do repositório https://github.com/chuhuaxian/HF-RGBD. No caso das imagens de profundidade, os valores foram normalizados para serem inteiros e estarem no intervalo [0, 255] apenas para fins de visualização. Por serem capturadas por um kinect, os valores de profundidade são dados em milímetros no intervalo [500, 8000].

#### Imagem RGB (entrada 1):

![00000-color](https://user-images.githubusercontent.com/31515305/120489256-e1247500-c38d-11eb-82b2-a9606e9d86ef.png)

#### Imagem de profundidade medida (entrada 2):

![rawDepthNorm](https://user-images.githubusercontent.com/31515305/120489239-dd90ee00-c38d-11eb-9cb9-c106c01c5c1a.png)

 #### Imagem de profundidade processada (saída desejada):

![depthNorm](https://user-images.githubusercontent.com/31515305/120489234-dc5fc100-c38d-11eb-9d61-808ae1ffbcc6.png)

## Etapas e métodos utilizados

O problema proposto é um exemplo de aplicação de redes neurais convolucionais. Com relação à arquitetura dessas redes, o problema é um exemplo em que redes auto-codificadoras podem ser utilizadas, uma vez que a saída esperada é uma reconstrução da entrada fornecida. Além disso, o cenário proposto também é um exemplo de situação em que o modelo desenvolvido busca realizar uma regressão, já que o objetivo é calcular o valor de profundidade para cada pixel. Para isso, é possível utilizar como função de perda a ser minimizada o erro absoluto médio (MAE). Entretanto, [2] apresenta dois outros componentes a serem minimizados numa tarefa de predição de profundidade que ajudam a obter melhores resultados e que estão relacionados ao contexto da disciplina: os gradientes e a similaridade estrutural. Sendo assim, a função de perda utilizada para treinamento é uma composição do MAE com a dissimilaridade estrutural e com a soma da diferença dos gradientes da imagem de profundidade gerada e a imagem de profundidade correta.

A fim de priorizar o preenchimento dos buracos e evitar que essas regiões fossem preenchidas apenas com um valor médio que minimizaria o erro global, mas sem apresentar um bom resultado local, o MAE foi ponderado pela transformada da distância. Para isso, foi calculada uma imagem binária a partir da profundidade medida indicando as regiões com buracos. Assim, é obtido um mapa com a distância de dos pixels de cada buraco até a borda, dando mais peso para o centro dessas regiões, como mostrado abaixo.

![unnamed](https://user-images.githubusercontent.com/31515305/125619382-2c59f7b5-cab6-4eb7-a85d-2c38357d70d7.png)
![unnamed2](https://user-images.githubusercontent.com/31515305/125619596-d4f3e92f-117d-4b9c-8740-330722a137fa.png)

A rede neural empregada é então composta por 5 blocos de codificação e 5 de decodificação. A entrada da sequência de codificação é uma imagem RGB-D (4 canais concatenados) de resolução 384x544x4 e a saída é a imagem de profundidade de resolução 384x544x1. A cada bloco de codificação, o número de canais da imagem é dobrado, enquanto a dimensão espacial é dividida pela metade graças a aplicação de camadas de redução de regiões 2x2 pelo valor médio. São utilizados também, a normalização em lotes e a convolução dos valores da entrada por um filtro de pesos a ser treinado. Todas as funções de ativação empregadas são a LeakyReLU com inclinação igual a -0,01 para valores negativos, a fim de evitar que alguns neurônios parem de atuar caso o valor dos pesos se torne negativo.

Durante a decodificação, as imagens são interpoladas bilinearmente para dobrar as dimensões espaciais do tensor de entrada. Para reduzir pela metade o número de canais, uma convolução 1x1 é aplicada. Além disso, os blocos de decodificação utilizam também a entrada concatenada de blocos anteriores de codificação que possuem as mesmas dimensões espaciais, a fim de propagar melhor detalhes da imagem ao longo da rede. No bloco é ainda aplicada uma convolução 3x3 sem alterar as dimensões da imagem, com a mesma função de ativação citada anteriormente. Esses passos se repetem ao longo dos blocos de decodificação, até que, na última etapa, uma convolução 1x1 restaura a profundidade a partir de um tensor formado pela saída do último bloco de decodificação e pela imagem RGB-D dada como entrada. 

## Resultados

O modelo apresentado no código foi treinado por 70 épocas de treinamento em 90% das tuplas de imagens do conjunto de dados. Na figura abaixo é possível ver os gráficos da função de perda e das suas componentes ao longo do tempo decaírem e atingirem uma região de pouca mudança, sugerindo que, para o cenário proposto, o número de épocas de treinamento empregado é satisfatório.

![Captura de tela 2021-07-14 091056](https://user-images.githubusercontent.com/31515305/125621233-93961a57-b2ba-4aa2-b17d-cf550c8c957f.png)

Com relação aos resultados obtidos, a imagem abaixo mostra a saída para um conjunto de amostras extraído do conjunto de testes. Na primeira linha são mostradas as componentes RGB da entrada, seguidas pelas medições de profundidade na próxima linha. Na terceira linha, aparecem as saídas calculadas pelo modelo, enquanto que as últimas quatro imagens mostram os valores de profundidade assumidos como verdadeiros. Como as imagens de profundidade são normalizadas para o intervalo [0, 255] durante o processo de plotagem, algumas tonalidades podem ser diferentes entre as imagens com e sem buracos.

![grid](https://user-images.githubusercontent.com/31515305/125622657-424132b5-d3e8-4e6c-8233-34290c9997ef.png)

Abaixo são mostrados de forma mais detalhados os resultados para duas outras imagens. Para ambas as cenas, é possível notar que os buracos são preenchidos, mas os contornos não são tão bem definidos quanto aqueles apresentados pela imagem de profundidade verdadeira. Além disso, os limites mostrados de valores de profundidade presentes na cena são diferentes. 

Apesar disso, houve uma redução do MAE entre a imagem de profundidade medida e a completada com relação à profundidade real de 1289,6816 para 78,068855 e de 1687,5479 para 102,43216 na segunda cena. Esses resultados não necessariamente refletem a qualidade do preenchimento, pois são referentes à imagem como um todo e não apenas aos buracos. Também é preciso notar que o simples fato de preencher os buracos com um valor de profundidade médio estimado, sem muita atenção aos detalhes, já seria capaz de reduzir o MAE. Sendo assim, um dos principais desafios deste problema é encontrar a métrica ideal para treinar e avaliar o modelo desenvolvido.

![índice](https://user-images.githubusercontent.com/31515305/125623252-e3daaca6-a7c1-4fe9-a589-b98fe9791a12.png)

![índice2](https://user-images.githubusercontent.com/31515305/125623256-7cfcab1c-f692-48e0-b5cc-76e33f615f87.png)

## Referências
[1] Xian, C., Zhang, D., Dai, C., & Wang, C. C. (2020). Fast Generation of High-Fidelity RGB-D Images by Deep Learning With Adaptive Convolution. IEEE Transactions on Automation Science and Engineering. Disponível em https://arxiv.org/abs/2002.05067 

[2] Alhashim, I., & Wonka, P. (2018). High quality monocular depth estimation via transfer learning. Disponivel em https://arxiv.org/pdf/1812.11941.pdf

