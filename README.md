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

## Etapas e métodos a serem utilizados

O problema proposto é um exemplo de aplicação de redes neurais convolucionais. Com relação à arquitetura dessas redes, o problema é um exemplo em que redes auto-codificadoras podem ser utilizadas, uma vez que a saída esperada é uma reconstrução da entrada fornecida. Além disso, o cenário proposto também é um exemplo de situação em que o modelo desenvolvido busca realizar uma regressão, já que o objetivo é calcular o valor de profundidade para cada pixel. Para isso, é possível utilizar como função de perda a ser minimizada o erro quadrático médio. Entretanto, [2] apresenta dois outros componentes a serem minimizados numa tarefa de predição de profundidade que ajudam a obter melhores resultados e que estão relacionados ao contexto da disciplina: os gradientes e a similaridade estrutural.

Até o momento, um modelo preliminar foi implementado para completar a profundidade utilizando apenas a componente de profundidade da imagem RGB-D como entrada. Foi utilizada uma rede neural convolucional em forma de U e o erro quadrático médio foi minimizado. Apesar disso, maneiras para utilizar a componente RGB estão sendo estudadas e vão desde a possibilidade de estimar a profundidade também a partir da componente RGB [2] e utilizar os valores calculados nas regiões faltantes nas medições do sensor ou utilizar a imagem RGB numa etapa de refinamento dos resultados obtidos pelo modelo inicial (profundidade para profundidade), conforme é feito em [1]. Por fim, dados os resultados preliminares obtidos, o foco recai sobre melhorar a previsão feita para as arestas das imagens. É nessa etapa que podem entrar a similaridade estrutural e o filtro Sobel implementado.

## Resultados preliminares

O modelo simples apresentado no código foi treinado por apenas 10 épocas de treinamento, já que o projeto ainda está em fase de construção e treinamentos longos acabam consumindo tempo demais até ser possível observar algum resultado promissor. Apenas a título de comparação, o modelo completo e mais elaborado apresentado em [1] foi treinado por 100 épocas neste mesmo conjunto de dados. Sendo assim, por definição, o erro obtido pela estimativa apresentada ainda é muito alto. Até o ponto em que o treinamento foi executado, o erro quadrático médio da previsão era da ordem de 200 mm. Entretanto, já é possível notar que, para o exemplo abaixo, que foi retirado de um conjunto fora do conjunto de treinamento, as regiões faltantes começaram a ser preenchidas e a ordem de grandeza da profundidade dos objetos (cor das poltronas por exemplo) parece mais alinhada com aquela apresentada pelo valor desejado.

#### Imagem colorida da cena: 
![00378-color](https://user-images.githubusercontent.com/31515305/123012443-a0ba8480-d398-11eb-897d-21e67259b9f6.png)

#### Medição de profundidade do sensor:
![label](https://user-images.githubusercontent.com/31515305/123012465-ab751980-d398-11eb-9f73-77e48d9d175c.png)

#### Profundidade preenchida pelo modelo preliminar:
![prediction](https://user-images.githubusercontent.com/31515305/123012477-b039cd80-d398-11eb-9095-8be1119d3adc.png)

#### Profundidade verdadeira:
![índice](https://user-images.githubusercontent.com/31515305/123012478-b2039100-d398-11eb-916b-b9e331110e41.png)


## Referências
[1] Xian, C., Zhang, D., Dai, C., & Wang, C. C. (2020). Fast Generation of High-Fidelity RGB-D Images by Deep Learning With Adaptive Convolution. IEEE Transactions on Automation Science and Engineering. Disponível em https://arxiv.org/abs/2002.05067 

[2] Alhashim, I., & Wonka, P. (2018). High quality monocular depth estimation via transfer learning. Disponivel em https://arxiv.org/pdf/1812.11941.pdf

