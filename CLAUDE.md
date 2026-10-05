# CLAUDE.md – Demonstração didática de CNN (MNIST)

Trabalho acadêmico de ADS sobre **CNN – Convolutional Neural Networks**. Este projeto é uma
demonstração visual, feita para ser apresentada em sala de aula (3 a 5 minutos) por pessoas sem
conhecimento avançado de IA. A prioridade é **clareza didática**, não desempenho.

## Arquivos

| Arquivo | Função |
|---|---|
| `cnn_demo.py` | **Fonte da verdade.** Células marcadas com `# %%` (código) e `# %% [markdown]` (texto). Roda no VS Code ("Run Cell") ou como script. |
| `cnn_demo.ipynb` | **Gerado** a partir do `.py` (compatível com Jupyter e Google Colab). Não editar à mão. |
| `gerar_notebook.py` | Converte `cnn_demo.py` em `cnn_demo.ipynb`. |
| `ROTEIRO_APRESENTACAO.md` | Roteiro falado de 3 a 5 minutos, célula por célula. |
| `requirements.txt` | Dependências. |

## Como executar

```bash
pip install -r requirements.txt
python cnn_demo.py              # abre uma janela do Matplotlib por figura
python gerar_notebook.py        # regenera o .ipynb após editar o .py
```

Teste sem interface gráfica (útil para verificar erros): `MPLBACKEND=Agg python cnn_demo.py`.
Python 3.13 + TensorFlow 2.21 / Keras 3 foram testados. O treino (5 épocas) leva ~1 min em CPU
e atinge ~98% de acurácia no teste. Não requer GPU.

## Fluxo demonstrado

`Imagem → Convolução → Feature Map → ReLU → Pooling → (repete) → Flatten → Dense → Classificação`

Estrutura das células de `cnn_demo.py` (manter esta ordem):

1. Importações e funções auxiliares (`mostrar_matriz`, `convolucao`)
2. Carregar MNIST
3. Visualizar a imagem (um "7"): dimensão, pixels 0–255, matriz recortada
4. Convolução manual: imagem 5x5, kernel 3x3, passos individuais e animação
5. Filtros diferentes (horizontal, vertical, Laplaciano, diagonal) no dígito real
6. ReLU (matriz do exemplo, gráfico e aplicação em feature map real)
7. Max Pooling 2x2 (matriz 4x4 do exemplo e imagem real)
8. Criação da CNN (Keras `Sequential`)
9. `model.summary()` e explicação das camadas
10. Treinamento (5 épocas, `validation_split=0.1`)
11. Gráficos de Accuracy e Loss
12. Avaliação no conjunto de teste
13. Previsões (10 imagens, confiança, grade de erros, barras de probabilidade)
14. Filtros aprendidos da `conv1`
15. Feature maps (imagem→filtro→mapa, etapa por etapa, 2ª camada)
16. Conclusão "O que aprendemos?" e diagrama do fluxo

## Arquitetura da CNN

`Input(28,28,1) → Conv2D(8,3x3) → ReLU → MaxPool(2) → Conv2D(16,3x3) → ReLU → MaxPool(2) →
Flatten(400) → Dense(64, relu) → Dense(10, softmax)`, ~27 mil parâmetros.
A ReLU é uma camada **separada** de propósito, para aparecer no `summary` e nos feature maps.
Os nomes das camadas (`conv1`, `relu1`, `pool1`, `conv2`, `relu2`, `pool2`, ...) são usados nas
células de visualização; se renomear, atualize também as células 14 e 15.

## Convenções

- **Idioma:** comentários, textos e títulos de gráficos em português.
- **Didática primeiro:** cada bloco deve explicar o que acontece matemática e conceitualmente.
  Nada de `model.fit(...)` sem comentário. Evitar conceitos avançados sem explicação.
- **Visual:** sempre que possível, mostrar figura (Matplotlib). Filtros/feature maps com mapa de
  cores divergente (`RdBu_r`: vermelho = positivo, azul = negativo).
- **Dependências mínimas:** apenas TensorFlow/Keras, NumPy e Matplotlib. Não adicionar outras.
- **Leve:** deve rodar em computador comum e no Google Colab, sem GPU.
- **Reprodutibilidade:** sementes fixas (`np.random.seed(42)`, `tf.random.set_seed(42)`).
- **Cuidado com variáveis globais:** o script é uma sequência de células que compartilham escopo
  (ex.: `n`, `M`, `fm`, `img7`). Ao adicionar código, evite sobrescrever nomes usados depois.
- **Fluxo de edição:** altere `cnn_demo.py` → rode `python gerar_notebook.py` → confirme que roda
  de ponta a ponta (`MPLBACKEND=Agg`) → atualize o `ROTEIRO_APRESENTACAO.md` se a ordem/conteúdo mudar.
- **Animação:** usa `FuncAnimation` + `to_jshtml()` quando em Jupyter/VS Code Notebook; em script
  puro abre janela do Matplotlib.

## Roteiro da apresentação

Está em `ROTEIRO_APRESENTACAO.md`. Rodar a célula de treinamento (item 10) **antes** de apresentar.
