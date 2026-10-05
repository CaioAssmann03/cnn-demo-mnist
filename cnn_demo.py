# %% [markdown]
# # CNN – Redes Neurais Convolucionais: demonstração visual
#
# **Fluxo que vamos percorrer:**
#
# `Imagem → Convolução → Feature Map → ReLU → Pooling → CNN → Classificação`
#
# Primeiro fazemos tudo **"na mão"** (com NumPy) para entender a matemática.
# Depois criamos uma **CNN real** com TensorFlow/Keras e treinamos no MNIST.
#
# > Como usar: no VS Code, clique em "Run Cell" em cada bloco (ou "Run All").
# > No Google Colab, basta copiar o arquivo `cnn_demo.ipynb`. Não precisa de GPU.

# %% [markdown]
# ## 1. Instalação / importação

# %%
# No Google Colab o TensorFlow já vem instalado. No computador local:
#   pip install tensorflow numpy matplotlib
# (nenhuma outra biblioteca é necessária)

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from numpy.lib.stride_tricks import sliding_window_view

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Garante que setas e acentos apareçam corretamente no terminal do Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Semente fixa: os resultados ficam (quase) iguais a cada execução
np.random.seed(42)
tf.random.set_seed(42)

plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 10
print("TensorFlow:", tf.__version__)


# --- Funções auxiliares de desenho (usadas em várias etapas) -------------------
def mostrar_matriz(ax, M, cmap="gray", vmin=None, vmax=None, fmt="{:g}",
                   fontsize=11, titulo=None):
    """Desenha uma matriz como imagem e escreve o número dentro de cada célula."""
    M = np.asarray(M, dtype=float)
    if vmin is None:
        vmin = np.nanmin(M)
    if vmax is None:
        vmax = np.nanmax(M)
    if vmin == vmax:
        vmax = vmin + 1
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    for (i, j), v in np.ndenumerate(M):
        if np.isnan(v):
            continue
        r, g, b, _ = cmap_obj((v - vmin) / (vmax - vmin))
        cor = "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "white"
        ax.text(j, i, fmt.format(v), ha="center", va="center", color=cor,
                fontsize=fontsize, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.8)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    if titulo:
        ax.set_title(titulo, fontsize=10)


def convolucao(img, kernel):
    """Convolução 2D 'valid' (sem preenchimento), stride 1.

    Para cada posição, pega uma janela do tamanho do kernel, multiplica
    elemento a elemento e SOMA tudo -> um único número (um pixel do mapa).
    Obs.: nas CNNs (e aqui) não se inverte o kernel; tecnicamente é uma
    'correlação cruzada', mas todo mundo chama de convolução.
    """
    kh, kw = kernel.shape
    janelas = sliding_window_view(img, (kh, kw))      # todas as janelas
    return np.einsum("ijkl,kl->ij", janelas, kernel)  # multiplica e soma


# %% [markdown]
# ## 2. Importação do MNIST
# O MNIST tem 70 mil imagens de dígitos escritos à mão (0 a 9), em escala de
# cinza, com **28 × 28 pixels**. São 60 mil para treino e 10 mil para teste.

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Treino:", x_train.shape, "| Teste:", x_test.shape)


# %% [markdown]
# ## 3. Visualização da imagem
# Uma imagem em escala de cinza é apenas uma **matriz de números de 0 a 255**:
# `0` = preto (fundo) e `255` = branco (traço do número).
# O computador não "vê" um 7 — ele vê essa tabela de números.

# %%
indice = int(np.where(y_train == 7)[0][0])       # primeira imagem que é um "7"
img7 = x_train[indice]

print("Dimensão da imagem:", img7.shape, "(altura x largura)")
print("Tipo dos pixels   :", img7.dtype)
print("Menor / maior pixel:", img7.min(), "/", img7.max())

# Recorte 10x10 ao redor do centro do traço, para caber na tela
ys, xs = np.nonzero(img7)
cy, cx = int(ys.mean()), int(xs.mean())
r0 = int(np.clip(cy - 5, 0, 18)); c0 = int(np.clip(cx - 5, 0, 18))
recorte = img7[r0:r0 + 10, c0:c0 + 10]

print("\nMatriz de pixels (recorte 10x10 da região do número):")
print(recorte)

print("\nVersão simplificada (# = pixel aceso, . = fundo):")
for linha in img7:
    print("".join("#" if p > 128 else "." for p in linha))

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
axs[0].imshow(img7, cmap="gray")
axs[0].add_patch(Rectangle((c0 - 0.5, r0 - 0.5), 10, 10, fill=False,
                           ec="red", lw=2))
axs[0].set_title(f"Imagem original – número {y_train[indice]}  (28x28)")
axs[0].axis("off")
mostrar_matriz(axs[1], recorte, cmap="gray", vmin=0, vmax=255, fontsize=7,
               titulo="Números que o computador enxerga (recorte vermelho)")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4. Demonstração manual da convolução
# Um **kernel** (filtro) é uma matriz pequena (aqui 3×3) que "desliza" pela imagem.
# Em cada posição:
# 1. O kernel é colocado sobre uma região da imagem;
# 2. Os valores são **multiplicados** um a um;
# 3. Os resultados são **somados**;
# 4. Essa soma vira **um pixel** do mapa de características (*feature map*);
# 5. O kernel anda uma casa e o processo se repete pela imagem toda.

# %%
imagem_toy = np.array([[0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0]], dtype=float)

kernel_h = np.array([[-1, -1, -1],
                     [ 0,  0,  0],
                     [ 1,  1,  1]], dtype=float)   # bordas horizontais

saida_toy = convolucao(imagem_toy, kernel_h)       # 5x5 com kernel 3x3 -> 3x3
n_out = saida_toy.shape[0]


def desenhar_passo(axs, img, kernel, saida, i):
    """Desenha o passo i da convolução (posição i do kernel, lendo em linhas)."""
    n = saida.shape[1]
    r, c = divmod(i, n)
    janela = img[r:r + 3, c:c + 3]
    produto = janela * kernel + 0.0            # +0.0 evita mostrar "-0"

    parcial = np.full(saida.shape, np.nan)         # só o que já foi calculado
    parcial.flat[:i + 1] = saida.flat[:i + 1]

    for ax in axs:
        ax.clear()
    mostrar_matriz(axs[0], img, vmin=0, vmax=1, titulo="Imagem (kernel em vermelho)")
    axs[0].add_patch(Rectangle((c - 0.5, r - 0.5), 3, 3, fill=False, ec="red", lw=4))
    mostrar_matriz(axs[1], kernel, cmap="RdBu_r", vmin=-1, vmax=1, fmt="{:+g}",
                   titulo="Kernel (filtro)")
    mostrar_matriz(axs[2], produto, cmap="RdBu_r", vmin=-1, vmax=1, fmt="{:+g}",
                   titulo=f"Janela × kernel\nSOMA = {produto.sum():+g}")
    mostrar_matriz(axs[3], parcial, cmap="RdBu_r", vmin=-3, vmax=3, fmt="{:+g}",
                   titulo="Mapa de características")
    axs[3].add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="red", lw=4))


# --- Alguns passos individuais ---------------------------------------------------
passos = [0, 1, 4, 8]
fig, eixos = plt.subplots(len(passos), 4, figsize=(13, 3.3 * len(passos)))
for linha, p in enumerate(passos):
    desenhar_passo(eixos[linha], imagem_toy, kernel_h, saida_toy, p)
    eixos[linha][0].set_ylabel(f"Passo {p + 1}/9", fontsize=12, rotation=0, labelpad=45)
fig.suptitle("Convolução passo a passo: posicionar → multiplicar → somar → gravar no mapa",
             fontsize=13)
plt.tight_layout()
plt.show()

# Conta detalhada do primeiro passo
janela0 = imagem_toy[0:3, 0:3]
print("Passo 1 – janela da imagem:\n", janela0)
print("Multiplicando pelo kernel:\n", janela0 * kernel_h)
print("Soma =", (janela0 * kernel_h).sum(), "-> primeiro pixel do mapa")
print("\nMapa de características completo:\n", saida_toy)
print("Positivo (+) = borda de baixo p/ cima detectada; negativo (-) = o contrário.")

# %%
# --- Animação (roda no Jupyter/VS Code Notebook; em .py aparece uma janela) -----
from matplotlib.animation import FuncAnimation

fig_anim, eixos_anim = plt.subplots(1, 4, figsize=(13, 3.6))
anim = FuncAnimation(fig_anim,
                     lambda i: desenhar_passo(eixos_anim, imagem_toy, kernel_h, saida_toy, i),
                     frames=n_out * n_out, interval=900, repeat=True)
fig_anim.tight_layout()

try:
    get_ipython()                          # estamos em Jupyter/Colab/VS Code Notebook?
    from IPython.display import HTML, display
    plt.close(fig_anim)
    display(HTML(anim.to_jshtml()))        # player com botão de play
except NameError:
    plt.show()                             # script normal: janela animada


# %% [markdown]
# ## 5. Demonstração dos filtros
# **Filtros diferentes detectam características diferentes.**
# Agora aplicamos 4 kernels ao número 7 real do MNIST.
# Cores: **vermelho** = resposta positiva forte, **azul** = negativa, **branco** = ~0.

# %%
kernels = {
    "Bordas horizontais": np.array([[-1, -1, -1], [ 0, 0, 0], [ 1, 1, 1]], dtype=float),
    "Bordas verticais":   np.array([[-1,  0,  1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
    "Bordas (todas – Laplaciano)": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float),
    "Diagonal":           np.array([[ 0,  1,  1], [-1, 0, 1], [-1, -1, 0]], dtype=float),
}

img7_norm = img7 / 255.0                     # 0..1 (mais fácil de trabalhar)
mapas = {nome: convolucao(img7_norm, k) for nome, k in kernels.items()}

fig, axs = plt.subplots(2, 5, figsize=(16, 6.8))
axs[0, 0].imshow(img7_norm, cmap="gray"); axs[0, 0].set_title("Imagem original")
axs[1, 0].axis("off")
for col, (nome, k) in enumerate(kernels.items(), start=1):
    mostrar_matriz(axs[0, col], k, cmap="RdBu_r", vmin=-2, vmax=2, fmt="{:+g}",
                   fontsize=9, titulo=f"Kernel:\n{nome}")
    m = mapas[nome]
    lim = np.abs(m).max()
    axs[1, col].imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[1, col].set_title("Feature map resultante")
for ax in axs[:, 0]:
    ax.set_xticks([]); ax.set_yticks([])
for ax in axs[1, 1:]:
    ax.axis("off")
fig.suptitle("Mesma imagem, kernels diferentes → características diferentes", fontsize=14)
plt.tight_layout()
plt.show()

print("Horizontal: destaca o 'traço de cima' do 7 | Vertical: destaca a haste inclinada.")
print("Kernel  →  identifica determinados padrões da imagem.")


# %% [markdown]
# ## 6. Demonstração da ReLU
# **ReLU(x) = max(0, x)**: negativos viram 0, positivos ficam iguais.
#
# Por que usar? (1) introduz **não-linearidade** — sem ela, empilhar camadas seria
# o mesmo que uma única conta linear e a rede não aprenderia formas complexas;
# (2) mantém só o que o filtro "encontrou" (valores positivos); (3) é muito simples
# e rápida de calcular.

# %%
antes = np.array([[-2,  3, -1],
                  [ 4, -5,  2],
                  [-3,  1,  6]], dtype=float)
depois = np.maximum(0, antes)                # ReLU(x) = max(0, x)

print("Antes da ReLU:\n", antes.astype(int))
print("Depois da ReLU:\n", depois.astype(int))

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
mostrar_matriz(axs[0], antes, cmap="RdBu_r", vmin=-6, vmax=6, fmt="{:+g}",
               fontsize=14, titulo="Antes (valores negativos em azul)")
mostrar_matriz(axs[1], depois, cmap="Reds", vmin=0, vmax=6, fmt="{:g}",
               fontsize=14, titulo="Depois: ReLU(x) = max(0, x)")
x = np.linspace(-6, 6, 200)
axs[2].plot(x, np.maximum(0, x), lw=3, color="crimson")
axs[2].axhline(0, color="gray", lw=0.8); axs[2].axvline(0, color="gray", lw=0.8)
axs[2].set_title("Gráfico da função ReLU"); axs[2].set_xlabel("x"); axs[2].set_ylabel("ReLU(x)")
plt.tight_layout()
plt.show()

# ReLU aplicada num feature map real (bordas horizontais do 7)
fm = mapas["Bordas horizontais"]
fm_relu = np.maximum(0, fm)
fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
lim = np.abs(fm).max()
axs[0].imshow(fm, cmap="RdBu_r", vmin=-lim, vmax=lim); axs[0].set_title("Feature map (com negativos)")
axs[1].imshow(fm_relu, cmap="gray"); axs[1].set_title("Após ReLU (só respostas positivas)")
for ax in axs: ax.axis("off")
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 7. Demonstração do Max Pooling
# Uma janela 2×2 percorre a matriz **sem sobreposição** e guarda só o **maior valor**
# de cada bloco. A imagem fica menor (metade da altura e da largura), mas as
# características mais fortes são mantidas. Resultado: menos cálculos e maior
# tolerância a pequenos deslocamentos do desenho.

# %%
M = np.array([[1, 3, 2, 4],
              [5, 6, 1, 2],
              [7, 2, 8, 3],
              [4, 1, 5, 9]], dtype=float)

# reshape agrupa em blocos 2x2; max pega o maior de cada bloco
pool = M.reshape(2, 2, 2, 2).max(axis=(1, 3))
print("Entrada 4x4:\n", M.astype(int))
print("Max Pooling 2x2 -> 2x2:\n", pool.astype(int))

cores = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [2, 1]})
mostrar_matriz(axs[0], M, cmap="Greys", vmin=0, vmax=12, fontsize=16,
               titulo="Entrada 4x4 – janelas 2x2")
mostrar_matriz(axs[1], pool, cmap="Greys", vmin=0, vmax=12, fontsize=16,
               titulo="Saída 2x2 (o maior de cada janela)")
for k, (r, c) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
    axs[0].add_patch(Rectangle((c - 0.5, r - 0.5), 2, 2, fill=False, ec=cores[k], lw=5))
    axs[1].add_patch(Rectangle((c // 2 - 0.5, r // 2 - 0.5), 1, 1, fill=False, ec=cores[k], lw=5))
plt.tight_layout(); plt.show()

# Pooling em imagem real: 26x26 -> 13x13, o traço continua reconhecível
n = fm_relu.shape[0] // 2 * 2
fm_pool = fm_relu[:n, :n].reshape(n // 2, 2, n // 2, 2).max(axis=(1, 3))
fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
axs[0].imshow(fm_relu, cmap="gray"); axs[0].set_title(f"Antes do pooling {fm_relu.shape}")
axs[1].imshow(fm_pool, cmap="gray"); axs[1].set_title(f"Depois do pooling {fm_pool.shape}")
for ax in axs: ax.axis("off")
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 8. Criação da CNN
# Agora deixamos de escolher os filtros: a rede **aprende sozinha** os valores
# dos kernels durante o treinamento.
#
# Preparação dos dados: dividimos por 255 (pixels de 0 a 1 ajudam o treino) e
# adicionamos a dimensão de "canais" (1 canal = cinza).

# %%
x_train_n = (x_train / 255.0).astype("float32")[..., np.newaxis]   # (60000, 28, 28, 1)
x_test_n = (x_test / 255.0).astype("float32")[..., np.newaxis]     # (10000, 28, 28, 1)

modelo = keras.Sequential([
    layers.Input(shape=(28, 28, 1), name="entrada"),

    # Conv2D: aplica 8 filtros 3x3 na imagem -> 8 feature maps
    layers.Conv2D(8, (3, 3), name="conv1"),
    # ReLU: zera valores negativos
    layers.ReLU(name="relu1"),
    # MaxPooling: reduz o tamanho pela metade
    layers.MaxPooling2D((2, 2), name="pool1"),

    # Segunda convolução: 16 filtros que combinam as características da 1ª
    layers.Conv2D(16, (3, 3), name="conv2"),
    layers.ReLU(name="relu2"),
    layers.MaxPooling2D((2, 2), name="pool2"),

    # Flatten: transforma os mapas em um vetor único de números
    layers.Flatten(name="flatten"),
    # Dense: camada "tradicional" que combina tudo para decidir
    layers.Dense(64, activation="relu", name="dense"),
    # Saída: 10 neurônios (dígitos 0-9); softmax transforma em probabilidades
    layers.Dense(10, activation="softmax", name="saida"),
], name="CNN_MNIST")

modelo.compile(optimizer="adam",                        # ajusta os pesos
               loss="sparse_categorical_crossentropy",  # mede o erro da classificação
               metrics=["accuracy"])                    # % de acertos


# %% [markdown]
# ## 9. Resumo da arquitetura
# | Camada | O que faz |
# |---|---|
# | **Input** | Recebe a imagem 28×28×1 |
# | **Conv2D (8 filtros 3×3)** | Detecta padrões simples (bordas, linhas). 28→26 pixels |
# | **ReLU** | Remove valores negativos, adiciona não-linearidade |
# | **MaxPooling 2×2** | Reduz o tamanho pela metade: 26→13 |
# | **Conv2D (16 filtros 3×3)** | Combina padrões simples em padrões maiores. 13→11 |
# | **ReLU + MaxPooling** | Idem: 11→5 |
# | **Flatten** | 5×5×16 vira um vetor de 400 números |
# | **Dense (64)** | Combina todas as características |
# | **Dense (10, softmax)** | Probabilidade de cada dígito (0 a 9) |

# %%
modelo.summary()
# "Param #" = quantos números a rede precisa aprender (pesos dos kernels e neurônios).
# Ex.: conv1 = 8 filtros x (3x3 pesos) + 8 bias = 80 parâmetros.


# %% [markdown]
# ## 10. Treinamento
# A cada época a rede vê todas as imagens de treino. Para cada lote (*batch*):
# faz uma previsão → calcula o erro (*loss*) → ajusta os pesos dos filtros para
# errar menos (*backpropagation* + otimizador Adam). Separamos 10% do treino para
# **validação**: imagens que a rede não usa para aprender, só para conferir.

# %%
historico = modelo.fit(x_train_n, y_train,
                       epochs=5,               # 5 passadas completas pelos dados
                       batch_size=128,         # 128 imagens por ajuste
                       validation_split=0.1,   # 10% para validação
                       verbose=2)


# %% [markdown]
# ## 11. Gráficos de treinamento
# - **Accuracy**: % de acertos (quanto maior, melhor).
# - **Loss**: tamanho do erro (quanto menor, melhor).
# Se treino e validação andam juntos, a rede está *generalizando*, e não decorando.

# %%
h = historico.history
epocas = range(1, len(h["accuracy"]) + 1)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
axs[0].plot(epocas, h["accuracy"], "o-", label="Treino")
axs[0].plot(epocas, h["val_accuracy"], "s-", label="Validação")
axs[0].set_title("Accuracy x Épocas"); axs[0].set_xlabel("Época"); axs[0].set_ylabel("Accuracy")
axs[0].set_xticks(list(epocas)); axs[0].grid(alpha=0.3); axs[0].legend()

axs[1].plot(epocas, h["loss"], "o-", label="Treino")
axs[1].plot(epocas, h["val_loss"], "s-", label="Validação")
axs[1].set_title("Loss x Épocas"); axs[1].set_xlabel("Época"); axs[1].set_ylabel("Loss")
axs[1].set_xticks(list(epocas)); axs[1].grid(alpha=0.3); axs[1].legend()
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 12. Teste
# Avaliamos com as 10 mil imagens de teste — nunca vistas pela rede.

# %%
perda_teste, acc_teste = modelo.evaluate(x_test_n, y_test, verbose=0)
print(f"Accuracy no teste: {acc_teste * 100:.2f}%")
print(f"Loss no teste    : {perda_teste:.4f}")


# %% [markdown]
# ## 13. Previsões
# A saída softmax dá uma probabilidade para cada dígito; o palpite é o maior.
# Essa probabilidade é a **confiança**. A CNN também erra — e vale observar os erros.

# %%
probs = modelo.predict(x_test_n, verbose=0)         # (10000, 10)
preditos = probs.argmax(axis=1)                     # dígito de maior probabilidade
confiancas = probs.max(axis=1)


def grade_previsoes(indices, titulo):
    fig, axs = plt.subplots(2, 5, figsize=(13, 6.5))
    for ax, i in zip(axs.ravel(), indices):
        acertou = preditos[i] == y_test[i]
        ax.imshow(x_test[i], cmap="gray")
        ax.set_title(f"Real: {y_test[i]}   Predito: {preditos[i]}\n"
                     f"Confiança: {confiancas[i] * 100:.1f}%",
                     color="green" if acertou else "red", fontsize=11)
        ax.axis("off")
        if not acertou:                             # moldura vermelha nos erros
            for lado in ax.spines.values():
                lado.set_visible(True); lado.set_edgecolor("red"); lado.set_linewidth(4)
    fig.suptitle(titulo, fontsize=14)
    plt.tight_layout(); plt.show()


grade_previsoes(range(10), "10 primeiras imagens do teste (verde = acerto, vermelho = erro)")

erros = np.where(preditos != y_test)[0]
print(f"A CNN errou {len(erros)} de {len(y_test)} imagens de teste.")
if len(erros) >= 10:
    grade_previsoes(erros[:10], "Erros da CNN: ela também se engana (dígitos ambíguos)")

# Probabilidades de uma imagem: como a decisão é tomada
i = 0
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(range(10), probs[i] * 100, color=["crimson" if d == preditos[i] else "steelblue" for d in range(10)])
ax.set_xticks(range(10)); ax.set_xlabel("Dígito"); ax.set_ylabel("Probabilidade (%)")
ax.set_title(f"Probabilidades para a imagem {i} (real = {y_test[i]}) – vence o maior")
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 14. Visualização dos filtros aprendidos
# Os kernels da 1ª camada começaram **aleatórios**. Depois do treino, a rede ajustou
# seus números sozinha. Nada foi programado à mão!
#
# **Simplificação didática** do que costuma acontecer:
# - Primeiras camadas → bordas, linhas, formas simples
# - Camadas intermediárias → combinações de características
# - Camadas mais profundas → padrões mais complexos
# - Camada final → classificação
#
# (Em uma rede pequena como esta, os filtros são só "sugestões" disso — na prática
# nem todo filtro tem uma interpretação limpa.)

# %%
pesos = modelo.get_layer("conv1").get_weights()[0]      # formato (3, 3, 1, 8)
print("Formato dos pesos da conv1:", pesos.shape, "-> 8 kernels de 3x3")

fig, axs = plt.subplots(2, 4, figsize=(13, 7))
for f, ax in enumerate(axs.ravel()):
    k = pesos[:, :, 0, f]
    lim = np.abs(k).max()
    mostrar_matriz(ax, k, cmap="RdBu_r", vmin=-lim, vmax=lim, fmt="{:+.2f}",
                   fontsize=9, titulo=f"Filtro {f + 1}")
fig.suptitle("Filtros 3x3 aprendidos pela 1ª camada (vermelho = peso +, azul = peso −)", fontsize=14)
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 15. Feature Maps
# Passamos uma imagem pela rede e "espiamos" o que sai de cada etapa.
# Cada feature map mostra **onde na imagem** o filtro correspondente encontrou o
# padrão que procura: regiões claras = padrão presente; escuras = ausente.

# %%
# Cria um modelo auxiliar que devolve as saídas intermediárias
nomes = ["conv1", "relu1", "pool1", "relu2", "pool2"]
espia = keras.Model(inputs=modelo.inputs,
                    outputs=[modelo.get_layer(n).output for n in nomes])

amostra = x_test_n[0:1]                                  # a 1ª imagem de teste
saidas = dict(zip(nomes, [s[0] for s in espia.predict(amostra, verbose=0)]))

# (a) Fluxo: imagem -> filtro -> feature map (3 primeiros filtros)
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
for f in range(3):
    axs[f, 0].imshow(x_test[0], cmap="gray"); axs[f, 0].set_title("Imagem original")
    k = pesos[:, :, 0, f]; lim = np.abs(k).max()
    mostrar_matriz(axs[f, 1], k, cmap="RdBu_r", vmin=-lim, vmax=lim, fmt="{:+.1f}",
                   fontsize=9, titulo=f"Filtro {f + 1}")
    axs[f, 2].imshow(saidas["relu1"][:, :, f], cmap="magma")
    axs[f, 2].set_title(f"Feature Map {f + 1}")
    axs[f, 0].axis("off"); axs[f, 2].axis("off")
    axs[f, 0].annotate("", xy=(1.08, 0.5), xytext=(1.0, 0.5), xycoords="axes fraction",
                       arrowprops=dict(arrowstyle="->", lw=2))
fig.suptitle("Imagem  →  Filtro  →  Feature Map", fontsize=14)
plt.tight_layout(); plt.show()

# (b) Todos os 8 filtros, etapa por etapa
etapas = [("Original", None), ("conv1", "Após Convolução"),
          ("relu1", "Após ReLU"), ("pool1", "Após Pooling")]
fig, axs = plt.subplots(3, 9, figsize=(18, 6.5))
for linha, (chave, rotulo) in enumerate(etapas[1:]):
    axs[linha, 0].imshow(x_test[0], cmap="gray"); axs[linha, 0].set_title("Original", fontsize=9)
    axs[linha, 0].set_ylabel(rotulo, fontsize=11, rotation=0, labelpad=60)
    for f in range(8):
        axs[linha, f + 1].imshow(saidas[chave][:, :, f], cmap="magma")
        axs[linha, f + 1].set_title(f"Filtro {f + 1}", fontsize=9)
for ax in axs.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Feature maps da 1ª camada: convolução → ReLU → pooling (tamanho 26 → 13)", fontsize=14)
plt.tight_layout(); plt.show()

# (c) 2ª camada: padrões mais abstratos, menor resolução (5x5)
fig, axs = plt.subplots(2, 8, figsize=(16, 4.5))
for f, ax in enumerate(axs.ravel()):
    ax.imshow(saidas["pool2"][:, :, f], cmap="magma")
    ax.set_title(f"Mapa {f + 1}", fontsize=9); ax.axis("off")
fig.suptitle("2ª camada (após ReLU + pooling): 16 mapas de 5x5 – bem mais abstratos", fontsize=14)
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 16. Conclusão – O que aprendemos?
# 1. **CNN**: rede neural feita para dados com estrutura de grade (imagens). Em vez
#    de olhar cada pixel isolado, olha *pequenas regiões* e reaproveita os mesmos filtros.
# 2. **Convolução**: deslizar um kernel pela imagem, multiplicando e somando em cada posição.
# 3. **Kernel/filtro**: matriz pequena de números que funciona como um "detector" de um padrão.
# 4. **Feature Maps**: o resultado de aplicar um filtro — mostra *onde* o padrão aparece.
# 5. **ReLU**: zera valores negativos; dá não-linearidade para a rede aprender formas complexas.
# 6. **Pooling**: reduz o tamanho mantendo o mais importante; menos cálculo e mais tolerância a deslocamentos.
# 7. **Aprendizado dos filtros**: os kernels começam aleatórios; a cada lote a rede mede o erro
#    e ajusta os números (backpropagation) para errar menos.
# 8. **Classificação**: os mapas viram um vetor (Flatten), a camada Dense combina tudo e o
#    softmax dá a probabilidade de cada classe; a maior vence.

# %%
# Fluxo completo com os tamanhos REAIS da CNN treinada
etapas_fluxo = [
    ("IMAGEM", "28 x 28 x 1", "#cfd8dc"),
    ("CONVOLUÇÃO", "26 x 26 x 8", "#90caf9"),
    ("FEATURE MAP", "8 mapas de características", "#80cbc4"),
    ("RELU", "negativos → 0", "#ffcc80"),
    ("POOLING", "13 x 13 x 8", "#ef9a9a"),
    ("CONVOLUÇÃO", "11 x 11 x 16", "#90caf9"),
    ("FEATURE MAP", "16 mapas (5 x 5 após ReLU + Pooling)", "#80cbc4"),
    ("FLATTEN", "vetor de 400 números", "#ce93d8"),
    ("DENSE", "64 neurônios", "#a5d6a7"),
    ("CLASSIFICAÇÃO", "10 probabilidades → dígito 0-9", "#fff59d"),
]
n_et = len(etapas_fluxo)
fig, ax = plt.subplots(figsize=(7, 11))
ax.set_xlim(0, 10); ax.set_ylim(0, n_et * 1.5); ax.axis("off")
for i, (nome, det, cor) in enumerate(etapas_fluxo):
    y = (n_et - 1 - i) * 1.5 + 0.2
    ax.add_patch(FancyBboxPatch((1.5, y), 7, 0.95, boxstyle="round,pad=0.05",
                                fc=cor, ec="black", lw=1.5))
    ax.text(5, y + 0.6, nome, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(5, y + 0.25, det, ha="center", va="center", fontsize=9)
    if i < n_et - 1:
        ax.add_patch(FancyArrowPatch((5, y - 0.05), (5, y - 0.4), arrowstyle="-|>",
                                     mutation_scale=20, lw=2, color="black"))
ax.set_title("Fluxo completo de uma CNN", fontsize=15, fontweight="bold")
plt.tight_layout(); plt.show()

print("""
IMAGEM
  ↓
CONVOLUÇÃO
  ↓
FEATURE MAP
  ↓
RELU
  ↓
POOLING
  ↓
CONVOLUÇÃO
  ↓
FEATURE MAP
  ↓
FLATTEN
  ↓
DENSE
  ↓
CLASSIFICAÇÃO
""")
