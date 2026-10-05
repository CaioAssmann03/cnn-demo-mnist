# Roteiro de apresentação (3 a 5 minutos)

Execute uma célula por vez (`Run Cell` no VS Code). Rode a célula do **treinamento (item 10) antes de começar**
a apresentação, ou logo no início, pois leva cerca de 1 minuto.

| Tempo | Célula | O que falar |
|---|---|---|
| 0:00 | **Título / Imports** | "Hoje vou mostrar como uma CNN, ou Rede Neural Convolucional, 'enxerga' uma imagem. Primeiro faremos as contas na mão e depois veremos uma rede real aprendendo." |
| 0:20 | **3. Imagem** | "Para o computador, isto não é um 7: é uma tabela de números de 0 a 255. Preto é 0 e branco é 255. A CNN trabalha em cima dessa matriz." |
| 0:45 | **4. Convolução manual** (figura de 4 passos e animação) | "Um filtro, ou *kernel*, é uma matriz pequena que desliza pela imagem. Em cada posição ele multiplica, soma, e o resultado vira um pixel do mapa de características. Aqui o filtro detecta bordas horizontais: onde há transição de escuro para claro, o valor é alto." |
| 1:30 | **5. Filtros** | "Cada filtro detecta um padrão diferente. Este destaca o traço horizontal do 7, este a haste vertical, este todas as bordas. Kernel diferente, característica diferente." |
| 2:00 | **6. ReLU** | "A ReLU zera tudo que é negativo: max(0, x). Ela faz a rede guardar só o que foi detectado e dá não-linearidade, permitindo aprender formas complexas." |
| 2:15 | **7. Pooling** | "O Max Pooling pega o maior valor de cada bloco 2×2. A imagem fica pela metade, mas o essencial permanece. Isso deixa a rede mais rápida e tolerante a pequenas variações." |
| 2:35 | **8–9. CNN + summary** | "Agora a rede real: Conv → ReLU → Pooling, duas vezes, depois Flatten, Dense e a saída com 10 neurônios, um por dígito. Ela tem cerca de 27 mil parâmetros; a diferença é que os filtros ela aprende sozinha." |
| 3:00 | **11. Gráficos** (treino já feito) | "Em 5 épocas a acurácia passou de 88% para cerca de 98%, e o erro caiu. Treino e validação andam juntos: ela está generalizando, e não decorando." |
| 3:20 | **13. Previsões** | "Aqui as previsões com a confiança. Em verde os acertos. Em vermelho os erros: mesmo com 98% de acerto, a CNN se engana em dígitos ambíguos." |
| 3:45 | **14–15. Filtros e Feature Maps** | "Estes são os filtros que a rede aprendeu sozinha; começaram aleatórios. Aqui vemos a imagem passando por eles e gerando os mapas: cada um mostra *onde* um padrão apareceu. Nas primeiras camadas são bordas e linhas; nas mais profundas, padrões mais abstratos. Essa é uma simplificação didática." |
| 4:30 | **16. Conclusão / fluxo** | "Resumindo: Imagem → Convolução → Feature Map → ReLU → Pooling → Convolução → Flatten → Dense → Classificação. A rede aprende os filtros ajustando os pesos para errar menos, e a classificação final é a maior probabilidade." |

## Dicas
- Se o tempo for curto, pule a animação e os mapas da 2ª camada (item 15c).
- A animação da convolução só aparece com player em Jupyter/notebook; no `.py` puro abre uma janela do Matplotlib.
- Pergunta provável: *"Por que CNN e não uma rede comum?"* — Porque os mesmos filtros são reaproveitados na imagem toda (poucos parâmetros) e preservam a vizinhança dos pixels.
