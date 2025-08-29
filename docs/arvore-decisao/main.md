# 1. Matriz de Confusão

Mostra a performance do modelo.

O modelo acertou 281 previsões corretas (179 + 102).

Teve 119 erros (61 + 58).

Isso indica que o modelo tem uma boa taxa de acerto, mas ainda confunde alguns casos de pessoas que são fit e de quem não é fit.

👉 Isso pode estar ligado ao fato de que os dados não são totalmente balanceados (tem mais “not fit” que “fit”).

![Matriz de Confusão](../img/cm_baseline.png)

---

# 2. Distribuição de Fitness

O dataset tem mais pessoas Not Fit (0) (~60% da base) do que Fit (1) (~40%).

Isso mostra um leve desbalanceamento nos dados, o que pode influenciar o desempenho do modelo.

Olhando para gênero:

- Mulheres (F) têm maior proporção de Not Fit (0).
- Homens (M) estão mais equilibrados entre Fit e Not Fit.

👉 Ou seja, no conjunto analisado, há tendência maior de mulheres estarem “not fit”.

![Distribuição de Fitness](../img/Figure_1.png)

---

# 3. Árvore de Decisão

O nível de atividade física (`activity_index`) é o fator mais importante para definir a saúde/fitness.

Depois, surgem como variáveis decisivas:

- Tabagismo (`smokes`) → Fumantes têm mais chance de serem classificados como “not fit”.
- Peso (`weight_kg`) → Pesos mais altos aumentam o risco de “not fit”.
- Qualidade da nutrição (`nutrition_quality`) → Nutrição melhor contribui para ser “fit”.

👉 Isso reforça que o modelo usa hábitos de vida como principais indicadores de condição física.

![Árvore de Decisão](../img/tree_top_depth3.png)

---

## Resumo ampliado

O modelo consegue distinguir razoavelmente quem está fit e not fit, mas ainda erra uma parte considerável. Os dados mostram que existe mais gente “not fit”, principalmente entre as mulheres. A árvore de decisão destaca que atividade física, fumar, peso e qualidade da nutrição são os fatores que mais explicam a diferença entre estar ou não em boa condição física.