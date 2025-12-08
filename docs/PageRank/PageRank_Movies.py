"""
PageRank em rede de filmes (The Movies Dataset - credits + movies_metadata)

- Cada nó é um filme
- Criamos arestas entre filmes que compartilham pelo menos
  um ator principal OU diretor
- Grafo é transformado em direcionado (arestas bidirecionais)
- Implementamos PageRank do zero (power iteration) e
  comparamos com networkx.pagerank
"""

import os
import ast
import json
import itertools
import collections

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# =========================================================
# 1. CARREGAR E PREPARAR O DATASET
# =========================================================

# Ajuste este caminho para a pasta onde estão os CSVs
base_path = r"C:\Users\mateus.colmeal\Documents\GitHub\machine-learning\docs\PageRank\data"

credits_path = os.path.join(base_path, "credits.csv")
meta_path = os.path.join(base_path, "movies_metadata.csv")

print("Carregando CSVs...")
credits = pd.read_csv(credits_path)
meta = pd.read_csv(meta_path, low_memory=False)

# Converter id para string para poder fazer merge
credits["id"] = credits["id"].astype(str)
meta["id"] = meta["id"].astype(str)

# Mantemos apenas colunas necessárias de movies_metadata
meta_small = meta[["id", "title", "vote_count"]]

# Merge por id
df = credits.merge(meta_small, on="id", how="inner")

# Converter vote_count pra numérico e tirar NaN
df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
df = df.dropna(subset=["vote_count"])

# Para não ficar gigante, vamos pegar apenas os filmes mais "populares"
N_TOP = 300
df_top = df.sort_values("vote_count", ascending=False).head(N_TOP).reset_index(drop=True)
pd.set_option("display.max_rows", None)
print(df_top[["id", "title", "vote_count"]])
print(f"Número de filmes considerados: {len(df_top)}")

# Dicionário id -> título (pra interpretar depois)
id_to_title = dict(zip(df_top["id"], df_top["title"]))


# =========================================================
# 2. CONSTRUIR O GRAFO A PARTIR DE ELENCO/DIRETOR
# =========================================================

def safe_load_list(x):
    """
    Converte string de lista de dicionários (JSON) em lista Python.
    Trato casos com aspas simples/duplas.
    """
    if not isinstance(x, str):
        return []
    try:
        # muitos créditos vêm com aspas simples, então trocamos por duplas
        return json.loads(x.replace("'", '"'))
    except Exception:
        try:
            return ast.literal_eval(x)
        except Exception:
            return []


person_to_movies = collections.defaultdict(set)

for _, row in df_top.iterrows():
    mid = row["id"]

    cast_list = safe_load_list(row["cast"])
    crew_list = safe_load_list(row["crew"])

    # Pegamos só os 3 primeiros atores para não explodir o grafo
    cast_names = [
        c.get("name")
        for c in cast_list[:3]
        if isinstance(c, dict) and c.get("name")
    ]

    # Diretores
    dir_names = [
        c.get("name")
        for c in crew_list
        if isinstance(c, dict) and c.get("job") == "Director" and c.get("name")
    ]

    persons = set(cast_names + dir_names)

    for p in persons:
        person_to_movies[p].add(mid)

print(f"Número de pessoas (atores/diretores) consideradas: {len(person_to_movies)}")

# Criar grafo não direcionado de filmes
G_undirected = nx.Graph()
for mid in df_top["id"]:
    G_undirected.add_node(mid)

# Para cada pessoa, conectamos todos os filmes em que ela trabalhou
for person, movies in person_to_movies.items():
    movies = list(movies)
    if len(movies) < 2:
        continue  # pessoa que só participou de 1 filme não cria aresta
    for a, b in itertools.combinations(movies, 2):
        # Podemos guardar a pessoa como atributo da aresta (opcional)
        if not G_undirected.has_edge(a, b):
            G_undirected.add_edge(a, b, persons={person})
        else:
            G_undirected[a][b]["persons"].add(person)

print("Nós no grafo (filmes):", G_undirected.number_of_nodes())
print("Arestas (conexões por pessoa em comum):", G_undirected.number_of_edges())

# Transformar para direcionado para usar PageRank canônico
G = G_undirected.to_directed()


# =========================================================
# 3. IMPLEMENTAÇÃO DO PAGERANK DO ZERO (POWER ITERATION)
# =========================================================

def pagerank_power_iteration(G, d=0.85, tol=1e-6, max_iter=100):
    """
    Implementação do PageRank pela fórmula iterativa:

    PR(i) = (1-d)/N + d * sum_j( PR(j) / L(j) )

    onde:
      - N é o número de nós
      - L(j) é o número de links de saída do nó j
      - a soma percorre os nós j que apontam para i

    Aqui usamos a forma matricial:
      p_{k+1} = d * M * p_k + (1-d)/N
    onde M é a matriz de transição coluna-estocástica.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    # Matriz de transição M (coluna-estocástica)
    M = np.zeros((n, n), dtype=float)

    for j, node_j in enumerate(nodes):
        out_neighbors = list(G.successors(node_j))
        if len(out_neighbors) == 0:
            # nó "dangling" (sem saída) distribui igualmente pra todos
            M[:, j] = 1.0 / n
        else:
            prob = 1.0 / len(out_neighbors)
            for node_i in out_neighbors:
                i = idx[node_i]
                M[i, j] = prob

    # Vetor inicial uniforme
    p = np.ones(n) / n

    for it in range(max_iter):
        p_new = d * (M @ p) + (1 - d) / n
        diff = np.linalg.norm(p_new - p, 1)

        p = p_new
        if diff < tol:
            print(f"[d={d}] Convergiu em {it+1} iterações (dif={diff:.2e})")
            break
    else:
        print(f"[d={d}] NÃO convergiu até max_iter (dif={diff:.2e})")

    # Normaliza (só por segurança)
    p = p / p.sum()

    return {node: p[idx[node]] for node in nodes}


# =========================================================
# 4. PAGERANK PARA d = 0.85 E COMPARAÇÃO COM NETWORKX
# =========================================================

d_default = 0.85
pr_manual = pagerank_power_iteration(G, d=d_default)
pr_nx = nx.pagerank(G, alpha=d_default)

# Comparar diferença
nodes = list(G.nodes())
diffs = [abs(pr_manual[n] - pr_nx[n]) for n in nodes]
print(f"Máxima diferença manual x networkx (d={d_default}): {max(diffs):.2e}")

# Top 10 filmes por PageRank (manual)
top_10 = sorted(pr_manual.items(), key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 filmes por PageRank (d=0.85):")
for mid, score in top_10:
    title = id_to_title.get(mid, "(sem título)")
    print(f"{mid}\t{title}\t{score:.4f}")


# =========================================================
# 5. VISUALIZAÇÕES (PARA O RELATÓRIO)
# =========================================================

images_dir = os.path.join(base_path, "images")
os.makedirs(images_dir, exist_ok=True)
graph_img = os.path.join(images_dir, f"graph_pagerank_d{d_default}.png")
top10_img = os.path.join(images_dir, "top10_pagerank_d0_85.png")

# --- 5.1. Grafo com tamanho do nó proporcional ao PageRank ---
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G_undirected, seed=42)

node_sizes = [3000 * pr_manual[n] for n in G_undirected.nodes()]
node_colors = [pr_manual[n] for n in G_undirected.nodes()]

# desenhar no eixo ax
nx.draw_networkx_nodes(
    G_undirected, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap="viridis",
    ax=ax
)
nx.draw_networkx_edges(G_undirected, pos, alpha=0.3, ax=ax)

# labels apenas para os maiores valores
labels_small = {n: id_to_title.get(n, "") for n, v in pr_manual.items()
                if v >= np.percentile(list(pr_manual.values()), 90)}

nx.draw_networkx_labels(G_undirected, pos, labels=labels_small, font_size=7, ax=ax)

# criar o mappable e adicionar colorbar corretamente
sm = plt.cm.ScalarMappable(cmap="viridis")
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Valor de PageRank")

ax.set_title(f"PageRank em rede de filmes (d={d_default})")
ax.set_axis_off()

plt.tight_layout()
plt.savefig(graph_img, dpi=300)
plt.close()


# --- 5.2. Barra dos Top 10 filmes ---
labels_top10 = [id_to_title.get(mid, mid) for mid, _ in top_10]
scores_top10 = [s for _, s in top_10]

plt.figure(figsize=(10, 4))
plt.bar(range(len(scores_top10)), scores_top10)
plt.xticks(range(len(scores_top10)), labels_top10, rotation=45, ha="right")
plt.ylabel("PageRank")
plt.title("Top 10 filmes por PageRank (d=0.85)")
plt.tight_layout()
top10_img = os.path.join(images_dir, "top10_pagerank_d0_85.png")
plt.savefig(top10_img, dpi=300)
plt.close()

print("\nImagens salvas em:")
print(" -", graph_img)
print(" -", top10_img)


# =========================================================
# 6. VARIAÇÃO DO FATOR d (0.5, 0.85, 0.99)
# =========================================================

ds = [0.5, 0.85, 0.99]
pr_by_d = {}

for d in ds:
    print(f"\nCalculando PageRank para d={d}...")
    pr_by_d[d] = pagerank_power_iteration(G, d=d)
    # checar diferença com networkx (só para garantia)
    pr_nx_d = nx.pagerank(G, alpha=d)
    diffs_d = [abs(pr_by_d[d][n] - pr_nx_d[n]) for n in nodes]
    print(f"Máx diferença manual x networkx (d={d}): {max(diffs_d):.2e}")

# Tabela textual de variação para os mesmos top10 de d=0.85
print("\nVariação do PageRank dos Top 10 filmes conforme d:")
print("Filme\t\t d=0.5\t d=0.85\t d=0.99")
for mid, _ in top_10:
    vals = [pr_by_d[d][mid] for d in ds]
    title = id_to_title.get(mid, mid)
    print(f"{title[:25]:25s}\t" + "\t".join(f"{v:.4f}" for v in vals))

# --- 6.1. Distribuição dos valores de PageRank para diferentes d ---
plt.figure(figsize=(8, 4))
for d in ds:
    scores_sorted = sorted(pr_by_d[d].values(), reverse=True)
    plt.plot(scores_sorted, label=f"d={d}")
plt.title("Distribuição dos valores de PageRank para diferentes d")
plt.xlabel("Nós (ordenados por PageRank)")
plt.ylabel("PageRank")
plt.legend()
plt.tight_layout()
dist_img = os.path.join(images_dir, "pagerank_distribution_by_d.png")
plt.savefig(dist_img, dpi=300)
plt.close()

print("\nImagem da distribuição salva em:", dist_img)