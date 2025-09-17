K-Nearest Neighbors:

O KNN (K-Nearest Neighbors) é um algoritmo de aprendizado supervisionado usado tanto para classificação quanto para regressão.

🌟 Ideia principal

Para prever a classe ou valor de um novo ponto, o KNN procura os K vizinhos mais próximos dele no conjunto de treinamento.

A decisão é baseada nesses vizinhos:

Classificação: escolhe a classe mais frequente entre os vizinhos.

Regressão: calcula a média (ou outra medida) dos valores dos  vizinhos.

🔑 Passos do algoritmo

Escolher o valor de K (número de vizinhos a considerar).

Calcular a distância entre o novo ponto e todos os pontos do conjunto de treino (normalmente distância Euclidiana).

Selecionar os K pontos mais próximos.

Fazer a previsão:

Classe mais votada → classificação.

Média/mediana → regressão.

✅ Vantagens

Simples e intuitivo.

Funciona bem em problemas com fronteiras de decisão complexas.

⚠️ Desvantagens

Custo alto em predição (precisa calcular distância para todos os pontos).

Sensível a atributos com escalas diferentes → normalização dos dados é essencial.

Escolher o valor de K pode ser difícil (muito pequeno → ruído; muito grande → perda de detalhes).

👉 Exemplo rápido:
Se K=3 e os três vizinhos mais próximos de um ponto novo forem [“Gato”, “Cachorro”, “Gato”], o KNN prevê “Gato” (maioria).