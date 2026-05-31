# Case Capim — Predição de Inadimplência em Crédito

Análise e modelagem de risco de crédito com base em dados de contratos e parcelas de clientes. O objetivo é identificar os clientes com maior probabilidade de inadimplência (atraso superior a 60 dias) utilizando Machine Learning.

---

## Estrutura do Projeto

```
caseCapim/
├── case_capim.ipynb          # Notebook principal com toda a análise
├── Base de Contratos.xlsx    # Dados cadastrais e contratuais (300 contratos)
└── Base de Parcelas.xlsx     # Histórico de parcelas (5.659 registros)
```

---

## Etapas do Processo

### 1. Carregamento e EDA Inicial

- Leitura das duas bases: `Base de Contratos.xlsx` (300 contratos únicos) e `Base de Parcelas.xlsx` (5.659 parcelas)
- Mapeamento de nulos: `RENDA_ESTIMADA` (36%), `SCORE_CREDITO` (8%), `PROFISSAO` (4,33%) e `DATA_PAGAMENTO` (58,58% — parcelas não pagas)
- Identificação de 24 estados únicos e 7 categorias de profissão

### 2. Tratamento de Datas

- Remoção do componente de hora das colunas de data (apenas `00:00:00`, sem informação)

### 3. Criação do Target — INADIMPLENTE_OVER60

- Cálculo de `DIAS_ATRASO` por parcela: diferença entre `DATA_PAGAMENTO` e `DATA_VENCIMENTO`
- Parcelas sem pagamento receberam a data de referência (`2026-04-01`, maior data da base) para representar inadimplência em aberto
- Agrupamento por contrato: `ATRASO_MAX` e `ATRASO_MEDIO`
- **Over30** (atraso > 30 dias): 45% dos contratos
- **Over60** (atraso > 60 dias): 34,3% dos contratos → **target escolhido** por melhor equilíbrio entre as classes e por excluir atrasos pontuais

### 4. Tratamento de Nulos

| Coluna                    | Tratamento                                      | Justificativa                                        |
| ------------------------- | ----------------------------------------------- | ---------------------------------------------------- |
| `PROFISSAO`               | Preenchido com `"unknown"`                      | Ausência de profissão pode ser indicador de risco    |
| `SCORE_CREDITO`           | Mediana da base (775,5)                         | Imputação conservadora e robusta a outliers          |
| `RENDA_ESTIMADA`          | Mediana por profissão                           | Preserva a heterogeneidade salarial entre categorias |
| `RENDA_ESTIMADA` negativa | Convertida para nulo, depois tratada como acima | Dado inconsistente                                   |

### 5. Feature Engineering

- **`REGIAO`**: agrupamento dos 24 estados em 5 regiões geográficas (Norte, Nordeste, Sudeste, Sul, Centro-Oeste)
- **`COMPROMETIMENTO_RENDA`**: razão entre o valor da 1ª parcela e a renda estimada — proxy de capacidade de pagamento (range: 2% a 71%)
- **One-hot encoding** de `REGIAO` e `PROFISSAO` (`drop_first=False`): 13 novas colunas binárias

### 6. Divisão Treino/Teste

- Split 70/30 estratificado por `INADIMPLENTE_OVER60` (`random_state=42`)
- Treino: 210 amostras | Teste: 90 amostras
- Proporção de inadimplentes preservada: ~34,3% em ambos os conjuntos

### 7. Modelagem

Dois modelos testados com 19 features:

| Modelo                      | Acurácia | Precisão | Recall | F1    | ROC-AUC |
| --------------------------- | -------- | -------- | ------ | ----- | ------- |
| Regressão Logística         | 58,9%    | 20,0%    | 6,5%   | 0,098 | 0,491   |
| Random Forest (max_depth=4) | 64,4%    | 33,3%    | 3,2%   | 0,059 | 0,616   |

O **Random Forest** foi selecionado por AUC superior. A Regressão Logística ficou abaixo de 0,5 de AUC, indicando fraca separação linear das classes.

### 8. Ajuste com class_weight='balanced' e Threshold

Para corrigir o desequilíbrio de classes, o RF foi retreinado com `class_weight='balanced'`. O threshold padrão (0,5) foi substituído por valores menores para aumentar o recall:

| Threshold | Acurácia | Precisão | Recall    | F1    | Inadimplentes detectados |
| --------- | -------- | -------- | --------- | ----- | ------------------------ |
| 0,35      | 40,0%    | 33,8%    | **77,4%** | 0,471 | 24 / 31                  |
| 0,40      | 52,2%    | 38,9%    | 67,7%     | 0,494 | 21 / 31                  |

**Threshold recomendado: 0,35** — maximiza a detecção de inadimplentes, alinhado à lógica de risco de crédito onde o custo de conceder crédito a um inadimplente é maior do que negar a um bom pagador.

### 9. Feature Importance

Top 7 variáveis explicam ~87% do poder preditivo do modelo:

| #   | Variável                | Importância |
| --- | ----------------------- | ----------- |
| 1   | `SCORE_CREDITO`         | 19,96%      |
| 2   | `RENDA_ESTIMADA`        | 19,46%      |
| 3   | `COMPROMETIMENTO_RENDA` | 11,55%      |
| 4   | `VALOR_EMPRESTADO`      | 10,74%      |
| 5   | `PROFISSAO_unknown`     | 8,87%       |
| 6   | `IDADE`                 | 8,26%       |
| 7   | `PRAZO_MESES`           | 7,98%       |

Limiares extraídos das árvores do modelo (mediana dos splits):

| Feature                 | Limiar de risco |
| ----------------------- | --------------- |
| `SCORE_CREDITO`         | < 801           |
| `RENDA_ESTIMADA`        | < R$ 2.013      |
| `COMPROMETIMENTO_RENDA` | > 13% da renda  |

---

## Conclusões

### 1. Quais clientes têm maior risco de não pagar?

O perfil de maior risco de inadimplência (over 60 dias) é composto pela combinação dos seguintes fatores:

- **Score de crédito abaixo de ~800**: É o principal divisor de águas de risco, respondendo por 20% do poder preditivo do algoritmo.
- **Renda estagnada (concentração majoritária com teto até R$ 2.300)**: Clientes sem evolução de renda, com menor capacidade de absorver choques financeiros.
- **Comprometimento de renda acima de 13%**: A partir desse limiar, a parcela da dívida passa a representar um peso crítico no fluxo de caixa mensal.
- **Profissão não informada (`unknown`)**: A ausência do dado cadastral se revelou o 5º fator mais relevante, indicando forte correlação entre perfis incompletos e o calote.
- **Empréstimos de maior valor com prazos mais longos**: A combinação de empréstimos de maior valor esticados em prazos mais longos.

> Em suma: Clientes que omitem informações cadastrais, sem histórico de crédito robusto, com renda limitada e que comprometem uma parcela significativa dela com a dívida, formam o principal bolsão de risco da carteira.

---

### 2. Como podemos melhorar nossa estratégia?

Podemos implementar esteiras de decisão (Fast Tracks) baseadas nos limiares exatos encontrados pelo modelo:

1. Aprovação Direta: O modelo identificou um volume massivo de segurança quando o Comprometimento de Renda é baixíssimo (entre 4% e 5%) aliado a um Score superior a 800 pontos. Esse perfil pode receber aprovação instantânea, maximizando a conversão do funil de vendas.

2. Zona de Precificação por Risco: Como a mediana de Comprometimento de Renda da base geral é de 14%, clientes que solicitem crédito comprometendo mais de 20% da renda e possuam Score abaixo de 800 caem na zona de risco elevado. Para este grupo, a estratégia pode ser bifásica:

Conservadora: Negação automática para controle imediato da inadimplência.

## Ousada (Crescimento): Aprovação condicionada a taxas de juros mais altas, prazos mais curtos ou exigência de garantias, permitindo rentabilizar um perfil de risco e expandir a carteira de forma controlada.

## Tecnologias Utilizadas

- Python 3.x
- pandas, numpy, matplotlib
- scikit-learn (LogisticRegression, RandomForestClassifier, StandardScaler, métricas)
