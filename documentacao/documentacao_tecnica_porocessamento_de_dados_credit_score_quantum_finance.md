# Documentação Técnica: Processamento de Dados para Previsão de Score de Crédito - Quantum Finance

**Autor:** Daniel Estrella Couto

**Projeto:** Quantum Finance - Credit Score Prediction

**Notebook:** processamento-dados.ipynb



---



## 1. Introdução

Este documento detalha as estratégias de pré-processamento de dados, tratamento de valores ausentes e inconsistências, feature engineering e análises exploratórias realizadas em um projeto de Machine Learning focado na previsão de score de crédito. O objetivo é fornecer uma visão abrangente e profissional das etapas executadas, justificando cada decisão técnica e demonstrando a robustez da abordagem.

## 2. Análise Preliminar e Carregamento dos Dados

Nesta seção, descrevemos o processo inicial de carregamento do dataset e a primeira inspeção dos dados, que é crucial para entender a estrutura, os tipos de variáveis e a presença de valores ausentes ou inconsistentes.

### 2.1. Carregamento do Dataset

O dataset foi carregado utilizando a biblioteca `pandas` do Python. Esta etapa é fundamental para iniciar qualquer projeto de análise de dados, garantindo que os dados estejam acessíveis e em um formato adequado para manipulação.

```python
import pandas as pd

# Carregamento do dataset
df = pd.read_csv('nome_do_arquivo.csv') # Substituir pelo nome real do arquivo
```

### 2.2. Visão Geral Inicial dos Dados

Após o carregamento, uma análise inicial foi realizada para obter uma compreensão básica da estrutura do DataFrame. Isso incluiu a verificação das primeiras linhas, o formato das colunas e a identificação de informações gerais sobre o dataset.

```python
df.head()
df.info()
df.describe()
```

### 2.3. Identificação de Tipos de Dados e Valores Ausentes

Uma etapa crítica no pré-processamento é a identificação dos tipos de dados de cada coluna e a detecção de valores ausentes. Valores ausentes podem impactar significativamente a performance de modelos de Machine Learning e, portanto, precisam ser tratados adequadamente. A análise foi feita para quantificar a presença de nulos e entender a natureza de cada coluna.

```python
df.isnull().sum()
df.dtypes
```

## 3. Tratamento de Valores Ausentes e Inconsistências

Esta seção aborda as estratégias detalhadas para lidar com valores ausentes e inconsistências em colunas específicas, garantindo a qualidade e a integridade dos dados para as fases subsequentes.

### 3.1. Coluna `mix_credito`: Análise de Data Leakage e Remoção

A coluna `mix_credito` foi identificada como tendo uma forte correlação com a variável target (`score_credito`). Uma análise aprofundada revelou que essa correlação era tão alta que poderia levar a um problema de *data leakage*. O *data leakage* ocorre quando informações do target são inadvertidamente incluídas nas features de treinamento, fazendo com que o modelo

aparentemente performe bem no treinamento, mas falhe miseravelmente em dados novos e não vistos. Para evitar que o modelo "trapaceasse" e para garantir que ele aprendesse padrões reais do comportamento do cliente, a decisão foi remover a coluna `mix_credito` do dataset antes do treinamento. Esta é uma prática crucial para a construção de modelos robustos e generalizáveis.

```python
# Calcular distribuição percentual de mix_credito (ignorando os nulos "_")
mix_credito_sem_nulo = df_categoricas[df_categoricas["mix_credito"] != "_"]["mix_credito"].value_counts(normalize=True) * 100

# Calcular distribuição percentual de score_credito
score_credito_pct = df_categoricas["score_credito"].value_counts(normalize=True) * 100

# Exibir os resultados
print("=== Distribuição percentual de mix_credito (sem nulos) ===")
print(mix_credito_sem_nulo.round(2))

print("\n=== Distribuição percentual de score_credito ===")
print(score_credito_pct.round(2))

# Remover a coluna mix_credito do dataframe categórico
df_categoricas = df_categoricas.drop(columns=["mix_credito"])

# Conferir se a coluna foi removida
print("Colunas restantes em df_categoricas:")
print(df_categoricas.columns.tolist())
```

### 3.2. Coluna `pagamento_valor_minimo`: Tratamento de Valores 'NM'

A coluna `pagamento_valor_minimo` apresentava valores `NM` (Not Mentioned/Não Mencionado), que representavam dados ausentes. Inicialmente, foi levantada a hipótese de que o `comportamento_pagamento` do cliente poderia ser utilizado para inferir esses valores. A lógica sugeria que clientes com `Small_value_payments` (pagamentos de baixo valor) poderiam ter maior probabilidade de pagar apenas o valor mínimo (`Yes`), enquanto outros com `Medium` ou `Large_value_payments` (pagamentos de valor médio ou alto) tenderiam a não pagar apenas o mínimo (`No`).

Para validar essa hipótese, foi construída uma tabela de contingência cruzando `pagamento_valor_minimo` e `comportamento_pagamento`. A análise revelou que a distribuição de `Yes` e `No` era relativamente equilibrada em todos os comportamentos, com `Yes` sendo majoritário, mas sem uma dominância absoluta (variando entre 44% e 57%). O percentual de `NM` também se manteve estável em todos os grupos, em torno de 11-12%, sem concentração em um comportamento específico. Essa evidência estatística indicou que não havia uma correlação forte o suficiente para justificar a imputação baseada no `comportamento_pagamento`.

Diante disso, a abordagem mais adequada foi manter `NM` como uma categoria própria, renomeando-a para `Not Informed` (Não Informado). Essa decisão preserva a informação de que o cliente não respondeu a essa questão, tratando os valores ausentes como uma categoria significativa em si, em vez de imputá-los de forma arbitrária e potencialmente enviesada.

```python
# Exibir os 30 primeiros valores únicos
df_categoricas.pagamento_valor_minimo.unique()[:30]

# Ver a distribuição de valores em pagamento_valor_minimo
distribuicao_pagamento_min = df_categoricas["pagamento_valor_minimo"].value_counts(normalize=True) * 100

print("=== Distribuição de valores em pagamento_valor_minimo (%) ===")
print(distribuicao_pagamento_min)

# Criar a tabela de contingência (crosstab) entre pagamento_valor_minimo e comportamento_pagamento
tabela_contingencia = pd.crosstab(
    df_categoricas["pagamento_valor_minimo"],
    df_categoricas["comportamento_pagamento"],
    normalize="columns"  # normaliza por coluna para ver % de Yes/No/NM dentro de cada comportamento
) * 100

# Exibir a tabela com porcentagens arredondadas
print("=== Distribuição (%) de Payment_of_Min_Amount por Payment_Behaviour ===")
print(tabela_contingencia.round(2))

# Substituir NM por "Not Informed" para padronização
df_categoricas["pagamento_valor_minimo"] = df_categoricas["pagamento_valor_minimo"].replace("NM", "Not Informed")

# Conferir resultado
print(df_categoricas["pagamento_valor_minimo"].value_counts())
```

### 3.3. Coluna `comportamento_pagamento`: Análise e Tratamento de Valores Inválidos

A coluna `comportamento_pagamento` apresentava um valor inválido `!@9#%8`. As categorias válidas eram coerentes, divididas em `High_spent` (alto gasto) e `Low_spent` (baixo gasto), cada uma subdividida por valor de pagamento (`Small`, `Medium`, `Large`). A presença do valor inválido sugeria um erro de input que precisava ser investigado.

Para entender o significado desses registros inválidos, foram realizadas duas análises estatísticas complementares utilizando a `renda_anual` como métrica comparativa. A primeira análise agrupou os registros em `High_spent`, `Low_spent` e `Invalid` (para os registros `!@9#%8`). A segunda análise refinou essa visão, agrupando os comportamentos em `Small_value_payments`, `Medium_value_payments`, `Large_value_payments` e `Invalid`.

Os resultados mostraram que os registros `Invalid` tinham mediana, média e desvio-padrão da `renda_anual` muito próximos aos dos grupos `High_spent` e `Low_spent`, e também aos grupos de `value_payments`. Isso indicou que os registros inválidos não se alinhavam a um padrão de comportamento de gasto ou valor de pagamento específico que justificasse a criação de uma nova categoria ou a imputação em uma categoria existente. Em vez disso, a distribuição de renda dos registros inválidos era similar à distribuição geral, sugerindo que o `!@9#%8` era, de fato, um erro de digitação ou um valor genérico que não representava um comportamento distinto.

Com base nessa análise, a decisão foi remover os registros com o valor `!@9#%8` da coluna `comportamento_pagamento`. A remoção foi preferida à imputação ou criação de uma nova categoria, pois o valor não trazia informação útil e sua manutenção poderia introduzir ruído no modelo. A quantidade de registros afetados era pequena o suficiente para não impactar significativamente o tamanho do dataset.

```python
# Exibir os 30 primeiros valores únicos
df_categoricas.comportamento_pagamento.unique()[:30]

# Ver a distribuição de valores em comportamento_pagamento
distribuicao_comportamento = df_categoricas["comportamento_pagamento"].value_counts(normalize=True) * 100

print("=== Distribuição de valores em comportamento_pagamento (%) ===")
print(distribuicao_comportamento)

# Criar uma cópia de trabalho
df_temp = df_processado_type_ok.copy()

# Criar uma coluna indicando apenas "High_spent" ou "Low_spent"
df_temp["grupo_spent"] = df_temp["comportamento_pagamento"].apply(
    lambda x: "High_spent" if "High_spent" in str(x) else ("Low_spent" if "Low_spent" in str(x) else "Invalid")
)

# Calcular estatísticas para High_spent e Low_spent
estatisticas_grupo = df_temp.groupby("grupo_spent")["renda_anual"].agg(
    mediana_renda="median",
    desvio_renda="std",
    media_renda="mean",
    total_registros="count"
)

print("=== Estatísticas de renda anual por grupo_spent ===")
print(estatisticas_grupo)

# Estatísticas só para os registros inválidos (!@9#%8)
estatisticas_invalidos = df_temp.loc[df_temp["comportamento_pagamento"] == "!@9#%8", "renda_anual"].agg(
    ["median", "std", "mean", "count"]
)

print("\n=== Estatísticas de renda anual para registros inválidos ===")
print(estatisticas_invalidos)

# Criar cópia de trabalho
df_temp = df_processado_type_ok.copy()

# Criar uma coluna 'grupo_valor' com base no padrão do texto
def classificar_pagamento(valor):
    if "Small_value_payments" in str(valor):
        return "Small_value_payments"
    elif "Medium_value_payments" in str(valor):
        return "Medium_value_payments"
    elif "Large_value_payments" in str(valor):
        return "Large_value_payments"
    else:
        return "Invalid"

df_temp["grupo_valor"] = df_temp["comportamento_pagamento"].apply(classificar_pagamento)

# Agrupar por grupo_valor e calcular estatísticas de renda anual
estatisticas_valor = df_temp.groupby("grupo_valor")["renda_anual"].agg(
    mediana_renda="median",
    desvio_renda="std",
    media_renda="mean",
    total_registros="count"
).sort_values(by="mediana_renda", ascending=False)

print("=== Estatísticas de renda anual por grupo de valor ===")
print(estatisticas_valor)

# Remover os registros com o valor inválido
df_categoricas = df_categoricas[df_categoricas["comportamento_pagamento"] != "!@9#%8"]

# Conferir se a remoção foi feita corretamente
print("\nValores únicos em comportamento_pagamento após remoção:")
print(df_categoricas["comportamento_pagamento"].unique())
```

### 3.4. Coluna `ocupacao`: Tratamento de Valores Ausentes

A coluna `ocupacao` apresentava valores `_______` que indicavam dados ausentes. Após a substituição desses valores por `NaN`, foi observado que aproximadamente 7% dos registros estavam ausentes. A análise da distribuição de renda anual para cada ocupação e para os valores ausentes (`_______`) revelou que a mediana, média e desvio-padrão da renda dos registros ausentes eram muito próximos aos das demais profissões. Isso sugeriu que a ausência de informação não estava correlacionada a uma profissão específica, mas sim a uma decisão do indivíduo de não informar, caracterizando um cenário de Missing Not At Random (MNAR).

Para preservar essa informação e evitar distorções na distribuição das profissões, a estratégia adotada foi criar uma nova categoria `Not Informed` para os valores ausentes. Essa abordagem é mais robusta do que a imputação arbitrária, pois mantém a integridade dos dados e permite que o modelo aprenda com a ausência de informação como uma característica em si.

```python
# Exibir os 30 primeiros valores únicos
df_categoricas.ocupacao.unique()[:30]

# 1. Substituir os valores '_______' por NaN
df_categoricas["ocupacao"] = df_categoricas["ocupacao"].replace("_______", np.nan)

# 2. Calcular a porcentagem de cada ocupação
porcentagem_ocupacao = df_categoricas["ocupacao"].value_counts(normalize=True) * 100

# 3. Calcular a porcentagem de valores NaN
porcentagem_nan = df_categoricas["ocupacao"].isna().mean() * 100

# Exibir os resultados
print("=== Porcentagem por ocupação ===")
print(porcentagem_ocupacao)

print("\nPorcentagem de valores NaN:", porcentagem_nan)

# Criar uma cópia
df_temp = df_processado_type_ok.copy()

# Agrupar por profissão e calcular estatísticas de renda anual
estatisticas_ocupacao = df_temp.groupby("ocupacao")["renda_anual"].agg(
    mediana_renda="median",
    desvio_renda="std",
    media_renda="mean",
    total_registros="count"
).sort_values(by="mediana_renda", ascending=False)

# Exibir resultados no output
print(estatisticas_ocupacao)

# Substituir os valores nulos (NaN) por "Not Informed"
df_categoricas["ocupacao"] = df_categoricas["ocupacao"].fillna("Not Informed")

# Conferir se a substituição foi feita corretamente
porcentagem_ocupacao = df_categoricas["ocupacao"].value_counts(normalize=True) * 100
print("=== Porcentagem por ocupação após substituição ===")
print(porcentagem_ocupacao)
```

### 3.5. Coluna `tipos_emprestimos`: Padronização de Categorias

A coluna `tipos_emprestimos` continha diversas categorias que representavam tipos de empréstimos. Foi identificada a necessidade de padronizar a categoria `Not Specified` para `Not Informed`, a fim de manter a consistência na nomenclatura de valores ausentes ou não informados em todo o dataset. Essa padronização facilita a interpretação e o processamento dos dados, além de evitar a criação de categorias redundantes durante a codificação.

```python
# Exibir os 30 primeiros valores únicos
df_categoricas.tipos_emprestimos.unique()[:30]

# Substituir 'Not Specified' por 'Not Informed' para padronização
df_categoricas["tipos_emprestimos"] = df_categoricas["tipos_emprestimos"].replace("Not Specified", "Not Informed")

# Conferir resultado
print("=== Distribuição de valores em tipos_emprestimos (%) ===")
print(df_categoricas["tipos_emprestimos"].value_counts(normalize=True) * 100)
```

## 4. Feature Engineering e Transformações

Esta seção detalha as técnicas de feature engineering e transformações aplicadas às variáveis numéricas do dataset. O objetivo é otimizar a representação dos dados para melhorar o desempenho dos modelos de Machine Learning, lidando com outliers, assimetrias e a criação de novas features quando aplicável.

### 4.1. Coluna `idade`: Tratamento de Outliers e Discretização

A coluna `idade` foi analisada para identificar e tratar outliers. A presença de valores extremos pode distorcer as análises estatísticas e o treinamento de modelos. Após a identificação, os outliers foram tratados utilizando a técnica de winsorização, onde valores abaixo do percentil 1 e acima do percentil 99 foram substituídos pelos valores desses percentis, respectivamente. Isso ajuda a mitigar o impacto de valores extremos sem remover os dados completamente.

Além do tratamento de outliers, a coluna `idade` foi discretizada em faixas etárias. A discretização transforma uma variável contínua em uma variável categórica ordinal, o que pode ser benéfico para alguns modelos que não lidam bem com a linearidade ou para capturar relações não lineares. As faixas etárias foram definidas de forma a criar grupos significativos para a análise de crédito.

```python
# Exibir estatísticas descritivas da coluna idade
print("=== Estatísticas descritivas da coluna idade ===")
print(df_numericas["idade"].describe())

# Identificar e tratar outliers (exemplo com winsorização)
Q1 = df_numericas["idade"].quantile(0.01)
Q3 = df_numericas["idade"].quantile(0.99)
df_numericas["idade"] = df_numericas["idade"].clip(lower=Q1, upper=Q3)

# Discretização da coluna idade em faixas etárias
bins = [18, 25, 35, 45, 55, 65, 100]
labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
df_numericas["faixa_etaria"] = pd.cut(df_numericas["idade"], bins=bins, labels=labels, right=False)

# Conferir a distribuição das novas faixas etárias
print("\n=== Distribuição de faixas etárias ===")
print(df_numericas["faixa_etaria"].value_counts(normalize=True) * 100)
```

### 4.2. Coluna `renda_anual`: Tratamento de Outliers e Transformação Logarítmica

A coluna `renda_anual` apresentava uma distribuição altamente assimétrica e a presença de outliers extremos, o que é comum em variáveis de renda. Para mitigar o impacto desses outliers e normalizar a distribuição, foi aplicada uma transformação logarítmica (log1p, que lida bem com valores zero ou próximos de zero). Antes da transformação, os outliers foram tratados com winsorização, substituindo valores abaixo do percentil 1 e acima do percentil 99 pelos respectivos limites. Essa abordagem ajuda a reduzir a influência de valores extremos e a tornar a distribuição mais próxima de uma normal, o que é benéfico para muitos algoritmos de Machine Learning.

```python
# Exibir estatísticas descritivas da coluna renda_anual
print("=== Estatísticas descritivas da coluna renda_anual ===")
print(df_numericas["renda_anual"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["renda_anual"].quantile(0.01)
Q3 = df_numericas["renda_anual"].quantile(0.99)
df_numericas["renda_anual"] = df_numericas["renda_anual"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["renda_anual_log"] = np.log1p(df_numericas["renda_anual"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna renda_anual_log ===")
print(df_numericas["renda_anual_log"].describe())
```

### 4.3. Coluna `salario_liquido_mensal`: Tratamento de Outliers e Transformação Logarítmica

Similar à `renda_anual`, a coluna `salario_liquido_mensal` também exibia assimetria e outliers. O tratamento seguiu a mesma lógica: winsorização nos percentis 1 e 99 para limitar os valores extremos, seguida pela aplicação da transformação logarítmica (log1p). Essa transformação é eficaz para reduzir a assimetria e estabilizar a variância, tornando a variável mais adequada para modelos que assumem distribuições mais simétricas.

```python
# Exibir estatísticas descritivas da coluna salario_liquido_mensal
print("=== Estatísticas descritivas da coluna salario_liquido_mensal ===")
print(df_numericas["salario_liquido_mensal"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["salario_liquido_mensal"].quantile(0.01)
Q3 = df_numericas["salario_liquido_mensal"].quantile(0.99)
df_numericas["salario_liquido_mensal"] = df_numericas["salario_liquido_mensal"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["salario_liquido_mensal_log"] = np.log1p(df_numericas["salario_liquido_mensal"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna salario_liquido_mensal_log ===")
print(df_numericas["salario_liquido_mensal_log"].describe())
```

### 4.4. Coluna `qtd_contas_bancarias`: Tratamento de Outliers

A coluna `qtd_contas_bancarias` foi analisada para a presença de outliers. Embora seja uma variável discreta, valores excessivamente altos podem indicar anomalias ou clientes com perfis muito específicos que podem distorcer o modelo. O tratamento de outliers foi realizado por winsorização nos percentis 1 e 99, garantindo que os valores extremos fossem limitados sem perder a informação da distribuição central.

```python
# Exibir estatísticas descritivas da coluna qtd_contas_bancarias
print("=== Estatísticas descritivas da coluna qtd_contas_bancarias ===")
print(df_numericas["qtd_contas_bancarias"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["qtd_contas_bancarias"].quantile(0.01)
Q3 = df_numericas["qtd_contas_bancarias"].quantile(0.99)
df_numericas["qtd_contas_bancarias"] = df_numericas["qtd_contas_bancarias"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna qtd_contas_bancarias após tratamento ===")
print(df_numericas["qtd_contas_bancarias"].describe())
```

### 4.5. Coluna `qtd_cartoes_credito`: Tratamento de Outliers

Similar à `qtd_contas_bancarias`, a coluna `qtd_cartoes_credito` também é uma variável discreta que pode conter outliers. O tratamento foi realizado com winsorização nos percentis 1 e 99 para limitar a influência de valores extremos, como um número excepcionalmente alto ou baixo de cartões de crédito, que poderiam ser ruído ou representar casos muito específicos que não generalizam bem.

```python
# Exibir estatísticas descritivas da coluna qtd_cartoes_credito
print("=== Estatísticas descritivas da coluna qtd_cartoes_credito ===")
print(df_numericas["qtd_cartoes_credito"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["qtd_cartoes_credito"].quantile(0.01)
Q3 = df_numericas["qtd_cartoes_credito"].quantile(0.99)
df_numericas["qtd_cartoes_credito"] = df_numericas["qtd_cartoes_credito"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna qtd_cartoes_credito após tratamento ===")
print(df_numericas["qtd_cartoes_credito"].describe())
```

### 4.6. Coluna `taxa_juros`: Tratamento de Outliers

A coluna `taxa_juros` é uma variável numérica contínua que pode apresentar outliers, especialmente taxas de juros muito altas ou muito baixas. O tratamento foi feito por winsorização nos percentis 1 e 99, a fim de suavizar a influência desses valores extremos e garantir que a distribuição da variável seja mais representativa para o treinamento do modelo.

```python
# Exibir estatísticas descritivas da coluna taxa_juros
print("=== Estatísticas descritivas da coluna taxa_juros ===")
print(df_numericas["taxa_juros"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["taxa_juros"].quantile(0.01)
Q3 = df_numericas["taxa_juros"].quantile(0.99)
df_numericas["taxa_juros"] = df_numericas["taxa_juros"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna taxa_juros após tratamento ===")
print(df_numericas["taxa_juros"].describe())
```

### 4.7. Coluna `qtd_emprestimos`: Tratamento de Outliers

A coluna `qtd_emprestimos` representa o número de empréstimos e, como outras variáveis de contagem, pode ter outliers. A winsorização nos percentis 1 e 99 foi aplicada para limitar a influência de clientes com um número excepcionalmente alto ou baixo de empréstimos, garantindo que o modelo não seja excessivamente influenciado por esses casos extremos.

```python
# Exibir estatísticas descritivas da coluna qtd_emprestimos
print("=== Estatísticas descritivas da coluna qtd_emprestimos ===")
print(df_numericas["qtd_emprestimos"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["qtd_emprestimos"].quantile(0.01)
Q3 = df_numericas["qtd_emprestimos"].quantile(0.99)
df_numericas["qtd_emprestimos"] = df_numericas["qtd_emprestimos"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna qtd_emprestimos após tratamento ===")
print(df_numericas["qtd_emprestimos"].describe())
```

### 4.8. Coluna `dias_atraso_pagamento`: Tratamento de Outliers

A coluna `dias_atraso_pagamento` indica o número de dias de atraso em pagamentos. Valores muito altos nesta coluna são outliers críticos, pois representam um comportamento de crédito de alto risco. A winsorização nos percentis 1 e 99 foi utilizada para limitar esses valores extremos, mantendo a informação de atraso, mas suavizando o impacto dos atrasos mais severos, que poderiam dominar o modelo.

```python
# Exibir estatísticas descritivas da coluna dias_atraso_pagamento
print("=== Estatísticas descritivas da coluna dias_atraso_pagamento ===")
print(df_numericas["dias_atraso_pagamento"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["dias_atraso_pagamento"].quantile(0.01)
Q3 = df_numericas["dias_atraso_pagamento"].quantile(0.99)
df_numericas["dias_atraso_pagamento"] = df_numericas["dias_atraso_pagamento"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna dias_atraso_pagamento após tratamento ===")
print(df_numericas["dias_atraso_pagamento"].describe())
```

### 4.9. Coluna `qtd_pagamentos_atrasados`: Tratamento de Outliers

Similar à coluna de dias de atraso, a `qtd_pagamentos_atrasados` também é uma métrica de risco. Outliers nesta coluna (um número muito alto de pagamentos atrasados) foram tratados com winsorização nos percentis 1 e 99. Essa técnica ajuda a manter a representatividade da variável sem permitir que poucos casos extremos distorçam o aprendizado do modelo.

```python
# Exibir estatísticas descritivas da coluna qtd_pagamentos_atrasados
print("=== Estatísticas descritivas da coluna qtd_pagamentos_atrasados ===")
print(df_numericas["qtd_pagamentos_atrasados"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["qtd_pagamentos_atrasados"].quantile(0.01)
Q3 = df_numericas["qtd_pagamentos_atrasados"].quantile(0.99)
df_numericas["qtd_pagamentos_atrasados"] = df_numericas["qtd_pagamentos_atrasados"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna qtd_pagamentos_atrasados após tratamento ===")
print(df_numericas["qtd_pagamentos_atrasados"].describe())
```

### 4.10. Coluna `variacao_limite_credito`: Tratamento de Outliers

A coluna `variacao_limite_credito` pode apresentar outliers em ambas as extremidades (variações muito positivas ou muito negativas). A winsorização nos percentis 1 e 99 foi aplicada para limitar esses valores extremos, garantindo que a variável contribua de forma mais estável para o modelo, sem ser excessivamente influenciada por mudanças atípicas no limite de crédito.

```python
# Exibir estatísticas descritivas da coluna variacao_limite_credito
print("=== Estatísticas descritivas da coluna variacao_limite_credito ===")
print(df_numericas["variacao_limite_credito"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["variacao_limite_credito"].quantile(0.01)
Q3 = df_numericas["variacao_limite_credito"].quantile(0.99)
df_numericas["variacao_limite_credito"] = df_numericas["variacao_limite_credito"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna variacao_limite_credito após tratamento ===")
print(df_numericas["variacao_limite_credito"].describe())
```

### 4.11. Coluna `qtd_consultas_credito`: Tratamento de Outliers

A coluna `qtd_consultas_credito` representa o número de consultas de crédito. Um número excessivamente alto de consultas pode indicar um comportamento de busca de crédito arriscado. Outliers nesta coluna foram tratados com winsorização nos percentis 1 e 99, para mitigar o impacto de valores extremos e garantir que a variável seja mais robusta para o modelo.

```python
# Exibir estatísticas descritivas da coluna qtd_consultas_credito
print("=== Estatísticas descritivas da coluna qtd_consultas_credito ===")
print(df_numericas["qtd_consultas_credito"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["qtd_consultas_credito"].quantile(0.01)
Q3 = df_numericas["qtd_consultas_credito"].quantile(0.99)
df_numericas["qtd_consultas_credito"] = df_numericas["qtd_consultas_credito"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna qtd_consultas_credito após tratamento ===")
print(df_numericas["qtd_consultas_credito"].describe())
```

### 4.12. Coluna `divida_pendente`: Tratamento de Outliers e Transformação Logarítmica

A coluna `divida_pendente` é uma variável financeira que frequentemente apresenta assimetria e outliers. O tratamento incluiu a winsorização nos percentis 1 e 99 para limitar os valores extremos, seguida pela aplicação da transformação logarítmica (log1p). Essa combinação ajuda a normalizar a distribuição da dívida pendente, tornando-a mais adequada para modelos que assumem distribuições mais simétricas e reduzindo a influência de dívidas excepcionalmente altas.

```python
# Exibir estatísticas descritivas da coluna divida_pendente
print("=== Estatísticas descritivas da coluna divida_pendente ===")
print(df_numericas["divida_pendente"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["divida_pendente"].quantile(0.01)
Q3 = df_numericas["divida_pendente"].quantile(0.99)
df_numericas["divida_pendente"] = df_numericas["divida_pendente"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["divida_pendente_log"] = np.log1p(df_numericas["divida_pendente"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna divida_pendente_log ===")
print(df_numericas["divida_pendente_log"].describe())
```

### 4.13. Coluna `percentual_utilizacao_credito`: Tratamento de Outliers

A coluna `percentual_utilizacao_credito` indica o quão próximo o cliente está do seu limite de crédito. Valores muito altos (próximos a 100%) ou muito baixos podem ser outliers. A winsorização nos percentis 1 e 99 foi aplicada para limitar esses valores extremos, garantindo que a variável seja mais representativa do comportamento geral de utilização de crédito e menos suscetível a casos atípicos.

```python
# Exibir estatísticas descritivas da coluna percentual_utilizacao_credito
print("=== Estatísticas descritivas da coluna percentual_utilizacao_credito ===")
print(df_numericas["percentual_utilizacao_credito"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["percentual_utilizacao_credito"].quantile(0.01)
Q3 = df_numericas["percentual_utilizacao_credito"].quantile(0.99)
df_numericas["percentual_utilizacao_credito"] = df_numericas["percentual_utilizacao_credito"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna percentual_utilizacao_credito após tratamento ===")
print(df_numericas["percentual_utilizacao_credito"].describe())
```

### 4.14. Coluna `total_emprestimos_mensal`: Tratamento de Outliers e Transformação Logarítmica

A coluna `total_emprestimos_mensal` (total de empréstimos mensais) é outra variável financeira que tende a ter uma distribuição assimétrica e outliers. O tratamento envolveu a winsorização nos percentis 1 e 99, seguida pela transformação logarítmica (log1p). Essa abordagem padroniza a variável, reduzindo a assimetria e a influência de valores extremos, o que é fundamental para a estabilidade do modelo.

```python
# Exibir estatísticas descritivas da coluna total_emprestimos_mensal
print("=== Estatísticas descritivas da coluna total_emprestimos_mensal ===")
print(df_numericas["total_emprestimos_mensal"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["total_emprestimos_mensal"].quantile(0.01)
Q3 = df_numericas["total_emprestimos_mensal"].quantile(0.99)
df_numericas["total_emprestimos_mensal"] = df_numericas["total_emprestimos_mensal"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["total_emprestimos_mensal_log"] = np.log1p(df_numericas["total_emprestimos_mensal"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna total_emprestimos_mensal_log ===")
print(df_numericas["total_emprestimos_mensal_log"].describe())
```

### 4.15. Coluna `valor_investido_mensal`: Tratamento de Outliers e Transformação Logarítmica

A coluna `valor_investido_mensal` (valor investido mensalmente) também é uma variável financeira que pode apresentar uma distribuição assimétrica e outliers. O tratamento foi realizado com winsorização nos percentis 1 e 99, seguida pela transformação logarítmica (log1p). Essa estratégia visa normalizar a distribuição e reduzir a influência de valores de investimento excepcionalmente altos, que poderiam enviesar o modelo.

```python
# Exibir estatísticas descritivas da coluna valor_investido_mensal
print("=== Estatísticas descritivas da coluna valor_investido_mensal ===")
print(df_numericas["valor_investido_mensal"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["valor_investido_mensal"].quantile(0.01)
Q3 = df_numericas["valor_investido_mensal"].quantile(0.99)
df_numericas["valor_investido_mensal"] = df_numericas["valor_investido_mensal"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["valor_investido_mensal_log"] = np.log1p(df_numericas["valor_investido_mensal"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna valor_investido_mensal_log ===")
print(df_numericas["valor_investido_mensal_log"].describe())
```

### 4.16. Coluna `saldo_mensal`: Tratamento de Outliers e Transformação Logarítmica

A coluna `saldo_mensal` (saldo mensal) é outra variável financeira que pode se beneficiar do tratamento de outliers e da transformação logarítmica. A winsorização nos percentis 1 e 99 foi aplicada para limitar os valores extremos, seguida pela transformação logarítmica (log1p). Essa abordagem ajuda a estabilizar a variância e a normalizar a distribuição, tornando a variável mais adequada para o treinamento do modelo.

```python
# Exibir estatísticas descritivas da coluna saldo_mensal
print("=== Estatísticas descritivas da coluna saldo_mensal ===")
print(df_numericas["saldo_mensal"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["saldo_mensal"].quantile(0.01)
Q3 = df_numericas["saldo_mensal"].quantile(0.99)
df_numericas["saldo_mensal"] = df_numericas["saldo_mensal"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["saldo_mensal_log"] = np.log1p(df_numericas["saldo_mensal"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna saldo_mensal_log ===")
print(df_numericas["saldo_mensal_log"].describe())
```

### 4.17. Coluna `tempo_historico_credito_meses`: Tratamento de Outliers

A coluna `tempo_historico_credito_meses` (tempo de histórico de crédito em meses) é uma variável importante para a avaliação de crédito. Outliers nesta coluna (tempos de histórico muito curtos ou muito longos) foram tratados com winsorização nos percentis 1 e 99. Essa técnica ajuda a garantir que a variável seja mais robusta e menos suscetível a valores extremos que poderiam distorcer a análise de crédito.

```python
# Exibir estatísticas descritivas da coluna tempo_historico_credito_meses
print("=== Estatísticas descritivas da coluna tempo_historico_credito_meses ===")
print(df_numericas["tempo_historico_credito_meses"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["tempo_historico_credito_meses"].quantile(0.01)
Q3 = df_numericas["tempo_historico_credito_meses"].quantile(0.99)
df_numericas["tempo_historico_credito_meses"] = df_numericas["tempo_historico_credito_meses"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna tempo_historico_credito_meses após tratamento ===")
print(df_numericas["tempo_historico_credito_meses"].describe())
```

## 5. Análise de Correlação e Multicolinearidade

Esta seção descreve a análise de correlação entre as variáveis numéricas e o target, bem como a investigação de multicolinearidade entre as features, utilizando o VIF (Variance Inflation Factor).

### 5.1. Correlação com o Target (Pearson e Spearman)

Foram calculadas as correlações de Pearson e Spearman entre as variáveis numéricas e o target (`score_credito_num`). A correlação de Pearson mede a relação linear, enquanto a de Spearman mede a relação monotônica, sendo mais robusta a outliers e distribuições não normais. A análise revelou que nenhuma variável apresentava uma correlação extremamente alta com o target, o que é um bom indicativo para evitar *data leakage* e garantir que o modelo precise de múltiplas features para fazer suas previsões. Os resultados foram ordenados pelo valor absoluto da correlação para identificar as variáveis mais influentes.

```python
# Calcula a correlação de Pearson
corr_pearson = df_numericas_e_maptarget.corr(method="pearson")

# Ordena as correlações do target pelo valor absoluto, mas preserva o sinal
corr_target_pearson = corr_pearson["score_credito_num"]
corr_target_pearson = corr_target_pearson.reindex(corr_target_pearson.abs().sort_values(ascending=False).index)

print("Correlação do target (score_credito_num) com as variáveis (Pearson - ordenada pelo valor absoluto):\n")
print(corr_target_pearson)

# Calcula a correlação de Spearman
corr_spearman = df_numericas_e_maptarget.corr(method="spearman")

# Ordena as correlações do target pelo valor absoluto, mas preserva o sinal
corr_target_spearman = corr_spearman["score_credito_num"]
corr_target_spearman = corr_target_spearman.reindex(corr_target_spearman.abs().sort_values(ascending=False).index)

print("\nCorrelação do target (score_credito_num) com as variáveis (Spearman - ordenada pelo valor absoluto):\n")
print(corr_target_spearman)
```

### 5.2. Análise de VIF (Variance Inflation Factor)

Para investigar a multicolinearidade entre as variáveis numéricas, foi calculado o VIF (Variance Inflation Factor). O VIF mede o quanto a variância de um coeficiente de regressão estimado é inflacionada devido à multicolinearidade. Valores de VIF acima de 5 ou 10 geralmente indicam problemas. A análise mostrou que todas as variáveis apresentavam valores de VIF entre 1.00 e 2.50, o que é considerado um nível saudável de correlação e muito abaixo dos limites problemáticos. Mesmo a alta correlação observada entre `renda_anual` e `salario_liquido_mensal` não resultou em um VIF elevado, indicando que ambas as variáveis poderiam ser mantidas sem causar problemas significativos de multicolinearidade no modelo. Diante disso, decidiu-se não remover nenhuma variável neste momento, deixando eventuais ajustes de seleção de variáveis para serem avaliados após os primeiros testes de modelagem.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

def calcular_vif(df):
    """
    Calcula o VIF (Variance Inflation Factor) manualmente para cada coluna do DataFrame.
    """
    vif_dict = {}
    X = df.values

    for i in range(df.shape[1]):
        y = X[:, i]  # variável alvo (uma coluna)
        X_outras = np.delete(X, i, axis=1)  # todas as outras variáveis

        modelo = LinearRegression()
        modelo.fit(X_outras, y)
        r2 = modelo.score(X_outras, y)

        vif = 1 / (1 - r2)
        vif_dict[df.columns[i]] = vif

    return pd.DataFrame({"Variável": list(vif_dict.keys()), "VIF": list(vif_dict.values())}).sort_values(by="VIF", ascending=False)

# Executa o cálculo de VIF nas variáveis numéricas
vif_df = calcular_vif(df_numericas)

print("📊 Fatores de Inflação de Variância (VIF) - Calculados sem statsmodels:\")
print(vif_df)
```

## 6. Codificação e Exportação Final

Esta seção finaliza o pré-processamento dos dados, abordando o mapeamento da variável target e a codificação das variáveis categóricas, culminando na exportação do dataset final pronto para a fase de modelagem.

### 6.1. Mapeamento do Target (`score_credito`)

Antes de aplicar o One-Hot Encoding nas variáveis categóricas, a variável target `score_credito` foi mapeada para valores numéricos ordinais: `Poor = 0`, `Standard = 1` e `Good = 2`. Essa etapa é crucial para preparar o target para algoritmos de Machine Learning, que geralmente exigem entradas numéricas. Ao mapear o target separadamente, evitamos que ele seja erroneamente codificado junto com as features categóricas, mantendo uma distinção clara entre features e target. O dataset final já terá o target no formato correto, facilitando a divisão em conjuntos de treino e teste na etapa de modelagem.

```python
# Faz uma cópia do df_categoricas para não alterar o original
df_categoricas_mapeado = df_categoricas.copy()

# Mapeia a coluna score_credito diretamente
mapa_target = {"Poor": 0, "Standard": 1, "Good": 2}
df_categoricas_mapeado["score_credito"] = df_categoricas_mapeado["score_credito"].map(mapa_target)

# Confere o resultado
df_categoricas_mapeado[["score_credito"]].head()
```

### 6.2. One-Hot Encoding das Variáveis Categóricas

As variáveis categóricas restantes (exceto o target já mapeado) foram codificadas utilizando a técnica de One-Hot Encoding. Esta técnica transforma cada categoria em uma nova coluna binária (0 ou 1), o que é essencial para que algoritmos de Machine Learning possam processar variáveis categóricas. A opção `drop_first=True` foi utilizada para evitar a armadilha da multicolinearidade, removendo uma das categorias de cada variável original. Isso garante que o modelo não seja prejudicado por variáveis linearmente dependentes.

```python
# Faz uma cópia do df_categoricas_mapeado
df_categoricas_get_dummies = df_categoricas_mapeado.copy()

# Aplica get_dummies em todas as colunas categóricas, exceto o target
df_categoricas_get_dummies = pd.get_dummies(
    df_categoricas_get_dummies.drop(columns=["score_credito"]), 
    drop_first=True
)

# Confere as primeiras linhas
df_categoricas_get_dummies.head()
```

### 6.3. Concatenação e Exportação do Dataset Final

Finalmente, as variáveis numéricas (já tratadas e transformadas) e as variáveis categóricas (já codificadas) foram concatenadas para formar o dataset final. A variável target (`score_credito`), já mapeada numericamente, foi adicionada como a última coluna. O DataFrame resultante, `df_final`, está agora completamente pré-processado e pronto para ser utilizado na fase de modelagem. Este dataset foi exportado para um arquivo CSV, garantindo que todas as transformações e tratamentos sejam persistidos e possam ser facilmente carregados para o treinamento de modelos.

```python
# Concatena numéricas e categóricas codificadas
df_final = pd.concat([df_numericas, df_categoricas_get_dummies], axis=1)

# Adiciona o target (já mapeado) como última coluna
df_final["score_credito"] = df_categoricas_mapeado["score_credito"]

# Conferir o DataFrame final
df_final.info()

# Exibir as primeiras linhas do DataFrame final
df_final.head()

# Exportar o DataFrame final para um arquivo CSV
df_final.to_csv("df_processado_final.csv", index=False)
```

## 7. Conclusão

Este documento detalhou exaustivamente as etapas de pré-processamento de dados, tratamento de valores ausentes e inconsistências, feature engineering e análise de multicolinearidade realizadas no projeto de previsão de score de crédito. Cada decisão foi justificada com base em análises estatísticas e melhores práticas de Machine Learning, visando a construção de um modelo robusto e confiável. O dataset final, `df_processado_final.csv`, está agora preparado para a fase de modelagem, onde diferentes algoritmos poderão ser aplicados e avaliados para prever o score de crédito com alta precisão e interpretabilidade. A documentação serve como um guia completo para futuras iterações e para garantir a reprodutibilidade do pipeline de dados.

### 4.12. Coluna `divida_pendente`: Tratamento de Outliers e Transformação Logarítmica

A coluna `divida_pendente` é uma variável financeira que frequentemente apresenta assimetria e outliers. O tratamento incluiu a winsorização nos percentis 1 e 99 para limitar os valores extremos, seguida pela aplicação da transformação logarítmica (log1p). Essa combinação ajuda a normalizar a distribuição da dívida pendente, tornando-a mais adequada para modelos que assumem distribuições mais simétricas e reduzindo a influência de dívidas excepcionalmente altas.

```python
# Exibir estatísticas descritivas da coluna divida_pendente
print("=== Estatísticas descritivas da coluna divida_pendente ===")
print(df_numericas["divida_pendente"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["divida_pendente"].quantile(0.01)
Q3 = df_numericas["divida_pendente"].quantile(0.99)
df_numericas["divida_pendente"] = df_numericas["divida_pendente"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["divida_pendente_log"] = np.log1p(df_numericas["divida_pendente"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna divida_pendente_log ===")
print(df_numericas["divida_pendente_log"].describe())
```

### 4.13. Coluna `percentual_utilizacao_credito`: Tratamento de Outliers

A coluna `percentual_utilizacao_credito` indica o quão próximo o cliente está do seu limite de crédito. Valores muito altos (próximos a 100%) ou muito baixos podem ser outliers. A winsorização nos percentis 1 e 99 foi aplicada para limitar esses valores extremos, garantindo que a variável seja mais representativa do comportamento geral de utilização de crédito e menos suscetível a casos atípicos.

```python
# Exibir estatísticas descritivas da coluna percentual_utilizacao_credito
print("=== Estatísticas descritivas da coluna percentual_utilizacao_credito ===")
print(df_numericas["percentual_utilizacao_credito"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["percentual_utilizacao_credito"].quantile(0.01)
Q3 = df_numericas["percentual_utilizacao_credito"].quantile(0.99)
df_numericas["percentual_utilizacao_credito"] = df_numericas["percentual_utilizacao_credito"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna percentual_utilizacao_credito após tratamento ===")
print(df_numericas["percentual_utilizacao_credito"].describe())
```

### 4.14. Coluna `total_emprestimos_mensal`: Tratamento de Outliers e Transformação Logarítmica

A coluna `total_emprestimos_mensal` (total de empréstimos mensais) é outra variável financeira que tende a ter uma distribuição assimétrica e outliers. O tratamento envolveu a winsorização nos percentis 1 e 99, seguida pela transformação logarítmica (log1p). Essa abordagem padroniza a variável, reduzindo a assimetria e a influência de valores extremos, o que é fundamental para a estabilidade do modelo.

```python
# Exibir estatísticas descritivas da coluna total_emprestimos_mensal
print("=== Estatísticas descritivas da coluna total_emprestimos_mensal ===")
print(df_numericas["total_emprestimos_mensal"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["total_emprestimos_mensal"].quantile(0.01)
Q3 = df_numericas["total_emprestimos_mensal"].quantile(0.99)
df_numericas["total_emprestimos_mensal"] = df_numericas["total_emprestimos_mensal"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["total_emprestimos_mensal_log"] = np.log1p(df_numericas["total_emprestimos_mensal"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna total_emprestimos_mensal_log ===")
print(df_numericas["total_emprestimos_mensal_log"].describe())
```

### 4.15. Coluna `valor_investido_mensal`: Tratamento de Outliers e Transformação Logarítmica

A coluna `valor_investido_mensal` (valor investido mensalmente) também é uma variável financeira que pode apresentar uma distribuição assimétrica e outliers. O tratamento foi realizado com winsorização nos percentis 1 e 99, seguida pela transformação logarítmica (log1p). Essa estratégia visa normalizar a distribuição e reduzir a influência de valores de investimento excepcionalmente altos, que poderiam enviesar o modelo.

```python
# Exibir estatísticas descritivas da coluna valor_investido_mensal
print("=== Estatísticas descritivas da coluna valor_investido_mensal ===")
print(df_numericas["valor_investido_mensal"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["valor_investido_mensal"].quantile(0.01)
Q3 = df_numericas["valor_investido_mensal"].quantile(0.99)
df_numericas["valor_investido_mensal"] = df_numericas["valor_investido_mensal"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["valor_investido_mensal_log"] = np.log1p(df_numericas["valor_investido_mensal"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna valor_investido_mensal_log ===")
print(df_numericas["valor_investido_mensal_log"].describe())
```

### 4.16. Coluna `saldo_mensal`: Tratamento de Outliers e Transformação Logarítmica

A coluna `saldo_mensal` (saldo mensal) é outra variável financeira que pode se beneficiar do tratamento de outliers e da transformação logarítmica. A winsorização nos percentis 1 e 99 foi aplicada para limitar os valores extremos, seguida pela transformação logarítmica (log1p). Essa abordagem ajuda a estabilizar a variância e a normalizar a distribuição, tornando a variável mais adequada para o treinamento do modelo.

```python
# Exibir estatísticas descritivas da coluna saldo_mensal
print("=== Estatísticas descritivas da coluna saldo_mensal ===")
print(df_numericas["saldo_mensal"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["saldo_mensal"].quantile(0.01)
Q3 = df_numericas["saldo_mensal"].quantile(0.99)
df_numericas["saldo_mensal"] = df_numericas["saldo_mensal"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["saldo_mensal_log"] = np.log1p(df_numericas["saldo_mensal"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna saldo_mensal_log ===")
print(df_numericas["saldo_mensal_log"].describe())
```

### 4.17. Coluna `tempo_historico_credito_meses`: Tratamento de Outliers

A coluna `tempo_historico_credito_meses` (tempo de histórico de crédito em meses) é uma variável importante para a avaliação de crédito. Outliers nesta coluna (tempos de histórico muito curtos ou muito longos) foram tratados com winsorização nos percentis 1 e 99. Essa técnica ajuda a garantir que a variável seja mais robusta e menos suscetível a valores extremos que poderiam distorcer a análise de crédito.

```python
# Exibir estatísticas descritivas da coluna tempo_historico_credito_meses
print("=== Estatísticas descritivas da coluna tempo_historico_credito_meses ===")
print(df_numericas["tempo_historico_credito_meses"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["tempo_historico_credito_meses"].quantile(0.01)
Q3 = df_numericas["tempo_historico_credito_meses"].quantile(0.99)
df_numericas["tempo_historico_credito_meses"] = df_numericas["tempo_historico_credito_meses"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna tempo_historico_credito_meses após tratamento ===")
print(df_numericas["tempo_historico_credito_meses"].describe())
```

## 5. Análise de Correlação e Multicolinearidade

Esta seção descreve a análise de correlação entre as variáveis numéricas e o target, bem como a investigação de multicolinearidade entre as features, utilizando o VIF (Variance Inflation Factor).

### 5.1. Correlação com o Target (Pearson e Spearman)

Foram calculadas as correlações de Pearson e Spearman entre as variáveis numéricas e o target (`score_credito_num`). A correlação de Pearson mede a relação linear, enquanto a de Spearman mede a relação monotônica, sendo mais robusta a outliers e distribuições não normais. A análise revelou que nenhuma variável apresentava uma correlação extremamente alta com o target, o que é um bom indicativo para evitar *data leakage* e garantir que o modelo precise de múltiplas features para fazer suas previsões. Os resultados foram ordenados pelo valor absoluto da correlação para identificar as variáveis mais influentes.

```python
# Calcula a correlação de Pearson
corr_pearson = df_numericas_e_maptarget.corr(method="pearson")

# Ordena as correlações do target pelo valor absoluto, mas preserva o sinal
corr_target_pearson = corr_pearson["score_credito_num"]
corr_target_pearson = corr_target_pearson.reindex(corr_target_pearson.abs().sort_values(ascending=False).index)

print("Correlação do target (score_credito_num) com as variáveis (Pearson - ordenada pelo valor absoluto):\n")
print(corr_target_pearson)

# Calcula a correlação de Spearman
corr_spearman = df_numericas_e_maptarget.corr(method="spearman")

# Ordena as correlações do target pelo valor absoluto, mas preserva o sinal
corr_target_spearman = corr_spearman["score_credito_num"]
corr_target_spearman = corr_target_spearman.reindex(corr_target_spearman.abs().sort_values(ascending=False).index)

print("\nCorrelação do target (score_credito_num) com as variáveis (Spearman - ordenada pelo valor absoluto):\n")
print(corr_target_spearman)
```

### 5.2. Análise de VIF (Variance Inflation Factor)

Para investigar a multicolinearidade entre as variáveis numéricas, foi calculado o VIF (Variance Inflation Factor). O VIF mede o quanto a variância de um coeficiente de regressão estimado é inflacionada devido à multicolinearidade. Valores de VIF acima de 5 ou 10 geralmente indicam problemas. A análise mostrou que todas as variáveis apresentavam valores de VIF entre 1.00 e 2.50, o que é considerado um nível saudável de correlação e muito abaixo dos limites problemáticos. Mesmo a alta correlação observada entre `renda_anual` e `salario_liquido_mensal` não resultou em um VIF elevado, indicando que ambas as variáveis poderiam ser mantidas sem causar problemas significativos de multicolinearidade no modelo. Diante disso, decidiu-se não remover nenhuma variável neste momento, deixando eventuais ajustes de seleção de variáveis para serem avaliados após os primeiros testes de modelagem.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

def calcular_vif(df):
    """
    Calcula o VIF (Variance Inflation Factor) manualmente para cada coluna do DataFrame.
    """
    vif_dict = {}
    X = df.values

    for i in range(df.shape[1]):
        y = X[:, i]  # variável alvo (uma coluna)
        X_outras = np.delete(X, i, axis=1)  # todas as outras variáveis

        modelo = LinearRegression()
        modelo.fit(X_outras, y)
        r2 = modelo.score(X_outras, y)

        vif = 1 / (1 - r2)
        vif_dict[df.columns[i]] = vif

    return pd.DataFrame({"Variável": list(vif_dict.keys()), "VIF": list(vif_dict.values())}).sort_values(by="VIF", ascending=False)

# Executa o cálculo de VIF nas variáveis numéricas
vif_df = calcular_vif(df_numericas)

print("📊 Fatores de Inflação de Variância (VIF) - Calculados sem statsmodels:\")
print(vif_df)
```

## 6. Codificação e Exportação Final

Esta seção finaliza o pré-processamento dos dados, abordando o mapeamento da variável target e a codificação das variáveis categóricas, culminando na exportação do dataset final pronto para a fase de modelagem.

### 6.1. Mapeamento do Target (`score_credito`)

Antes de aplicar o One-Hot Encoding nas variáveis categóricas, a variável target `score_credito` foi mapeada para valores numéricos ordinais: `Poor = 0`, `Standard = 1` e `Good = 2`. Essa etapa é crucial para preparar o target para algoritmos de Machine Learning, que geralmente exigem entradas numéricas. Ao mapear o target separadamente, evitamos que ele seja erroneamente codificado junto com as features categóricas, mantendo uma distinção clara entre features e target. O dataset final já terá o target no formato correto, facilitando a divisão em conjuntos de treino e teste na etapa de modelagem.

```python
# Faz uma cópia do df_categoricas para não alterar o original
df_categoricas_mapeado = df_categoricas.copy()

# Mapeia a coluna score_credito diretamente
mapa_target = {"Poor": 0, "Standard": 1, "Good": 2}
df_categoricas_mapeado["score_credito"] = df_categoricas_mapeado["score_credito"].map(mapa_target)

# Confere o resultado
df_categoricas_mapeado[["score_credito"]].head()
```

### 6.2. One-Hot Encoding das Variáveis Categóricas

As variáveis categóricas restantes (exceto o target já mapeado) foram codificadas utilizando a técnica de One-Hot Encoding. Esta técnica transforma cada categoria em uma nova coluna binária (0 ou 1), o que é essencial para que algoritmos de Machine Learning possam processar variáveis categóricas. A opção `drop_first=True` foi utilizada para evitar a armadilha da multicolinearidade, removendo uma das categorias de cada variável original. Isso garante que o modelo não seja prejudicado por variáveis linearmente dependentes.

```python
# Faz uma cópia do df_categoricas_mapeado
df_categoricas_get_dummies = df_categoricas_mapeado.copy()

# Aplica get_dummies em todas as colunas categóricas, exceto o target
df_categoricas_get_dummies = pd.get_dummies(
    df_categoricas_get_dummies.drop(columns=["score_credito"]), 
    drop_first=True
)

# Confere as primeiras linhas
df_categoricas_get_dummies.head()
```

### 6.3. Concatenação e Exportação do Dataset Final

Finalmente, as variáveis numéricas (já tratadas e transformadas) e as variáveis categóricas (já codificadas) foram concatenadas para formar o dataset final. A variável target (`score_credito`), já mapeada numericamente, foi adicionada como a última coluna. O DataFrame resultante, `df_final`, está agora completamente pré-processado e pronto para ser utilizado na fase de modelagem. Este dataset foi exportado para um arquivo CSV, garantindo que todas as transformações e tratamentos sejam persistidos e possam ser facilmente carregados para o treinamento de modelos.

```python
# Concatena numéricas e categóricas codificadas
df_final = pd.concat([df_numericas, df_categoricas_get_dummies], axis=1)

# Adiciona o target (já mapeado) como última coluna
df_final["score_credito"] = df_categoricas_mapeado["score_credito"]

# Conferir o DataFrame final
df_final.info()

# Exibir as primeiras linhas do DataFrame final
df_final.head()

# Exportar o DataFrame final para um arquivo CSV
df_final.to_csv("df_processado_final.csv", index=False)
```

## 7. Conclusão

Este documento detalhou exaustivamente as etapas de pré-processamento de dados, tratamento de valores ausentes e inconsistências, feature engineering e análise de multicolinearidade realizadas no projeto de previsão de score de crédito. Cada decisão foi justificada com base em análises estatísticas e melhores práticas de Machine Learning, visando a construção de um modelo robusto e confiável. O dataset final, `df_processado_final.csv`, está agora preparado para a fase de modelagem, onde diferentes algoritmos poderão ser aplicados e avaliados para prever o score de crédito com alta precisão e interpretabilidade. A documentação serve como um guia completo para futuras iterações e para garantir a reprodutibilidade do pipeline de dados.

## 5. Análise de Correlação e Multicolinearidade

Esta seção descreve a análise de correlação entre as variáveis numéricas e o target, bem como a investigação de multicolinearidade entre as features, utilizando o VIF (Variance Inflation Factor).

### 5.1. Correlação com o Target (Pearson e Spearman)

Foram calculadas as correlações de Pearson e Spearman entre as variáveis numéricas e o target (`score_credito_num`). A correlação de Pearson mede a relação linear, enquanto a de Spearman mede a relação monotônica, sendo mais robusta a outliers e distribuições não normais. A análise revelou que nenhuma variável apresentava uma correlação extremamente alta com o target, o que é um bom indicativo para evitar *data leakage* e garantir que o modelo precise de múltiplas features para fazer suas previsões. Os resultados foram ordenados pelo valor absoluto da correlação para identificar as variáveis mais influentes.

```python
# Calcula a correlação de Pearson
corr_pearson = df_numericas_e_maptarget.corr(method="pearson")

# Ordena as correlações do target pelo valor absoluto, mas preserva o sinal
corr_target_pearson = corr_pearson["score_credito_num"]
corr_target_pearson = corr_target_pearson.reindex(corr_target_pearson.abs().sort_values(ascending=False).index)

print("Correlação do target (score_credito_num) com as variáveis (Pearson - ordenada pelo valor absoluto):\n")
print(corr_target_pearson)

# Calcula a correlação de Spearman
corr_spearman = df_numericas_e_maptarget.corr(method="spearman")

# Ordena as correlações do target pelo valor absoluto, mas preserva o sinal
corr_target_spearman = corr_spearman["score_credito_num"]
corr_target_spearman = corr_target_spearman.reindex(corr_target_spearman.abs().sort_values(ascending=False).index)

print("\nCorrelação do target (score_credito_num) com as variáveis (Spearman - ordenada pelo valor absoluto):\n")
print(corr_target_spearman)
```

### 5.2. Análise de VIF (Variance Inflation Factor)

Para investigar a multicolinearidade entre as variáveis numéricas, foi calculado o VIF (Variance Inflation Factor). O VIF mede o quanto a variância de um coeficiente de regressão estimado é inflacionada devido à multicolinearidade. Valores de VIF acima de 5 ou 10 geralmente indicam problemas. A análise mostrou que todas as variáveis apresentavam valores de VIF entre 1.00 e 2.50, o que é considerado um nível saudável de correlação e muito abaixo dos limites problemáticos. Mesmo a alta correlação observada entre `renda_anual` e `salario_liquido_mensal` não resultou em um VIF elevado, indicando que ambas as variáveis poderiam ser mantidas sem causar problemas significativos de multicolinearidade no modelo. Diante disso, decidiu-se não remover nenhuma variável neste momento, deixando eventuais ajustes de seleção de variáveis para serem avaliados após os primeiros testes de modelagem.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

def calcular_vif(df):
    """
    Calcula o VIF (Variance Inflation Factor) manualmente para cada coluna do DataFrame.
    """
    vif_dict = {}
    X = df.values

    for i in range(df.shape[1]):
        y = X[:, i]  # variável alvo (uma coluna)
        X_outras = np.delete(X, i, axis=1)  # todas as outras variáveis

        modelo = LinearRegression()
        modelo.fit(X_outras, y)
        r2 = modelo.score(X_outras, y)

        vif = 1 / (1 - r2)
        vif_dict[df.columns[i]] = vif

    return pd.DataFrame({"Variável": list(vif_dict.keys()), "VIF": list(vif_dict.values())}).sort_values(by="VIF", ascending=False)

# Executa o cálculo de VIF nas variáveis numéricas
vif_df = calcular_vif(df_numericas)

print("📊 Fatores de Inflação de Variância (VIF) - Calculados sem statsmodels:\")
print(vif_df)
```

## 6. Codificação e Exportação Final

Esta seção finaliza o pré-processamento dos dados, abordando o mapeamento da variável target e a codificação das variáveis categóricas, culminando na exportação do dataset final pronto para a fase de modelagem.

### 6.1. Mapeamento do Target (`score_credito`)

Antes de aplicar o One-Hot Encoding nas variáveis categóricas, a variável target `score_credito` foi mapeada para valores numéricos ordinais: `Poor = 0`, `Standard = 1` e `Good = 2`. Essa etapa é crucial para preparar o target para algoritmos de Machine Learning, que geralmente exigem entradas numéricas. Ao mapear o target separadamente, evitamos que ele seja erroneamente codificado junto com as features categóricas, mantendo uma distinção clara entre features e target. O dataset final já terá o target no formato correto, facilitando a divisão em conjuntos de treino e teste na etapa de modelagem.

```python
# Faz uma cópia do df_categoricas para não alterar o original
df_categoricas_mapeado = df_categoricas.copy()

# Mapeia a coluna score_credito diretamente
mapa_target = {"Poor": 0, "Standard": 1, "Good": 2}
df_categoricas_mapeado["score_credito"] = df_categoricas_mapeado["score_credito"].map(mapa_target)

# Confere o resultado
df_categoricas_mapeado[["score_credito"]].head()
```

### 6.2. One-Hot Encoding das Variáveis Categóricas

As variáveis categóricas restantes (exceto o target já mapeado) foram codificadas utilizando a técnica de One-Hot Encoding. Esta técnica transforma cada categoria em uma nova coluna binária (0 ou 1), o que é essencial para que algoritmos de Machine Learning possam processar variáveis categóricas. A opção `drop_first=True` foi utilizada para evitar a armadilha da multicolinearidade, removendo uma das categorias de cada variável original. Isso garante que o modelo não seja prejudicado por variáveis linearmente dependentes.

```python
# Faz uma cópia do df_categoricas_mapeado
df_categoricas_get_dummies = df_categoricas_mapeado.copy()

# Aplica get_dummies em todas as colunas categóricas, exceto o target
df_categoricas_get_dummies = pd.get_dummies(
    df_categoricas_get_dummies.drop(columns=["score_credito"]), 
    drop_first=True
)

# Confere as primeiras linhas
df_categoricas_get_dummies.head()
```

### 6.3. Concatenação e Exportação do Dataset Final

Finalmente, as variáveis numéricas (já tratadas e transformadas) e as variáveis categóricas (já codificadas) foram concatenadas para formar o dataset final. A variável target (`score_credito`), já mapeada numericamente, foi adicionada como a última coluna. O DataFrame resultante, `df_final`, está agora completamente pré-processado e pronto para ser utilizado na fase de modelagem. Este dataset foi exportado para um arquivo CSV, garantindo que todas as transformações e tratamentos sejam persistidos e possam ser facilmente carregados para o treinamento de modelos.

```python
# Concatena numéricas e categóricas codificadas
df_final = pd.concat([df_numericas, df_categoricas_get_dummies], axis=1)

# Adiciona o target (já mapeado) como última coluna
df_final["score_credito"] = df_categoricas_mapeado["score_credito"]

# Conferir o DataFrame final
df_final.info()

# Exibir as primeiras linhas do DataFrame final
df_final.head()

# Exportar o DataFrame final para um arquivo CSV
df_final.to_csv("df_processado_final.csv", index=False)
```

## 7. Conclusão

Este documento detalhou exaustivamente as etapas de pré-processamento de dados, tratamento de valores ausentes e inconsistências, feature engineering e análise de multicolinearidade realizadas no projeto de previsão de score de crédito. Cada decisão foi justificada com base em análises estatísticas e melhores práticas de Machine Learning, visando a construção de um modelo robusto e confiável. O dataset final, `df_processado_final.csv`, está agora preparado para a fase de modelagem, onde diferentes algoritmos poderão ser aplicados e avaliados para prever o score de crédito com alta precisão e interpretabilidade. A documentação serve como um guia completo para futuras iterações e para garantir a reprodutibilidade do pipeline de dados.

## 5. Análise de Correlação e Multicolinearidade

Esta seção descreve a análise de correlação entre as variáveis numéricas e o target, bem como a investigação de multicolinearidade entre as features, utilizando o VIF (Variance Inflation Factor).

### 5.1. Correlação com o Target (Pearson e Spearman)

Foram calculadas as correlações de Pearson e Spearman entre as variáveis numéricas e o target (`score_credito_num`). A correlação de Pearson mede a relação linear, enquanto a de Spearman mede a relação monotônica, sendo mais robusta a outliers e distribuições não normais. A análise revelou que nenhuma variável apresentava uma correlação extremamente alta com o target, o que é um bom indicativo para evitar *data leakage* e garantir que o modelo precise de múltiplas features para fazer suas previsões. Os resultados foram ordenados pelo valor absoluto da correlação para identificar as variáveis mais influentes.

```python
# Calcula a correlação de Pearson
corr_pearson = df_numericas_e_maptarget.corr(method="pearson")

# Ordena as correlações do target pelo valor absoluto, mas preserva o sinal
corr_target_pearson = corr_pearson["score_credito_num"]
corr_target_pearson = corr_target_pearson.reindex(corr_target_pearson.abs().sort_values(ascending=False).index)

print("Correlação do target (score_credito_num) com as variáveis (Pearson - ordenada pelo valor absoluto):\n")
print(corr_target_pearson)

# Calcula a correlação de Spearman
corr_spearman = df_numericas_e_maptarget.corr(method="spearman")

# Ordena as correlações do target pelo valor absoluto, mas preserva o sinal
corr_target_spearman = corr_spearman["score_credito_num"]
corr_target_spearman = corr_target_spearman.reindex(corr_target_spearman.abs().sort_values(ascending=False).index)

print("\nCorrelação do target (score_credito_num) com as variáveis (Spearman - ordenada pelo valor absoluto):\n")
print(corr_target_spearman)
```

### 5.2. Análise de VIF (Variance Inflation Factor)

Para investigar a multicolinearidade entre as variáveis numéricas, foi calculado o VIF (Variance Inflation Factor). O VIF mede o quanto a variância de um coeficiente de regressão estimado é inflacionada devido à multicolinearidade. Valores de VIF acima de 5 ou 10 geralmente indicam problemas. A análise mostrou que todas as variáveis apresentavam valores de VIF entre 1.00 e 2.50, o que é considerado um nível saudável de correlação e muito abaixo dos limites problemáticos. Mesmo a alta correlação observada entre `renda_anual` e `salario_liquido_mensal` não resultou em um VIF elevado, indicando que ambas as variáveis poderiam ser mantidas sem causar problemas significativos de multicolinearidade no modelo. Diante disso, decidiu-se não remover nenhuma variável neste momento, deixando eventuais ajustes de seleção de variáveis para serem avaliados após os primeiros testes de modelagem.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

def calcular_vif(df):
    """
    Calcula o VIF (Variance Inflation Factor) manualmente para cada coluna do DataFrame.
    """
    vif_dict = {}
    X = df.values

    for i in range(df.shape[1]):
        y = X[:, i]  # variável alvo (uma coluna)
        X_outras = np.delete(X, i, axis=1)  # todas as outras variáveis

        modelo = LinearRegression()
        modelo.fit(X_outras, y)
        r2 = modelo.score(X_outras, y)

        vif = 1 / (1 - r2)
        vif_dict[df.columns[i]] = vif

    return pd.DataFrame({"Variável": list(vif_dict.keys()), "VIF": list(vif_dict.values())}).sort_values(by="VIF", ascending=False)

# Executa o cálculo de VIF nas variáveis numéricas
vif_df = calcular_vif(df_numericas)

print("📊 Fatores de Inflação de Variância (VIF) - Calculados sem statsmodels:\")
print(vif_df)
```

## 6. Codificação e Exportação Final

Esta seção finaliza o pré-processamento dos dados, abordando o mapeamento da variável target e a codificação das variáveis categóricas, culminando na exportação do dataset final pronto para a fase de modelagem.

### 6.1. Mapeamento do Target (`score_credito`)

Antes de aplicar o One-Hot Encoding nas variáveis categóricas, a variável target `score_credito` foi mapeada para valores numéricos ordinais: `Poor = 0`, `Standard = 1` e `Good = 2`. Essa etapa é crucial para preparar o target para algoritmos de Machine Learning, que geralmente exigem entradas numéricas. Ao mapear o target separadamente, evitamos que ele seja erroneamente codificado junto com as features categóricas, mantendo uma distinção clara entre features e target. O dataset final já terá o target no formato correto, facilitando a divisão em conjuntos de treino e teste na etapa de modelagem.

```python
# Faz uma cópia do df_categoricas para não alterar o original
df_categoricas_mapeado = df_categoricas.copy()

# Mapeia a coluna score_credito diretamente
mapa_target = {"Poor": 0, "Standard": 1, "Good": 2}
df_categoricas_mapeado["score_credito"] = df_categoricas_mapeado["score_credito"].map(mapa_target)

# Confere o resultado
df_categoricas_mapeado[["score_credito"]].head()
```

### 6.2. One-Hot Encoding das Variáveis Categóricas

As variáveis categóricas restantes (exceto o target já mapeado) foram codificadas utilizando a técnica de One-Hot Encoding. Esta técnica transforma cada categoria em uma nova coluna binária (0 ou 1), o que é essencial para que algoritmos de Machine Learning possam processar variáveis categóricas. A opção `drop_first=True` foi utilizada para evitar a armadilha da multicolinearidade, removendo uma das categorias de cada variável original. Isso garante que o modelo não seja prejudicado por variáveis linearmente dependentes.

```python
# Faz uma cópia do df_categoricas_mapeado
df_categoricas_get_dummies = df_categoricas_mapeado.copy()

# Aplica get_dummies em todas as colunas categóricas, exceto o target
df_categoricas_get_dummies = pd.get_dummies(
    df_categoricas_get_dummies.drop(columns=["score_credito"]), 
    drop_first=True
)

# Confere as primeiras linhas
df_categoricas_get_dummies.head()
```

### 6.3. Concatenação e Exportação do Dataset Final

Finalmente, as variáveis numéricas (já tratadas e transformadas) e as variáveis categóricas (já codificadas) foram concatenadas para formar o dataset final. A variável target (`score_credito`), já mapeada numericamente, foi adicionada como a última coluna. O DataFrame resultante, `df_final`, está agora completamente pré-processado e pronto para ser utilizado na fase de modelagem. Este dataset foi exportado para um arquivo CSV, garantindo que todas as transformações e tratamentos sejam persistidos e possam ser facilmente carregados para o treinamento de modelos.

```python
# Concatena numéricas e categóricas codificadas
df_final = pd.concat([df_numericas, df_categoricas_get_dummies], axis=1)

# Adiciona o target (já mapeado) como última coluna
df_final["score_credito"] = df_categoricas_mapeado["score_credito"]

# Conferir o DataFrame final
df_final.info()

# Exibir as primeiras linhas do DataFrame final
df_final.head()

# Exportar o DataFrame final para um arquivo CSV
df_final.to_csv("df_processado_final.csv", index=False)
```

## 7. Conclusão

Este documento detalhou exaustivamente as etapas de pré-processamento de dados, tratamento de valores ausentes e inconsistências, feature engineering e análise de multicolinearidade realizadas no projeto de previsão de score de crédito. Cada decisão foi justificada com base em análises estatísticas e melhores práticas de Machine Learning, visando a construção de um modelo robusto e confiável. O dataset final, `df_processado_final.csv`, está agora preparado para a fase de modelagem, onde diferentes algoritmos poderão ser aplicados e avaliados para prever o score de crédito com alta precisão e interpretabilidade. A documentação serve como um guia completo para futuras iterações e para garantir a reprodutibilidade do pipeline de dados.

## 6. Codificação e Exportação Final

Esta seção finaliza o pré-processamento dos dados, abordando o mapeamento da variável target e a codificação das variáveis categóricas, culminando na exportação do dataset final pronto para a fase de modelagem.

### 6.1. Mapeamento do Target (`score_credito`)

Antes de aplicar o One-Hot Encoding nas variáveis categóricas, a variável target `score_credito` foi mapeada para valores numéricos ordinais: `Poor = 0`, `Standard = 1` e `Good = 2`. Essa etapa é crucial para preparar o target para algoritmos de Machine Learning, que geralmente exigem entradas numéricas. Ao mapear o target separadamente, evitamos que ele seja erroneamente codificado junto com as features categóricas, mantendo uma distinção clara entre features e target. O dataset final já terá o target no formato correto, facilitando a divisão em conjuntos de treino e teste na etapa de modelagem.

```python
# Faz uma cópia do df_categoricas para não alterar o original
df_categoricas_mapeado = df_categoricas.copy()

# Mapeia a coluna score_credito diretamente
mapa_target = {"Poor": 0, "Standard": 1, "Good": 2}
df_categoricas_mapeado["score_credito"] = df_categoricas_mapeado["score_credito"].map(mapa_target)

# Confere o resultado
df_categoricas_mapeado[["score_credito"]].head()
```

### 6.2. One-Hot Encoding das Variáveis Categóricas

As variáveis categóricas restantes (exceto o target já mapeado) foram codificadas utilizando a técnica de One-Hot Encoding. Esta técnica transforma cada categoria em uma nova coluna binária (0 ou 1), o que é essencial para que algoritmos de Machine Learning possam processar variáveis categóricas. A opção `drop_first=True` foi utilizada para evitar a armadilha da multicolinearidade, removendo uma das categorias de cada variável original. Isso garante que o modelo não seja prejudicado por variáveis linearmente dependentes.

```python
# Faz uma cópia do df_categoricas_mapeado
df_categoricas_get_dummies = df_categoricas_mapeado.copy()

# Aplica get_dummies em todas as colunas categóricas, exceto o target
df_categoricas_get_dummies = pd.get_dummies(
    df_categoricas_get_dummies.drop(columns=["score_credito"]), 
    drop_first=True
)

# Confere as primeiras linhas
df_categoricas_get_dummies.head()
```

### 6.3. Concatenação e Exportação do Dataset Final

Finalmente, as variáveis numéricas (já tratadas e transformadas) e as variáveis categóricas (já codificadas) foram concatenadas para formar o dataset final. A variável target (`score_credito`), já mapeada numericamente, foi adicionada como a última coluna. O DataFrame resultante, `df_final`, está agora completamente pré-processado e pronto para ser utilizado na fase de modelagem. Este dataset foi exportado para um arquivo CSV, garantindo que todas as transformações e tratamentos sejam persistidos e possam ser facilmente carregados para o treinamento de modelos.

```python
# Concatena numéricas e categóricas codificadas
df_final = pd.concat([df_numericas, df_categoricas_get_dummies], axis=1)

# Adiciona o target (já mapeado) como última coluna
df_final["score_credito"] = df_categoricas_mapeado["score_credito"]

# Conferir o DataFrame final
df_final.info()

# Exibir as primeiras linhas do DataFrame final
df_final.head()

# Exportar o DataFrame final para um arquivo CSV
df_final.to_csv("df_processado_final.csv", index=False)
```

## 7. Conclusão

Este documento detalhou exaustivamente as etapas de pré-processamento de dados, tratamento de valores ausentes e inconsistências, feature engineering e análise de multicolinearidade realizadas no projeto de previsão de score de crédito. Cada decisão foi justificada com base em análises estatísticas e melhores práticas de Machine Learning, visando a construção de um modelo robusto e confiável. O dataset final, `df_processado_final.csv`, está agora preparado para a fase de modelagem, onde diferentes algoritmos poderão ser aplicados e avaliados para prever o score de crédito com alta precisão e interpretabilidade. A documentação serve como um guia completo para futuras iterações e para garantir a reprodutibilidade do pipeline de dados.

## 4. Feature Engineering e Transformações

Esta seção detalha as técnicas de feature engineering e transformações aplicadas às variáveis numéricas do dataset. O objetivo é otimizar a representação dos dados para melhorar o desempenho dos modelos de Machine Learning, lidando com outliers, assimetrias e a criação de novas features quando aplicável.

### 4.1. Coluna `idade`: Tratamento de Outliers e Discretização

A coluna `idade` foi analisada para identificar e tratar outliers. A presença de valores extremos pode distorcer as análises estatísticas e o treinamento de modelos. Após a identificação, os outliers foram tratados utilizando a técnica de winsorização, onde valores abaixo do percentil 1 e acima do percentil 99 foram substituídos pelos valores desses percentis, respectivamente. Isso ajuda a mitigar o impacto de valores extremos sem remover os dados completamente.

Além do tratamento de outliers, a coluna `idade` foi discretizada em faixas etárias. A discretização transforma uma variável contínua em uma variável categórica ordinal, o que pode ser benéfico para alguns modelos que não lidam bem com a linearidade ou para capturar relações não lineares. As faixas etárias foram definidas de forma a criar grupos significativos para a análise de crédito.

```python
# Exibir estatísticas descritivas da coluna idade
print("=== Estatísticas descritivas da coluna idade ===")
print(df_numericas["idade"].describe())

# Identificar e tratar outliers (exemplo com winsorização)
Q1 = df_numericas["idade"].quantile(0.01)
Q3 = df_numericas["idade"].quantile(0.99)
df_numericas["idade"] = df_numericas["idade"].clip(lower=Q1, upper=Q3)

# Discretização da coluna idade em faixas etárias
bins = [18, 25, 35, 45, 55, 65, 100]
labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
df_numericas["faixa_etaria"] = pd.cut(df_numericas["idade"], bins=bins, labels=labels, right=False)

# Conferir a distribuição das novas faixas etárias
print("\n=== Distribuição de faixas etárias ===")
print(df_numericas["faixa_etaria"].value_counts(normalize=True) * 100)
```

### 4.2. Coluna `renda_anual`: Tratamento de Outliers e Transformação Logarítmica

A coluna `renda_anual` apresentava uma distribuição altamente assimétrica e a presença de outliers extremos, o que é comum em variáveis de renda. Para mitigar o impacto desses outliers e normalizar a distribuição, foi aplicada uma transformação logarítmica (log1p, que lida bem com valores zero ou próximos de zero). Antes da transformação, os outliers foram tratados com winsorização, substituindo valores abaixo do percentil 1 e acima do percentil 99 pelos respectivos limites. Essa abordagem ajuda a reduzir a influência de valores extremos e a tornar a distribuição mais próxima de uma normal, o que é benéfico para muitos algoritmos de Machine Learning.

```python
# Exibir estatísticas descritivas da coluna renda_anual
print("=== Estatísticas descritivas da coluna renda_anual ===")
print(df_numericas["renda_anual"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["renda_anual"].quantile(0.01)
Q3 = df_numericas["renda_anual"].quantile(0.99)
df_numericas["renda_anual"] = df_numericas["renda_anual"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["renda_anual_log"] = np.log1p(df_numericas["renda_anual"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna renda_anual_log ===")
print(df_numericas["renda_anual_log"].describe())
```

### 4.3. Coluna `salario_liquido_mensal`: Tratamento de Outliers e Transformação Logarítmica

Similar à `renda_anual`, a coluna `salario_liquido_mensal` também exibia assimetria e outliers. O tratamento seguiu a mesma lógica: winsorização nos percentis 1 e 99 para limitar os valores extremos, seguida pela aplicação da transformação logarítmica (log1p). Essa transformação é eficaz para reduzir a assimetria e estabilizar a variância, tornando a variável mais adequada para modelos que assumem distribuições mais simétricas.

```python
# Exibir estatísticas descritivas da coluna salario_liquido_mensal
print("=== Estatísticas descritivas da coluna salario_liquido_mensal ===")
print(df_numericas["salario_liquido_mensal"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["salario_liquido_mensal"].quantile(0.01)
Q3 = df_numericas["salario_liquido_mensal"].quantile(0.99)
df_numericas["salario_liquido_mensal"] = df_numericas["salario_liquido_mensal"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["salario_liquido_mensal_log"] = np.log1p(df_numericas["salario_liquido_mensal"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna salario_liquido_mensal_log ===")
print(df_numericas["salario_liquido_mensal_log"].describe())
```

### 4.4. Coluna `qtd_contas_bancarias`: Tratamento de Outliers

A coluna `qtd_contas_bancarias` foi analisada para a presença de outliers. Embora seja uma variável discreta, valores excessivamente altos podem indicar anomalias ou clientes com perfis muito específicos que podem distorcer o modelo. O tratamento de outliers foi realizado por winsorização nos percentis 1 e 99, garantindo que os valores extremos fossem limitados sem perder a informação da distribuição central.

```python
# Exibir estatísticas descritivas da coluna qtd_contas_bancarias
print("=== Estatísticas descritivas da coluna qtd_contas_bancarias ===")
print(df_numericas["qtd_contas_bancarias"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["qtd_contas_bancarias"].quantile(0.01)
Q3 = df_numericas["qtd_contas_bancarias"].quantile(0.99)
df_numericas["qtd_contas_bancarias"] = df_numericas["qtd_contas_bancarias"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna qtd_contas_bancarias após tratamento ===")
print(df_numericas["qtd_contas_bancarias"].describe())
```

### 4.5. Coluna `qtd_cartoes_credito`: Tratamento de Outliers

Similar à `qtd_contas_bancarias`, a coluna `qtd_cartoes_credito` também é uma variável discreta que pode conter outliers. O tratamento foi realizado com winsorização nos percentis 1 e 99 para limitar a influência de valores extremos, como um número excepcionalmente alto ou baixo de cartões de crédito, que poderiam ser ruído ou representar casos muito específicos que não generalizam bem.

```python
# Exibir estatísticas descritivas da coluna qtd_cartoes_credito
print("=== Estatísticas descritivas da coluna qtd_cartoes_credito ===")
print(df_numericas["qtd_cartoes_credito"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["qtd_cartoes_credito"].quantile(0.01)
Q3 = df_numericas["qtd_cartoes_credito"].quantile(0.99)
df_numericas["qtd_cartoes_credito"] = df_numericas["qtd_cartoes_credito"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna qtd_cartoes_credito após tratamento ===")
print(df_numericas["qtd_cartoes_credito"].describe())
```

### 4.6. Coluna `taxa_juros`: Tratamento de Outliers

A coluna `taxa_juros` é uma variável numérica contínua que pode apresentar outliers, especialmente taxas de juros muito altas ou muito baixas. O tratamento foi feito por winsorização nos percentis 1 e 99, a fim de suavizar a influência desses valores extremos e garantir que a distribuição da variável seja mais representativa para o treinamento do modelo.

```python
# Exibir estatísticas descritivas da coluna taxa_juros
print("=== Estatísticas descritivas da coluna taxa_juros ===")
print(df_numericas["taxa_juros"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["taxa_juros"].quantile(0.01)
Q3 = df_numericas["taxa_juros"].quantile(0.99)
df_numericas["taxa_juros"] = df_numericas["taxa_juros"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna taxa_juros após tratamento ===")
print(df_numericas["taxa_juros"].describe())
```

### 4.7. Coluna `qtd_emprestimos`: Tratamento de Outliers

A coluna `qtd_emprestimos` representa o número de empréstimos e, como outras variáveis de contagem, pode ter outliers. A winsorização nos percentis 1 e 99 foi aplicada para limitar a influência de clientes com um número excepcionalmente alto ou baixo de empréstimos, garantindo que o modelo não seja excessivamente influenciado por esses casos extremos.

```python
# Exibir estatísticas descritivas da coluna qtd_emprestimos
print("=== Estatísticas descritivas da coluna qtd_emprestimos ===")
print(df_numericas["qtd_emprestimos"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["qtd_emprestimos"].quantile(0.01)
Q3 = df_numericas["qtd_emprestimos"].quantile(0.99)
df_numericas["qtd_emprestimos"] = df_numericas["qtd_emprestimos"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna qtd_emprestimos após tratamento ===")
print(df_numericas["qtd_emprestimos"].describe())
```

### 4.8. Coluna `dias_atraso_pagamento`: Tratamento de Outliers

A coluna `dias_atraso_pagamento` indica o número de dias de atraso em pagamentos. Valores muito altos nesta coluna são outliers críticos, pois representam um comportamento de crédito de alto risco. A winsorização nos percentis 1 e 99 foi utilizada para limitar esses valores extremos, mantendo a informação de atraso, mas suavizando o impacto dos atrasos mais severos, que poderiam dominar o modelo.

```python
# Exibir estatísticas descritivas da coluna dias_atraso_pagamento
print("=== Estatísticas descritivas da coluna dias_atraso_pagamento ===")
print(df_numericas["dias_atraso_pagamento"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["dias_atraso_pagamento"].quantile(0.01)
Q3 = df_numericas["dias_atraso_pagamento"].quantile(0.99)
df_numericas["dias_atraso_pagamento"] = df_numericas["dias_atraso_pagamento"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna dias_atraso_pagamento após tratamento ===")
print(df_numericas["dias_atraso_pagamento"].describe())
```

### 4.9. Coluna `qtd_pagamentos_atrasados`: Tratamento de Outliers

Similar à coluna de dias de atraso, a `qtd_pagamentos_atrasados` também é uma métrica de risco. Outliers nesta coluna (um número muito alto de pagamentos atrasados) foram tratados com winsorização nos percentis 1 e 99. Essa técnica ajuda a manter a representatividade da variável sem permitir que poucos casos extremos distorçam o aprendizado do modelo.

```python
# Exibir estatísticas descritivas da coluna qtd_pagamentos_atrasados
print("=== Estatísticas descritivas da coluna qtd_pagamentos_atrasados ===")
print(df_numericas["qtd_pagamentos_atrasados"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["qtd_pagamentos_atrasados"].quantile(0.01)
Q3 = df_numericas["qtd_pagamentos_atrasados"].quantile(0.99)
df_numericas["qtd_pagamentos_atrasados"] = df_numericas["qtd_pagamentos_atrasados"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna qtd_pagamentos_atrasados após tratamento ===")
print(df_numericas["qtd_pagamentos_atrasados"].describe())
```

### 4.10. Coluna `variacao_limite_credito`: Tratamento de Outliers

A coluna `variacao_limite_credito` pode apresentar outliers em ambas as extremidades (variações muito positivas ou muito negativas). A winsorização nos percentis 1 e 99 foi aplicada para limitar esses valores extremos, garantindo que a variável contribua de forma mais estável para o modelo, sem ser excessivamente influenciada por mudanças atípicas no limite de crédito.

```python
# Exibir estatísticas descritivas da coluna variacao_limite_credito
print("=== Estatísticas descritivas da coluna variacao_limite_credito ===")
print(df_numericas["variacao_limite_credito"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["variacao_limite_credito"].quantile(0.01)
Q3 = df_numericas["variacao_limite_credito"].quantile(0.99)
df_numericas["variacao_limite_credito"] = df_numericas["variacao_limite_credito"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna variacao_limite_credito após tratamento ===")
print(df_numericas["variacao_limite_credito"].describe())
```

### 4.11. Coluna `qtd_consultas_credito`: Tratamento de Outliers

A coluna `qtd_consultas_credito` representa o número de consultas de crédito. Um número excessivamente alto de consultas pode indicar um comportamento de busca de crédito arriscado. Outliers nesta coluna foram tratados com winsorização nos percentis 1 e 99, para mitigar o impacto de valores extremos e garantir que a variável seja mais robusta para o modelo.

```python
# Exibir estatísticas descritivas da coluna qtd_consultas_credito
print("=== Estatísticas descritivas da coluna qtd_consultas_credito ===")
print(df_numericas["qtd_consultas_credito"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["qtd_consultas_credito"].quantile(0.01)
Q3 = df_numericas["qtd_consultas_credito"].quantile(0.99)
df_numericas["qtd_consultas_credito"] = df_numericas["qtd_consultas_credito"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna qtd_consultas_credito após tratamento ===")
print(df_numericas["qtd_consultas_credito"].describe())
```

### 4.12. Coluna `divida_pendente`: Tratamento de Outliers e Transformação Logarítmica

A coluna `divida_pendente` é uma variável financeira que frequentemente apresenta assimetria e outliers. O tratamento incluiu a winsorização nos percentis 1 e 99 para limitar os valores extremos, seguida pela aplicação da transformação logarítmica (log1p). Essa combinação ajuda a normalizar a distribuição da dívida pendente, tornando-a mais adequada para modelos que assumem distribuições mais simétricas e reduzindo a influência de dívidas excepcionalmente altas.

```python
# Exibir estatísticas descritivas da coluna divida_pendente
print("=== Estatísticas descritivas da coluna divida_pendente ===")
print(df_numericas["divida_pendente"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["divida_pendente"].quantile(0.01)
Q3 = df_numericas["divida_pendente"].quantile(0.99)
df_numericas["divida_pendente"] = df_numericas["divida_pendente"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["divida_pendente_log"] = np.log1p(df_numericas["divida_pendente"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna divida_pendente_log ===")
print(df_numericas["divida_pendente_log"].describe())
```

### 4.13. Coluna `percentual_utilizacao_credito`: Tratamento de Outliers

A coluna `percentual_utilizacao_credito` indica o quão próximo o cliente está do seu limite de crédito. Valores muito altos (próximos a 100%) ou muito baixos podem ser outliers. A winsorização nos percentis 1 e 99 foi aplicada para limitar esses valores extremos, garantindo que a variável seja mais representativa do comportamento geral de utilização de crédito e menos suscetível a casos atípicos.

```python
# Exibir estatísticas descritivas da coluna percentual_utilizacao_credito
print("=== Estatísticas descritivas da coluna percentual_utilizacao_credito ===")
print(df_numericas["percentual_utilizacao_credito"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["percentual_utilizacao_credito"].quantile(0.01)
Q3 = df_numericas["percentual_utilizacao_credito"].quantile(0.99)
df_numericas["percentual_utilizacao_credito"] = df_numericas["percentual_utilizacao_credito"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna percentual_utilizacao_credito após tratamento ===")
print(df_numericas["percentual_utilizacao_credito"].describe())
```

### 4.14. Coluna `total_emprestimos_mensal`: Tratamento de Outliers e Transformação Logarítmica

A coluna `total_emprestimos_mensal` (total de empréstimos mensais) é outra variável financeira que tende a ter uma distribuição assimétrica e outliers. O tratamento envolveu a winsorização nos percentis 1 e 99, seguida pela transformação logarítmica (log1p). Essa abordagem padroniza a variável, reduzindo a assimetria e a influência de valores extremos, o que é fundamental para a estabilidade do modelo.

```python
# Exibir estatísticas descritivas da coluna total_emprestimos_mensal
print("=== Estatísticas descritivas da coluna total_emprestimos_mensal ===")
print(df_numericas["total_emprestimos_mensal"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["total_emprestimos_mensal"].quantile(0.01)
Q3 = df_numericas["total_emprestimos_mensal"].quantile(0.99)
df_numericas["total_emprestimos_mensal"] = df_numericas["total_emprestimos_mensal"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["total_emprestimos_mensal_log"] = np.log1p(df_numericas["total_emprestimos_mensal"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna total_emprestimos_mensal_log ===")
print(df_numericas["total_emprestimos_mensal_log"].describe())
```

### 4.15. Coluna `valor_investido_mensal`: Tratamento de Outliers e Transformação Logarítmica

A coluna `valor_investido_mensal` (valor investido mensalmente) também é uma variável financeira que pode apresentar uma distribuição assimétrica e outliers. O tratamento foi realizado com winsorização nos percentis 1 e 99, seguida pela transformação logarítmica (log1p). Essa estratégia visa normalizar a distribuição e reduzir a influência de valores de investimento excepcionalmente altos, que poderiam enviesar o modelo.

```python
# Exibir estatísticas descritivas da coluna valor_investido_mensal
print("=== Estatísticas descritivas da coluna valor_investido_mensal ===")
print(df_numericas["valor_investido_mensal"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["valor_investido_mensal"].quantile(0.01)
Q3 = df_numericas["valor_investido_mensal"].quantile(0.99)
df_numericas["valor_investido_mensal"] = df_numericas["valor_investido_mensal"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["valor_investido_mensal_log"] = np.log1p(df_numericas["valor_investido_mensal"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna valor_investido_mensal_log ===")
print(df_numericas["valor_investido_mensal_log"].describe())
```

### 4.16. Coluna `saldo_mensal`: Tratamento de Outliers e Transformação Logarítmica

A coluna `saldo_mensal` (saldo mensal) é outra variável financeira que pode se beneficiar do tratamento de outliers e da transformação logarítmica. A winsorização nos percentis 1 e 99 foi aplicada para limitar os valores extremos, seguida pela transformação logarítmica (log1p). Essa abordagem ajuda a estabilizar a variância e a normalizar a distribuição, tornando a variável mais adequada para o treinamento do modelo.

```python
# Exibir estatísticas descritivas da coluna saldo_mensal
print("=== Estatísticas descritivas da coluna saldo_mensal ===")
print(df_numericas["saldo_mensal"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["saldo_mensal"].quantile(0.01)
Q3 = df_numericas["saldo_mensal"].quantile(0.99)
df_numericas["saldo_mensal"] = df_numericas["saldo_mensal"].clip(lower=Q1, upper=Q3)

# Aplicar transformação logarítmica (log1p para lidar com valores zero)
df_numericas["saldo_mensal_log"] = np.log1p(df_numericas["saldo_mensal"])

# Exibir estatísticas descritivas da coluna transformada
print("\n=== Estatísticas descritivas da coluna saldo_mensal_log ===")
print(df_numericas["saldo_mensal_log"].describe())
```

### 4.17. Coluna `tempo_historico_credito_meses`: Tratamento de Outliers

A coluna `tempo_historico_credito_meses` (tempo de histórico de crédito em meses) é uma variável importante para a avaliação de crédito. Outliers nesta coluna (tempos de histórico muito curtos ou muito longos) foram tratados com winsorização nos percentis 1 e 99. Essa técnica ajuda a garantir que a variável seja mais robusta e menos suscetível a valores extremos que poderiam distorcer a análise de crédito.

```python
# Exibir estatísticas descritivas da coluna tempo_historico_credito_meses
print("=== Estatísticas descritivas da coluna tempo_historico_credito_meses ===")
print(df_numericas["tempo_historico_credito_meses"].describe())

# Identificar e tratar outliers (winsorização)
Q1 = df_numericas["tempo_historico_credito_meses"].quantile(0.01)
Q3 = df_numericas["tempo_historico_credito_meses"].quantile(0.99)
df_numericas["tempo_historico_credito_meses"] = df_numericas["tempo_historico_credito_meses"].clip(lower=Q1, upper=Q3)

# Conferir estatísticas após tratamento
print("\n=== Estatísticas descritivas da coluna tempo_historico_credito_meses após tratamento ===")
print(df_numericas["tempo_historico_credito_meses"].describe())
```

## 7. Conclusão

Este documento detalhou exaustivamente as etapas de pré-processamento de dados, tratamento de valores ausentes e inconsistências, feature engineering e análise de multicolinearidade realizadas no projeto de previsão de score de crédito. Cada decisão foi justificada com base em análises estatísticas e melhores práticas de Machine Learning, visando a construção de um modelo robusto e confiável. O dataset final, `df_processado_final.csv`, está agora preparado para a fase de modelagem, onde diferentes algoritmos poderão ser aplicados e avaliados para prever o score de crédito com alta precisão e interpretabilidade. A documentação serve como um guia completo para futuras iterações e para garantir a reprodutibilidade do pipeline de dados.

