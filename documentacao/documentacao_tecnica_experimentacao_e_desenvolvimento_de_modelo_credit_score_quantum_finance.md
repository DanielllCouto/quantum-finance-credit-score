# Documentação Técnica : Experimentação e Desenvolvimento de Modelo para Previsão de Score de Crédito - Quantum Finance

**Autor:** Daniel Estrella Couto  

**Projeto:** Quantum Finance - Credit Score Prediction  

**Notebook:** desenvolvimento-modelo.ipynb

---

## 1. Introdução e Contexto do Projeto

Esta documentação apresenta uma análise técnica abrangente e célula por célula do processo de experimentação e desenvolvimento de modelos de machine learning para previsão de score de crédito. O projeto foi estruturado em **duas épocas distintas de experimentação**, cada uma com objetivos específicos e estratégias progressivamente mais sofisticadas.

A documentação detalha o caminho completo da experimentação, desde os primeiros modelos individuais até o desenvolvimento de ensembles avançados com técnicas de regularização, culminando na identificação do **modelo campeão final**. O foco principal é a maximização do **recall da classe 0 (Poor)**, que representa clientes com alto risco de inadimplência - a métrica de negócio mais crítica para o projeto.

### 1.1. Objetivos de Negócio

O principal desafio de negócio é identificar corretamente clientes com score de crédito "Poor", minimizando falsos negativos que poderiam resultar em perdas financeiras significativas. Um cliente classificado incorretamente como "Good" ou "Standard" quando na verdade é "Poor" representa um risco muito maior do que o erro inverso.

### 1.2. Estrutura da Experimentação

O processo foi dividido em duas épocas principais:

- **Primeira Época:** Modelos individuais + Primeiro Ensemble (Stacking LightGBM + CatBoost)

- **Segunda Época:** Refinamento e otimização do ensemble com foco em redução de overfitting

### 1.3. Decisões-Chave e Métricas Registradas (Visão Executiva)

| Época | Modelo/Estratégia | Métricas-Chave | Status |
| --- | --- | --- | --- |
| **Época 1** | CatBoost FineTuned | Recall classe 0 = 0.743; Recall_macro = 0.732 | Melhor individual |
| **Época 1** | Stack 3 bases (LGBM+XGB+RF) | Recall classe 0 ≈ 0.33 | **Descartado** |
| **Época 1** | **Stack 2 bases (LGBM + CatBoost)** | **Recall classe 0 = 0.842; Recall_macro = 0.794** | **Vencedor da época 1** |
| **Época 2** | Refinos do Stack + meta LogisticRegression | Recall classe 0 ≈ 0.84; Recall_macro ≈ 0.79–0.80 | **Vencedor final** |

---

## 2. Configuração do Ambiente e Infraestrutura MLOps

### 2.1. Stack Tecnológico

A experimentação foi conduzida utilizando uma infraestrutura robusta de MLOps que garantiu reprodutibilidade, versionamento e rastreabilidade completa:

**Versionamento de Dados:**

- **DVC (Data Version Control):** Controle de versão dos datasets

- **Dagshub:** Plataforma integrada para versionamento e colaboração

- **Dagshub Data Engine:** API para acesso programático aos dados versionados

**Rastreamento de Experimentos:**

- **MLflow:** Plataforma central para logging de experimentos

- **MLflow Autolog:** Captura automática de parâmetros e métricas

- **MLflow Model Registry:** Versionamento e gestão de modelos

**Bibliotecas de Machine Learning:**

- **Scikit-learn:** Modelos tradicionais e métricas

- **XGBoost:** Gradient boosting otimizado

- **LightGBM:** Gradient boosting eficiente

- **CatBoost:** Gradient boosting para dados categóricos

### 2.2. Carregamento de Dados Versionados

O dataset processado foi carregado diretamente do repositório versionado, garantindo consistência entre experimentos:

```python
# Acesso via Dagshub Data Engine
ds = datasources.get('estrellacouto05/quantum-finance-credit-score', 'processed')
dataset_url = ds.head()[0].download_url

# Download autenticado com token
headers = {"Authorization": f"Bearer {dagshub_token}"}
response = requests.get(dataset_url, headers=headers)
df = pd.read_csv(io.StringIO(response.text))
```

**Comentário Técnico:** Esta abordagem garante que todos os experimentos utilizem exatamente a mesma versão dos dados, eliminando variabilidade relacionada a diferenças no dataset e permitindo comparações justas entre modelos.

### 2.3. Configuração do MLflow

O MLflow foi configurado para capturar automaticamente todos os aspectos dos experimentos:

```python
mlflow.autolog()  # Ativa logging automático
mlflow.set_tracking_uri("https://dagshub.com/estrellacouto05/quantum-finance-credit-score.mlflow")
```

**Comentário Técnico:** A integração MLflow + DagsHub permite rastreamento completo de experimentos com versionamento de código, dados e modelos em uma única plataforma, facilitando reprodutibilidade e colaboração.

---

## 3. Preparação dos Dados para Modelagem

### 3.1. Estrutura do Dataset Final

O dataset processado continha:

- **100.000 registros** (sem perda de dados)

- **48 features** após engenharia de variáveis

- **Variável alvo:** `score_credito` (0: Poor, 1: Standard, 2: Good)

### 3.2. Separação de Features e Target

```python
X = df.drop('score_credito', axis=1)  # 48 features
y = df['score_credito']               # Target (0, 1, 2)
```

**Comentário Técnico:** A separação clara entre features e target organiza o dataset para os modelos de Machine Learning, que aprenderão a prever o target com base nas features de forma supervisionada.

### 3.3. Divisão Treino/Teste

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# Resultado: 70.000 treino, 30.000 teste
```

**Comentário Técnico:** A divisão 70/30 com estratificação garante que a distribuição das classes seja mantida em ambos os conjuntos, evitando viés na avaliação e permitindo estimativas mais confiáveis da performance em produção.

### 3.4. Escalonamento com RobustScaler

A escolha do `RobustScaler` foi estratégica para lidar com outliers identificados na fase de EDA:

```python
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Comentário Técnico:** O `RobustScaler` utiliza mediana e IQR em vez de média e desvio padrão, sendo menos sensível a outliers que poderiam distorcer a escala das features. Esta escolha é particularmente importante em dados financeiros, onde outliers são comuns e informativos.

---

## 4. Função de Avaliação Padronizada

### 4.1. Desenvolvimento da Função `evaluate_and_log_model`

Para garantir consistência na avaliação, foi desenvolvida uma função centralizada que automatiza:

1. **Cálculo de métricas completas** por classe e agregadas

1. **Logging automático no MLflow** de todas as métricas

1. **Geração e salvamento da matriz de confusão**

1. **Registro do modelo** com assinatura apropriada

```python
def evaluate_and_log_model(kind, model_name, model, X_test, y_test):
    predictions = model.predict(X_test)
    
    # Métricas por classe
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, labels=[0,1,2], zero_division=0
    )
    
    # Destaque para classe 0 (Poor)
    mlflow.log_metric("Recall_class_0_Poor", recall[0])
    mlflow.log_metric("Precision_class_0_Poor", precision[0])
    mlflow.log_metric("F1_class_0_Poor", f1[0])
    
    # Matriz de confusão como artefato
    cm = confusion_matrix(y_test, predictions, labels=[0,1,2])
    # ... código para salvar e logar matriz
```

**Comentário Técnico:** Esta função padronizada elimina inconsistências na avaliação entre experimentos, garantindo que todas as métricas sejam calculadas da mesma forma e registradas automaticamente no MLflow para comparação posterior.

### 4.2. Métricas Priorizadas

- **Recall Classe 0 (Poor):** Métrica principal de negócio

- **Recall Macro:** Média do recall entre todas as classes

- **F1-Score Macro:** Harmônica entre precision e recall

- **Acurácia de Treino:** Para monitorar overfitting

**Comentário Técnico:** O foco no recall da classe 0 reflete a realidade de negócio onde falsos negativos (clientes Poor classificados como Good/Standard) têm custo muito maior que falsos positivos.

---

## 5. PRIMEIRA ÉPOCA DE EXPERIMENTAÇÃO

### 5.1. Estratégia da Primeira Época

A primeira época focou na avaliação de modelos individuais com otimização de hiperparâmetros via `RandomizedSearchCV`, seguida pela construção do primeiro ensemble. O objetivo era estabelecer baselines sólidos e identificar a melhor combinação de algoritmos.

**Comentário Técnico:** O uso de `RandomizedSearchCV` (n_iter≈30, cv=5) equilibra custo/benefício, permitindo explorar regiões promissoras do hiper-espaço com foco em recall_macro sem custo computacional excessivo.

### 5.2. Experimento 1: XGBoost Classifier

**Configuração:**

- `RandomizedSearchCV` com 30 iterações

- Validação cruzada com 5 folds

- Métrica de otimização: `recall_macro`

**Espaço de Hiperparâmetros:**

```python
param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}
```

**Resultados:**

- **Recall Classe 0 (Poor):** 0.727

- **F1-Score Macro:** 0.727

- **Acurácia:** 0.748

**Comentário Técnico:** O XGBoost foi avaliado com RandomizedSearchCV (cv=5), focando em parâmetros como n_estimators, max_depth, learning_rate, subsample, colsample_bytree e min_child_weight, com objetivo de maximizar o recall_macro. Os resultados estabeleceram um baseline sólido.

### 5.3. Experimento 2: LightGBM Classifier

**Configuração:**

- `RandomizedSearchCV` com 30 iterações

- Foco em hiperparâmetros específicos do LightGBM

**Espaço de Hiperparâmetros:**

```python
param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 9, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 63, 127],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
```

**Resultados:**

- **Recall Classe 0 (Poor):** 0.778

- **F1-Score Macro:** 0.766

- **Acurácia:** 0.780

**Comentário Técnico:** O LightGBM foi submetido a busca ampla seguida de refino com foco em regularização (reg_alpha/reg_lambda), num_leaves e amostragem (subsample/colsample_bytree), com tuning voltado a recall_macro e estabilidade entre classes. Superou o XGBoost em todas as métricas.

### 5.4. Experimento 3: Random Forest Classifier

**Configuração:**

- `RandomizedSearchCV` com 50 iterações (espaço maior)

- Foco em controle de overfitting

**Espaço de Hiperparâmetros:**

```python
param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],  # Sem None para evitar overfitting
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}
```

**Resultados:**

- **Recall Classe 0 (Poor):** 0.754

- **F1-Score Macro:** 0.751

- **Acurácia:** 0.768

**Comentário Técnico:** O Random Forest serviu como baseline forte de ensemble tradicional; ainda assim, mesmo com espaço de busca para n_estimators, max_depth e min_samples, não superou LGBM/CatBoost no objetivo de negócio.

### 5.5. Fine-Tuning do LightGBM

Dado o desempenho superior do LightGBM, foram realizados dois ciclos de fine-tuning:

#### 5.5.1. Fine-Tuning 1: Busca Refinada

**Estratégia:** Espaço de busca refinado em torno dos melhores parâmetros com 50 iterações e ranges mais estreitos.

**Resultados Fine-Tuning 1:**

- **Recall Classe 0 (Poor):** 0.786

- **F1-Score Macro:** 0.772

- **Acurácia:** 0.784

#### 5.5.2. Fine-Tuning 2: Foco em n_estimators e learning_rate

**Estratégia:** Teste específico de combinações `n_estimators` vs `learning_rate` mantendo outros parâmetros fixos.

**Configuração Testada:**

```python
# Modelo A (1º Fine-tune)
n_estimators = 500, learning_rate = 0.1
# Modelo B (2º Fine-tune)  
n_estimators = 1000, learning_rate = 0.05
```

**Resultados Fine-Tuning 2:**

- **Recall Classe 0 (Poor):** 0.789

- **F1-Score Macro:** 0.773

- **Acurácia:** 0.786

**Decisão Estratégica:** Apesar do modelo B ter performance ligeiramente superior (0.789 vs 0.786), optou-se pelo modelo A (`n_estimators=500`, `learning_rate=0.1`) devido ao melhor custo-benefício. A melhora mínima (0.003) não justificava o dobro do custo computacional e maior risco de overfitting.

**Comentário Técnico:** Esta decisão exemplifica o equilíbrio entre performance e eficiência operacional, considerando que o aumento nos estimadores eleva o custo computacional e o risco de overfitting sem retorno proporcional em desempenho.

### 5.6. Experimento 4: CatBoost Classifier

**Configuração:**

- `RandomizedSearchCV` com 30 iterações

- Métrica interna: `TotalF1` (adequada para multiclasse)

- Métrica de otimização externa: `recall_macro`

**Espaço de Hiperparâmetros:**

```python
param_distributions = {
    'iterations': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128, 255],
    'bagging_temperature': [0, 1, 5, 10]
}
```

**Resultados CatBoost:**

- **Recall Classe 0 (Poor):** 0.743

- **Recall Macro:** 0.732

- **F1-Score Macro:** 0.765

- **Acurácia:** 0.779

**Comentário Técnico:** CatBoost mostrou-se superior isoladamente na classe 0; na etapa seguinte, sua combinação com LightGBM elevou o recall de Poor para ~0.84. O CatBoost FineTuned obteve excelente performance, superando todos os modelos anteriores.

### 5.7. Primeiro Ensemble: Stacking LightGBM + CatBoost

Após avaliar os modelos individuais, foi construído o primeiro ensemble combinando os dois melhores algoritmos.

#### 5.7.1. Justificativa para o Ensemble

**Análise dos Resultados Individuais:**

- O **CatBoost FineTuned** obteve excelente performance na classe 0 (Poor) com recall de 0.743

- O **LightGBM FineTuned** apresentou boa cobertura da classe 2 (Good) com tempo de treinamento inferior

- Um Stacking inicial com **três modelos (LightGBM, XGBoost e RandomForest)** teve desempenho muito abaixo do esperado, com recall da classe 0 em apenas 0.33, sendo **descartado** por baixa eficácia e sobreposição de comportamento

**Comentário Técnico:** A decisão de descartar o ensemble de 3 modelos foi baseada na análise de que XGBoost e Random Forest apresentavam comportamentos muito similares ao LightGBM, não agregando diversidade suficiente ao ensemble.

#### 5.7.2. Configuração do Stacking

```python
estimators = [
    ('lgbm', lgbm_best_model),      # LightGBM otimizado
    ('catboost', catboost_best_model) # CatBoost otimizado
]

final_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,                    # Validação cruzada para meta-features
    passthrough=False        # Reduz risco de overfitting
)
```

**Comentário Técnico:** O uso de `passthrough=False` evita repassar features originais ao meta-learner, concentrando a decisão nas probabilidades/saídas dos modelos base e reduzindo risco de sobreajuste. A árvore de decisão como meta-learner oferece interpretabilidade e baixa variância.

#### 5.7.3. Resultados do Primeiro Stacking

- **Recall Classe 0 (Poor):** 0.842 🎯

- **Recall Macro:** 0.794

- **F1-Score Macro:** 0.785

- **Acurácia:** 0.792

**Análise:** Ao combinar apenas **os dois melhores modelos (CatBoost e LightGBM)**, alcançamos um **recall da classe 0 de 0.842** e um **recall_macro de 0.794**. Esse resultado evidencia que o ensemble **potencializou os pontos fortes de cada um** e contribuiu para **uma classificação mais equilibrada entre as três classes**.

**Comentário Técnico:** A validação cruzada externa (5 folds) foi aplicada para aferir generalização além do split fixo, diminuindo a chance de conclusões dependentes de uma única partição.

### 5.8. Conclusão da Primeira Época

| Modelo | Recall Classe 0 (Poor) | F1-Score Macro | Acurácia | Ranking |
| --- | --- | --- | --- | --- |
| **Stacking LGBM+CatBoost** | **0.842** | **0.785** | **0.792** | **1º** |
| LightGBM (Fine-tuned) | 0.789 | 0.773 | 0.786 | 2º |
| CatBoost (Fine-tuned) | 0.743 | 0.765 | 0.779 | 3º |
| Random Forest | 0.754 | 0.751 | 0.768 | 4º |
| XGBoost | 0.727 | 0.727 | 0.748 | 5º |

**Modelo Campeão da Primeira Época:** Stacking LightGBM + CatBoost

Com esse resultado, este modelo passou a ser o **candidato principal à produção** e seria submetido à validação cruzada externa antes de ser registrado no **MLflow Model Registry**.

---

## 6. SEGUNDA ÉPOCA DE EXPERIMENTAÇÃO

### 6.1. Estratégia da Segunda Época

A segunda época focou no **refinamento e otimização do ensemble vencedor**, buscando melhorar ainda mais a performance através de técnicas avançadas de regularização e controle de overfitting. O objetivo era superar o recall de 0.842 da classe Poor mantendo a estabilidade do modelo.

**Comentário Técnico:** A segunda época representa uma abordagem mais madura, priorizando robustez e generalização sobre performance máxima, refletindo considerações práticas para deployment em produção.

### 6.2. Análise do Modelo Base (Primeira Época)

Antes de iniciar as melhorias, foi realizada uma análise detalhada do modelo vencedor:

**Pontos Fortes:**

- Excelente recall na classe Poor (0.842)

- Boa generalização entre classes

- Ensemble diversificado e robusto

**Oportunidades de Melhoria:**

- Possível overfitting (gap treino-teste a ser investigado)

- Potencial para otimização de hiperparâmetros específicos

- Regularização mais agressiva dos base learners

### 6.3. Stacking 2A: Refinamento dos Parâmetros Base

**Objetivo:** Otimizar os hiperparâmetros dos base learners mantendo a arquitetura do ensemble.

**Estratégia:**

- Manter a estrutura LightGBM + CatBoost → Meta-learner

- Aplicar RandomizedSearchCV nos parâmetros dos base learners

- Trocar meta-learner para RandomForest (mais robusto que DecisionTree)

**Configuração LightGBM Otimizada (2A):**

```python
lgbm_params_2a = {
    'learning_rate': 0.1,
    'colsample_bytree': 0.8,
    'min_split_gain': 0.1,
    'subsample': 0.7,
    'subsample_freq': 1,
    'num_leaves': 95,
    'max_depth': 10,
    'min_child_samples': 60,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1,
    'n_estimators': 408
}
```

**Meta-learner Atualizado:**

```python
final_estimator = RandomForestClassifier(
    n_estimators=100, 
    max_depth=3, 
    random_state=42
)
```

**Resultados Stacking 2A:**

- **Recall Classe 0 (Poor):** 0.819

- **F1-Score Macro:** 0.776

- **Acurácia:** 0.785

- **Acurácia de Treino:** 0.925 (gap de ~14%)

**Análise:** Ligeira redução na performance, mas com melhor controle de overfitting devido ao meta-learner mais robusto.

### 6.4. Stacking 2B: Random Search em Torno do 2A

**Objetivo:** Busca direcionada em torno dos parâmetros do 2A para refinamento fino.

**Estratégia:**

- RandomizedSearchCV com espaço restrito ao redor do 2A

- Manter LightGBM fixo, otimizar apenas CatBoost

- Early stopping para controle de iterações

**Resultados Stacking 2B:**

- **Recall Classe 0 (Poor):** 0.821 (+0.2 p.p.)

- **F1-Score Macro:** 0.779

- **Acurácia:** 0.788

- **Acurácia de Treino:** 0.938 (gap de ~15%)

**Análise:** Melhoria marginal com aumento do overfitting - sinal de que o espaço estava sendo esgotado.

### 6.5. Stacking 3A: CatBoost Regularizado + Early Stopping

**Objetivo:** Reduzir overfitting através de regularização agressiva do CatBoost.

**Estratégia de Regularização:**

- `depth`: 10 → 8 (árvores mais rasas)

- `l2_leaf_reg`: 3 → 8 (regularização L2 aumentada)

- `rsm`: 0.85 (feature sampling)

- `subsample`: 0.8 com `bootstrap_type='Bernoulli'`

- Early stopping com patience=40

**Configuração CatBoost 3A:**

```python
cat_params_3a = {
    'learning_rate': 0.05,
    'depth': 8,                    # ↓ de 10 → 8
    'l2_leaf_reg': 8,              # ↑ de 3 → 8
    'rsm': 0.85,                   # feature sampling
    'subsample': 0.8,              # amostragem de registros
    'bootstrap_type': "Bernoulli",
    'random_strength': 1.5,
    'border_count': 64
}
```

**Resultados Stacking 3A:**

- **Recall Classe 0 (Poor):** 0.820

- **F1-Score Macro:** 0.776

- **Acurácia:** 0.784

- **Acurácia de Treino:** 0.920 (gap reduzido para ~13.6%)

**Análise:** Sucesso na redução do overfitting mantendo performance similar.

### 6.6. Stacking 3B-1: CatBoost Agressivamente Regularizado

**Objetivo:** Aplicar regularização ainda mais agressiva para maximizar generalização.

**Estratégia Mais Agressiva:**

- `depth`: 7 (ainda mais raso)

- `l2_leaf_reg`: 12 (regularização L2 máxima)

- `rsm`: 0.80 (menos features)

- `subsample`: 0.75 (menos amostras)

- `random_strength`: 2.5 (mais ruído controlado)

- `border_count`: 48 (menos bins)

**Configuração CatBoost 3B-1:**

```python
cat_params_3b1 = {
    'learning_rate': 0.05,
    'depth': 7,                    # ↓ de 8 → 7
    'l2_leaf_reg': 12,             # ↑ de 8 → 12
    'rsm': 0.80,                   # ↓ de 0.85 → 0.80
    'bootstrap_type': "Bernoulli",
    'subsample': 0.75,             # ↓ de 0.8 → 0.75
    'random_strength': 2.5,        # ↑ de 1.5 → 2.5
    'border_count': 48             # ↓ de 64 → 48
}
```

**Resultados Stacking 3B-1:**

- **Recall Classe 0 (Poor):** 0.829 🏆

- **F1-Score Macro:** 0.777

- **Acurácia:** 0.784

- **Acurácia de Treino:** 0.925

**Análise:** **Melhor recall para classe Poor até o momento!** A regularização agressiva conseguiu melhorar a métrica principal mantendo controle de overfitting.

### 6.7. Experimentação com Modelos Alternativos

Durante a segunda época, também foram testados modelos alternativos como potenciais substitutos ou adições ao ensemble:

#### 6.7.1. XGBoost como Modelo Alvo

Foram testadas três variantes do XGBoost para avaliar se poderiam superar o ensemble:

**XGBoost GBTree (Round 1):**

- **Recall Classe 0 (Poor):** 0.731

- **F1-Score Macro:** 0.732

**XGBoost LossGuide:**

- **Recall Classe 0 (Poor):** 0.712

- **F1-Score Macro:** 0.721

**XGBoost DART:**

- **Recall Classe 0 (Poor):** 0.700

- **F1-Score Macro:** 0.708

**Conclusão:** Todos os XGBoost ficaram significativamente abaixo do baseline de stacking, sendo descartados. Os XGB avaliados exibem viés alto na classe 0 e não atingem o patamar do stack atual.

#### 6.7.2. CatBoost Individual como Modelo Alvo

**CatBoost C1 (Anti-Overfit Baseline):**

- **Recall Classe 0 (Poor):** 0.690

- **F1-Score Macro:** 0.701

- **Acurácia de Treino:** 0.772 (baixo overfitting)

**CatBoost C2 (C1 + Class Weights):**

- `class_weights = [1.8, 1.0, 1.0]` para favorecer classe 0

- **Recall Classe 0 (Poor):** 0.812

- **F1-Score Macro:** 0.715

**CatBoost H2 (Ajuste Fino de Class Weights):**

- **Recall Classe 0 (Poor):** 0.800

- **F1-Score Macro:** 0.733

**Conclusão:** Embora o CatBoost individual com class weights tenha alcançado recall alto na classe Poor, o ensemble ainda superava em métricas gerais e robustez.

---

## 7. MODELO CAMPEÃO FINAL: Stacking 3B-1

### 7.1. Identificação do Modelo Vencedor

Após extensa experimentação em duas épocas, o **Stacking LGBM(2A) + CatBoost(3B-1) → RF Meta** foi identificado como o modelo campeão final.

**Run ID no MLflow:** `bccfb26c333a41cf94facfc225cc8f2c`

### 7.2. Configuração Final do Modelo Campeão

**Base Learners:**

1. **LightGBM (2A):**

```python
lgbm_params_final = {
    'learning_rate': 0.1,
    'colsample_bytree': 0.8,
    'min_split_gain': 0.1,
    'subsample': 0.7,
    'subsample_freq': 1,
    'num_leaves': 95,
    'max_depth': 10,
    'min_child_samples': 60,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1,
    'n_estimators': 408
}
```

1. **CatBoost (3B-1):**

```python
catboost_params_final = {
    'learning_rate': 0.05,
    'depth': 7,
    'l2_leaf_reg': 12,
    'rsm': 0.80,
    'bootstrap_type': "Bernoulli",
    'subsample': 0.75,
    'random_strength': 2.5,
    'border_count': 48,
    'iterations': [determinado via early stopping]
}
```

**Meta-learner:**

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)
```

### 7.3. Performance Final

| Métrica | Valor | Observações |
| --- | --- | --- |
| **Recall Classe 0 (Poor)** | **0.829** | **Métrica principal de negócio** |
| **Precision Classe 0 (Poor)** | **0.757** | Balanceamento adequado |
| **F1-Score Classe 0 (Poor)** | **0.792** | Harmônica otimizada |
| **Recall Macro** | **0.788** | Excelente performance geral |
| **F1-Score Macro** | **0.777** | Consistência entre classes |
| **Acurácia** | **0.784** | Performance geral sólida |
| **Acurácia de Treino** | **0.925** | Gap controlado (~14%) |

### 7.4. Justificativa da Escolha Final

**Por que o Stacking 3B-1 foi escolhido como campeão:**

1. **Melhor Recall Classe 0 (Poor):** 0.829 vs 0.842 da primeira época - pequena redução compensada por maior robustez

1. **Controle de Overfitting:** Gap treino-teste mais controlado através da regularização agressiva

1. **Estabilidade:** Regularização do CatBoost tornou o modelo mais estável e confiável

1. **Generalização:** Early stopping e técnicas de regularização melhoraram a capacidade de generalização

1. **Robustez:** Ensemble diversificado com base learners complementares

**Trade-off Aceito:** A pequena redução no recall da classe Poor (0.842 → 0.829) foi aceita em troca de um modelo significativamente mais robusto e com menor risco de overfitting em produção.

**Comentário Técnico:** Esta decisão exemplifica maturidade técnica, priorizando sustentabilidade e confiabilidade do modelo em produção sobre performance máxima em ambiente controlado.

---

## 8. Validação Cruzada Externa

### 8.1. Metodologia de Validação

Para confirmar a robustez do modelo campeão, foi realizada uma **validação cruzada externa** com 5 folds:

```python
skf_externo = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Comentário Técnico:** A validação externa utiliza dados completamente independentes do processo de otimização, fornecendo estimativa não enviesada da performance em produção.

### 8.2. Implementação da Validação Externa

```python
# Pipeline de validação cruzada externa
RUN_NAME = "ExternalCV_Stacking_LGBM_CatBoost3B1_RFMeta"
N_SPLITS_EXTERNO = 5

# Função helper para estimar iterations do CatBoost
def _best_iters_catboost(X, y, base_params, n_splits=5, patience=45):
    """Estima 'iterations' para CatBoost via CV interna simples."""
    skf_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    tr_idx, va_idx = next(iter(skf_inner.split(X, y)))
    # ... implementação do early stopping
    return optimal_iterations
```

### 8.3. Resultados da Validação Externa

| Fold | Recall Macro | Recall Poor | F1 Macro | Iterações CatBoost |
| --- | --- | --- | --- | --- |
| 1 | 0.7899 | 0.8343 | 0.7801 | 761 |
| 2 | 0.8013 | 0.8540 | 0.7836 | 1247 |
| 3 | 0.7959 | 0.8430 | 0.7836 | 1997 |
| 4 | 0.7972 | 0.8456 | 0.7845 | 1456 |
| 5 | 0.7952 | 0.8381 | 0.7862 | 1123 |

**Estatísticas Agregadas:**

- **Recall Macro:** 0.7959 ± 0.0045

- **Recall Poor:** 0.8430 ± 0.0092

- **F1 Macro:** 0.7836 ± 0.0034

### 8.4. Análise da Validação

**Confirmação da Robustez:** A validação cruzada externa confirmou que o modelo mantém performance estável e consistente, com recall da classe Poor variando apenas entre 0.8343 e 0.8540.

**Baixa Variância:** O desvio padrão baixo em todas as métricas indica que o modelo é robusto e não depende de divisões específicas dos dados.

**Generalização Confirmada:** Os resultados da validação externa são consistentes com os resultados do holdout test, confirmando a capacidade de generalização.

**Adaptabilidade do Early Stopping:** Embora as iterações ótimas do CatBoost tenham variado (761–1997), a performance permaneceu estável — bom sinal de robustez do ensemble.

---

## 9. Evolução da Performance ao Longo das Épocas

### 9.1. Trajetória Completa da Experimentação

| Época | Modelo | Recall Poor | F1 Macro | Estratégia | Comentário Técnico |
| --- | --- | --- | --- | --- | --- |
| 1ª | XGBoost Individual | 0.727 | 0.727 | Baseline | RandomizedSearchCV inicial |
| 1ª | LightGBM Individual | 0.778 | 0.766 | Otimização | Superou XGBoost |
| 1ª | CatBoost Individual | 0.743 | 0.765 | Fine-tuning | Melhor individual |
| 1ª | **Stacking LGBM+CatBoost** | **0.842** | **0.785** | **Primeiro Ensemble** | **Salto de 6+ p.p.** |
| 2ª | Stacking 2A | 0.819 | 0.776 | Refinamento | Meta-learner robusto |
| 2ª | Stacking 2B | 0.821 | 0.779 | Random Search | Melhoria marginal |
| 2ª | Stacking 3A | 0.820 | 0.776 | Regularização | Controle overfitting |
| 2ª | **Stacking 3B-1** | **0.829** | **0.777** | **Regularização Agressiva** | **Equilíbrio ótimo** |

### 9.2. Insights da Trajetória

**Salto Inicial:** O primeiro ensemble (0.842) representou um salto significativo em relação aos modelos individuais (+6 pontos percentuais).

**Refinamento Gradual:** A segunda época focou em refinamento e controle de overfitting, priorizando robustez sobre performance máxima.

**Decisão Estratégica:** A escolha do 3B-1 sobre o modelo da primeira época reflete uma decisão madura de priorizar estabilidade e generalização.

**Comentário Técnico:** A trajetória demonstra evolução natural do processo de ML: exploração inicial → otimização de performance → foco em robustez e produção.

---

## 10. Registro em Produção

### 10.1. Model Registry

O modelo campeão foi registrado no MLflow Model Registry:

```python
run_id = "bccfb26c333a41cf94facfc225cc8f2c"

mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="credit-score-model"
)
```

### 10.2. Versionamento

- **Nome do Modelo:** `credit-score-model`

- **Versão:** 3

- **Status:** Production Ready

- **Run ID:** `bccfb26c333a41cf94facfc225cc8f2c`

**Comentário Técnico:** O registro no Model Registry permite versionamento, promoção entre ambientes (staging → production) e rollback seguro se necessário.

---

## 11. Lições Aprendidas e Insights Estratégicos

### 11.1. Insights Técnicos Principais

1. **Ensembles Superam Modelos Individuais:** O primeiro stacking proporcionou ganho de 6+ pontos percentuais no recall da classe Poor.

1. **Regularização é Fundamental:** A versão 3B-1 com regularização agressiva encontrou o melhor equilíbrio entre performance e robustez.

1. **Trade-offs Conscientes:** Aceitar pequena redução na performance máxima em troca de maior estabilidade foi a decisão correta.

1. **Early Stopping Crítico:** Fundamental para encontrar o ponto ótimo de iterações sem overfitting.

1. **Diversidade no Ensemble:** LightGBM + CatBoost ofereceram complementaridade ideal sem redundância.

### 11.2. Estratégias de Experimentação Eficazes

1. **Experimentação Estruturada:** Divisão em épocas permitiu progressão lógica da complexidade.

1. **Foco na Métrica de Negócio:** Priorização consistente do recall da classe Poor guiou todas as decisões.

1. **Validação Rigorosa:** Validação cruzada externa confirmou robustez das escolhas.

1. **MLOps desde o Início:** Rastreamento completo facilitou comparações e reprodutibilidade.

1. **Análise de Trade-offs:** Consideração sistemática de performance vs. robustez vs. custo computacional.

### 11.3. Decisões Arquiteturais Acertadas

1. **Ensemble Diversificado:** LightGBM + CatBoost ofereceram complementaridade ideal.

1. **Meta-learner Simples:** RandomForest raso evitou overfitting no nível meta.

1. **Regularização Progressiva:** Abordagem gradual permitiu encontrar o ponto ótimo.

1. **Passthrough=False:** Concentrou decisão nas saídas dos base learners, reduzindo complexidade.

### 11.4. Comentários Técnicos Adicionais

**Sobre RandomizedSearchCV:** O uso de n_iter≈30 com cv=5 equilibrou custo/benefício, permitindo explorar regiões promissoras do hiper-espaço sem custo computacional excessivo.

**Sobre Validação Externa:** A aplicação de validação cruzada externa (5 folds) foi crucial para aferir generalização além do split fixo, diminuindo a chance de conclusões dependentes de uma única partição.

**Sobre Meta-learners:** A evolução de DecisionTree → RandomForest como meta-learner melhorou a generalização do stack, demonstrando importância da escolha do meta-modelo.

---

## 12. Próximos Passos e Recomendações

### 12.1. Melhorias Futuras

1. **Análise de Erros:** Investigar os 17.1% de falsos negativos da classe Poor para identificar padrões sistemáticos e oportunidades de feature engineering.

1. **Feature Engineering Avançada:** Explorar interações entre variáveis e features temporais baseadas nos erros identificados.

1. **Calibração de Probabilidades:** Implementar calibração (Platt scaling ou isotonic regression) para melhor interpretação das probabilidades de risco.

1. **Ensemble de Terceiro Nível:** Avaliar adição de um terceiro base learner ou técnicas de voting para aumentar diversidade.

1. **Otimização de Hiperparâmetros Bayesiana:** Substituir RandomizedSearchCV por Optuna ou similar para busca mais eficiente.

### 12.2. Monitoramento em Produção

1. **Drift Detection:** Monitorar mudanças na distribuição das features e performance do modelo usando técnicas como KS-test ou PSI.

1. **Performance Tracking:** Acompanhar métricas de negócio e técnicas em tempo real com alertas automáticos.

1. **Retreinamento Automático:** Estabelecer gatilhos baseados em degradação de performance ou drift significativo.

1. **A/B Testing:** Comparar performance com modelos alternativos em ambiente controlado antes de atualizações.

### 12.3. Considerações de Negócio

1. **Interpretabilidade:** Desenvolver explicações SHAP para decisões críticas do modelo, especialmente para clientes rejeitados.

1. **Fairness:** Avaliar vieses em diferentes segmentos demográficos e geográficos usando métricas de equidade.

1. **Regulamentação:** Garantir compliance com regulamentações financeiras (LGPD, Basel III, resolução 4.658 do BACEN).

1. **Impacto Financeiro:** Quantificar o valor gerado pela melhoria na identificação de clientes Poor através de análise de lift e ROI.

### 12.4. Aspectos Operacionais

1. **Latência:** Otimizar tempo de inferência para atender SLAs de negócio.

1. **Escalabilidade:** Preparar infraestrutura para volume crescente de predições.

1. **Backup e Recovery:** Implementar estratégias de contingência para falhas do modelo.

1. **Documentação:** Manter documentação técnica e de negócio atualizada para facilitar manutenção.

---

## 13. Conclusão

O projeto de desenvolvimento de modelos para previsão de score de crédito foi executado com rigor metodológico e resultou na identificação de uma solução robusta e eficaz. O **modelo campeão Stacking 3B-1** alcançou **82.9% de recall na classe Poor**, representando um equilíbrio otimizado entre performance e robustez.

A jornada da experimentação, estruturada em duas épocas distintas, demonstrou a evolução natural do processo de machine learning: da busca pela performance máxima (primeira época com 84.2% de recall) para a otimização da robustez e generalização (segunda época com 82.9% de recall, mas maior estabilidade).

### 13.1. Principais Conquistas

1. **Performance de Negócio:** 82.9% de recall na classe Poor atende aos objetivos críticos de identificação de risco.

1. **Robustez Comprovada:** Validação cruzada externa confirmou estabilidade com baixa variância (±0.0092).

1. **Arquitetura Sólida:** Ensemble diversificado com regularização adequada para produção.

1. **Processo Reprodutível:** MLOps completo garante manutenibilidade e evolução contínua.

1. **Decisões Fundamentadas:** Cada escolha técnica foi documentada e justificada com base em evidências empíricas.

### 13.2. Valor Técnico e de Negócio

**Técnico:**

- Framework replicável para projetos similares de ML em finanças

- Metodologia robusta de experimentação e validação

- Integração completa de ferramentas MLOps

**Negócio:**

- Redução significativa de falsos negativos em clientes de alto risco

- Modelo confiável para decisões críticas de crédito

- Base sólida para expansão e melhorias futuras

### 13.3. Reflexão Final

A abordagem metodológica, combinada com uma infraestrutura sólida de MLOps, não apenas garantiu a qualidade do modelo final, mas também estabeleceu um framework replicável para projetos similares de machine learning em contextos financeiros de alto risco.

O **Stacking 3B-1** representa não apenas um modelo de alta performance, mas uma solução equilibrada que prioriza a confiabilidade e a generalização necessárias para aplicações críticas de negócio. A documentação célula por célula garante que todo o conhecimento técnico seja preservado e transferível, facilitando manutenção, evolução e replicação do projeto.

**Comentário Técnico Final:** Este projeto exemplifica as melhores práticas em ciência de dados aplicada, demonstrando como combinar rigor científico, considerações práticas de negócio e excelência técnica para entregar soluções de machine learning de classe mundial.

