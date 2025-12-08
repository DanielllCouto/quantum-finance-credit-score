# Quantum Finance – Construção do Modelo de Previsão de Score de Crédito

## Sobre o Projeto

Este repositório é o pilar central do projeto de Machine Learning da Quantum Finance, dedicado à **construção e experimentação de um modelo robusto para previsão de score de crédito**. Ele abrange desde o **pré-processamento avançado de dados** e **engenharia de features**, passando pela **avaliação e otimização de múltiplos modelos supervisionados**, até a seleção final e registro do modelo campeão para implantação em produção.

Nosso objetivo primordial foi desenvolver uma solução de ML que não apenas entregasse alta **precisão**, mas que também fosse **reprodutível, rastreável e governável**.  
Para atingir esse objetivo, utilizamos um ecossistema integrado de ferramentas: **Dagshub, MLflow, DVC e GitHub**. ELas acompanham cada etapa do ciclo de vida do modelo, desde a limpeza e preparação dos dados até o registro e versionamento do modelo final. Essa integração garante **transparência, controle de versões e consistência entre ambientes**.

---
## Contexto do Projeto

Este repositório é parte de uma solução **end-to-end de Machine Learning** para previsão de score de crédito. Para uma visão completa da arquitetura, fluxo de CI/CD e integração entre os três componentes (modelagem, API e frontend), consulte o repositório central do projeto:

👉 **[Quantum Finance – Credit Score Project](https://github.com/DanielllCouto/quantum-finance-credit-score-project )**

Lá você encontrará a documentação geral, diagramas de arquitetura e uma visão holística de como este repositório se integra com a API de inferência e a aplicação Streamlit.

---

## Arquitetura da Solução

A arquitetura do projeto é modular e escalável, integrando ferramentas de MLOps para um ciclo de vida automatizado e governado.  
O **Dagshub** atua como plataforma central, unificando o versionamento de código (GitHub), dados (DVC com S3) e experimentos (MLflow). O fluxo inicia com o versionamento de dados brutos e processados via DVC, armazenados em um bucket S3.  

Os notebooks de desenvolvimento e experimentação utilizam esses dados versionados, e todos os experimentos são rastreados pelo **MLflow**, que registra parâmetros, métricas e artefatos. O modelo campeão é então registrado no **MLflow Model Registry**.  

Pipelines de **CI/CD com GitHub Actions** automatizam testes, geram relatórios de métricas e orquestram a notificação para o deploy da API de inferência, que consome o modelo registrado.  
Essa abordagem garante **rastreabilidade de ponta a ponta** e um ambiente de desenvolvimento e produção coeso.

---

## Tecnologias e Ferramentas Essenciais

* **Python 3.10+** – Linguagem principal para todo o pipeline de dados, modelagem e automação.  
* **Pandas, NumPy, Scikit-learn** – Manipulação de dados, pré-processamento, construção e avaliação de modelos base.  
* **LightGBM, CatBoost, XGBoost** – Algoritmos de Gradient Boosting de alta performance, amplamente utilizados na experimentação e no ensemble final.  
* **MLflow + DagsHub** – Plataforma MLOps integrada para rastreamento de experimentos, versionamento de datasets (DVC + S3) e gestão do ciclo de vida dos modelos (Model Registry).  
* **Matplotlib, Seaborn** – Visualização exploratória de dados (EDA) e análise de desempenho dos modelos.  
* **Pytest** – Framework de testes unitários e de integração para código e modelo.  
* **GitHub Actions** – Orquestração de pipelines de CI/CD, automação de testes, geração de relatórios e notificação de deploys.  

---

## Competências Técnicas Aprofundadas Demonstradas

### Pré-processamento Avançado de Dados

* **Tratamento de Dados:** Implementação de estratégias avançadas para tratamento de valores ausentes, utilizando **imputação por clusters KMeans** com base em variáveis correlacionadas, estatísticas e conhecimento de negócio.  
  Valores ausentes foram avaliados quanto ao **propósito da ausência** e, quando apresentavam **comportamento informativo**, foram **mantidos como variáveis indicativas (flags de missing)**.  
  Também foram realizadas **correções de inconsistências e mitigação de outliers** com base em análises estatísticas, garantindo a qualidade e robustez do dataset.

* **Mitigação de Data Leakage:** Identificação e exclusão preventiva de variáveis com risco de vazamento da variável alvo, assegurando que as métricas reflitam o desempenho real e a capacidade de generalização do modelo.

* **Engenharia de Features:** Criação de variáveis derivadas e transformações de distribuições, com **normalização e padronização dos dados** para otimizar sua representação e desempenho nos modelos de Machine Learning.

---

### Experimentação e Otimização de Modelos

* **Exploração de Algoritmos:** Foram testados diversos modelos de *ensemble*, como Random Forest, XGBoost, LightGBM e CatBoost, buscando entender o comportamento de cada um e identificar quais entregavam melhor desempenho e estabilidade para o problema de previsão de score de crédito.

* **Estratégias de Tuning:** Foram aplicadas técnicas de otimização de hiperparâmetros com GridSearchCV e RandomizedSearchCV para ajustar automaticamente as configurações dos modelos e alcançar o equilíbrio ideal entre precisão e capacidade de generalização, maximizando o desempenho geral obtido.

* **Construção de Ensembles (StackingClassifier):** Também foram desenvolvidos modelos *ensemble* utilizando a técnica de Stacking, combinando a força de múltiplos preditores para alcançar resultados superiores, com foco em LightGBM e CatBoost, que apresentaram o melhor desempenho nos experimentos.

---

### Validação e Avaliação

* **Validação Cruzada Externa:** Utilização de validação cruzada para aferir a generalização do modelo e garantir que as métricas de desempenho sejam robustas e representativas.

* **Métricas Estratégicas de Avaliação:** Foram adotadas métricas técnicas com **enfoque alinhado ao impacto de negócio**, priorizando a **identificação precisa de clientes de alto risco (classe 'Poor')**.  
  O **Recall da classe 'Poor'** foi considerado o principal indicador, uma vez que **maximizar a detecção de casos de risco** reduz diretamente **prejuízos financeiros decorrentes de classificações incorretas**.  
  Ainda assim, o modelo foi avaliado de forma **equilibrada e contextualizada**, considerando também o **Recall Macro**, o **F1-Score** e a **Matriz de Confusão**, garantindo que o desempenho geral não fosse comprometido em prol de uma única classe.

---

### Implementação de CI/CD e Monitoramento

Foram implementados pipelines de **Integração Contínua e Entrega Contínua (CI/CD)** via GitHub Actions, assegurando controle de qualidade, rastreabilidade e monitoramento contínuo do desempenho em produção.

* **Relatórios Automatizados de Métricas:** Um script dedicado conecta-se ao **MLflow** para comparar o modelo em produção com a versão experimental mais recente, gerando relatórios em formato Markdown com deltas de métricas (Δ). Essa análise oferece **feedback imediato sobre o impacto de novas alterações** no desempenho do modelo.

* **Testes Unitários do Modelo:** Implementação de testes robustos em **pytest** para garantir a integridade do modelo, validando carregamento, schema de entrada, previsões em dados conhecidos (*golden data*) e comportamento esperado em diferentes cenários.

* **Pipeline CI/CD com GitHub Actions:** Configuração de workflows automatizados para execução de testes, geração de relatórios e publicação de feedback direto nos Pull Requests. Essa automação garante **qualidade contínua, rastreabilidade de métricas e transparência no ciclo de desenvolvimento**.

* **Orquestração de Deploy da API:** Um segundo pipeline monitora o registro de novos modelos no **MLflow Model Registry** e, ao identificar um modelo campeão, **aciona automaticamente o workflow de deploy** no repositório da API de inferência, sincronizando os repositórios de modelo e API de forma segura via **GitHub REST API e OIDC Secrets**.

---

## Resultados Alcançados

✅ **Dataset Otimizado:** Processamento de dados resultou em um dataset balanceado, livre de *data leakage* e pronto para modelagem.  
✅ **Modelo Campeão de Alta Performance:** O modelo final, um **ensemble Stacking (LightGBM + CatBoost)**, alcançou **Recall superior a 0.83 na classe 'Poor'** em validação cruzada externa, superando significativamente os baselines.  
✅ **Pipeline Reprodutível:** Estabelecimento de um pipeline de ML totalmente documentado e reprodutível, desde a ingestão de dados até o registro do modelo.  
✅ **Integração MLOps Completa:** Implementação de CI/CD e versionamento de modelos via MLflow/DagsHub, assegurando governança e entrega contínua.

---

## Conecte-se

👨‍💻 **Daniel Estrella Couto**  
[LinkedIn](https://www.linkedin.com/in/daniel-estrella-couto) | [GitHub](https://github.com/estrellacouto05)
