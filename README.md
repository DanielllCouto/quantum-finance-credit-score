# Quantum Finance – Construção do Modelo de Previsão de Score de Crédito

## Sobre o Projeto

Este repositório é o pilar central do projeto de Machine Learning da Quantum Finance, dedicado à **construção e experimentação de um modelo robusto para previsão de score de crédito**. Ele abrange desde o **pré-processamento avançado de dados** e **engenharia de features**, passando pela **avaliação e otimização de múltiplos modelos supervisionados**, até a seleção final e registro do modelo campeão para implantação em produção.

Nosso objetivo primordial foi desenvolver uma solução de ML que não apenas entregasse alta **precisão**, mas que também fosse intrinsecamente **reprodutível, rastreável e governável**. Para isso, integramos uma infraestrutura **MLOps de ponta a ponta**, garantindo que cada etapa do ciclo de vida do modelo seja automatizada e transparente. 

## Tecnologias e Ferramentas Essenciais

*   **Python 3.10+**: Linguagem principal para todo o pipeline de dados, modelagem e automação.
*   **Pandas, NumPy, Scikit-learn**: Bibliotecas fundamentais para manipulação de dados, pré-processamento, construção de modelos base e avaliação.
*   **LightGBM, CatBoost, XGBoost**: Algoritmos de Gradient Boosting de alta performance, extensivamente utilizados na experimentação e na construção do modelo ensemble campeão.
*   **MLflow + DagsHub**: Plataforma MLOps integrada para rastreamento de experimentos, versionamento de datasets (DVC com S3) e gestão do ciclo de vida dos modelos (MLflow Model Registry).
*   **Matplotlib, Seaborn**: Ferramentas para visualização exploratória de dados (EDA) e análise de desempenho do modelo.
*   **Pytest**: Framework robusto para testes unitários e de integração do código e do modelo.
*   **GitHub Actions**: Orquestração de pipelines de Integração Contínua (CI) para automação de testes, geração de relatórios de métricas e notificação para deploy da API.

## Competências Técnicas Aprofundadas Demonstradas

### Pré-processamento Avançado de Dados

*   **Tratamento de Dados:** Implementação de estratégias sofisticadas para tratamento de valores ausentes (imputação por clusters KMeans), correção de inconsistências e mitigação de outliers, garantindo a qualidade e a robustez do dataset.
*   **Engenharia de Features:** Criação de variáveis derivadas e transformações de distribuições para otimizar a representação dos dados para os modelos de ML.
*   **Mitigação de Data Leakage:** Identificação e remoção proativa de variáveis que poderiam introduzir *data leakage* (ex: `mix_credito`), assegurando a validade das métricas de avaliação e a generalização do modelo.

### Experimentação e Otimização de Modelos

*   **Exploração de Algoritmos:** Avaliação comparativa de uma gama de modelos de *ensemble*, incluindo Random Forest, XGBoost, LightGBM e CatBoost, para identificar os mais adequados ao problema.
*   **Estratégias de Tuning:** Aplicação de técnicas de otimização de hiperparâmetros como `GridSearchCV` e `RandomizedSearchCV` para maximizar o desempenho dos modelos.
*   **Construção de Ensembles (StackingClassifier):** Desenvolvimento de modelos *ensemble* utilizando a técnica de Stacking, combinando a força de múltiplos preditores para alcançar resultados superiores, com foco em LightGBM e CatBoost.

### Validação e Avaliação Rigorosa

*   **Validação Cruzada Externa:** Utilização de validação cruzada para aferir a generalização do modelo e garantir que as métricas de desempenho sejam robustas e representativas.
*   **Métricas de Negócio:** Foco estratégico em métricas alinhadas aos objetivos de negócio, como **Recall Macro** e, crucialmente, o **Recall da classe 'Poor' (alto risco)**, além de F1-Score e Matriz de Confusão, para uma avaliação completa e contextualizada.
*   **Interpretação de Resultados:** Capacidade de traduzir métricas técnicas em insights acionáveis para o negócio, justificando as decisões de modelagem.

### MLOps na Experimentação e Governança

*   **Rastreabilidade Completa com MLflow/DagsHub:** Registro detalhado de cada experimento, incluindo parâmetros, métricas e artefatos (modelos serializados, gráficos), garantindo a capacidade de reproduzir qualquer `run` a qualquer momento.
*   **Versionamento de Experimentos e Modelos:** Utilização do DagsHub para versionar não apenas o código, mas também os dados (via DVC) e os modelos (via MLflow Model Registry), criando um histórico imutável de todas as iterações do projeto.
*   **Pipeline Reprodutível:** Construção de um pipeline que garante a reprodutibilidade de todas as etapas, desde o pré-processamento até a exportação do dataset final e o registro do modelo.

## Arquitetura da Solução

A arquitetura do projeto é modular e escalável, integrando ferramentas de MLOps para um ciclo de vida automatizado e governado. O **Dagshub** atua como a plataforma central, unificando o versionamento de código (GitHub), dados (DVC com S3) e experimentos (MLflow). O fluxo inicia com o versionamento de dados brutos e processados via DVC, armazenados em um bucket S3. Os notebooks de desenvolvimento e experimentação utilizam esses dados versionados, e todos os experimentos são rastreados pelo MLflow, que registra parâmetros, métricas e artefatos. O modelo campeão é então registrado no MLflow Model Registry. Pipelines de CI/CD com GitHub Actions automatizam testes, geram relatórios de métricas e orquestram a notificação para o deploy da API de inferência, que consome o modelo registrado. Esta abordagem garante rastreabilidade de ponta a ponta e um ambiente de desenvolvimento e produção coeso.

## Resultados Alcançados

✅ **Dataset Otimizado:** Processamento de dados resultou em um dataset balanceado, livre de *data leakage* e pronto para modelagem.  
✅ **Modelo Campeão de Alta Performance:** O modelo final, um **ensemble Stacking (LightGBM + CatBoost)**, demonstrou um **Recall superior a 0.83 na classe 'Poor'** em validação cruzada externa, superando significativamente os baselines.  
✅ **Pipeline Reprodutível:** Estabelecimento de um pipeline de ML totalmente documentado e reprodutível, desde a ingestão de dados até o registro do modelo.  
✅ **Integração MLOps Completa:** Implementação de CI/CD e versionamento de modelos via MLflow/DagsHub, assegurando a governança e a entrega contínua.  

## Boas Práticas MLOps Implementadas

Este projeto exemplifica a aplicação de diversas boas práticas de MLOps, garantindo a robustez e a sustentabilidade da solução de Machine Learning:

*   **Versionamento de Dados e Modelos:** Essencial para a reprodutibilidade e rastreabilidade, permitindo auditar e reverter a qualquer versão do dataset ou modelo.
*   **Detecção e Mitigação de Data Leakage:** Crucial para a validade das métricas de avaliação e para a generalização do modelo em cenários reais.
*   **Validação Cruzada Externa:** Garante que o desempenho do modelo seja robusto e não dependente de uma única partição de dados.
*   **Estratégias de Ensemble:** Utilização de técnicas avançadas como Stacking para combinar a força de múltiplos modelos, otimizando a performance e a resiliência.
*   **Tuning Automatizado de Hiperparâmetros:** Otimização sistemática dos modelos para extrair o máximo de desempenho de forma eficiente.
*   **Documentação Técnica Abrangente:** Detalhamento de todas as etapas do projeto, desde o pré-processamento até a experimentação, facilitando a colaboração e a manutenção.
*   **Integração CI/CD:** Automação de testes e geração de relatórios para feedback contínuo e garantia de qualidade, acelerando o ciclo de desenvolvimento e deploy.

## Conecte-se

👨‍💻 **Daniel Estrella Couto**
[LinkedIn](https://www.linkedin.com/in/daniel-estrella-couto) | [GitHub](https://github.com/estrellacouto05)
