# Documentação do Projeto de Machine Learning: Previsão de Score de Crédito - Quantum Finance

Esta documentação serve como um guia conciso e estratégico para o projeto de Machine Learning de previsão de score de crédito da Quantum Finance. Ela aborda as fases cruciais de inicialização do projeto, configuração do ambiente MLOps e a implementação de um robusto pipeline de CI/CD, garantindo a qualidade e a reprodutibilidade do modelo. Detalhes aprofundados sobre o processamento de dados e o desenvolvimento do modelo são fornecidos em documentos externos anexos, permitindo que esta documentação principal mantenha um foco estratégico nas práticas de MLOps e na entrega contínua.

## 1. Introdução e Visão Geral do Projeto

O projeto Quantum Finance Credit Score visa desenvolver um modelo de Machine Learning para prever o score de crédito de clientes, utilizando uma abordagem MLOps completa, desde a experimentação até a implantação em produção. O objetivo é fornecer uma ferramenta robusta e confiável para auxiliar nas decisões de concessão de crédito, minimizando riscos e otimizando processos.

### 1.1. Objetivo do Negócio

O principal objetivo é criar um sistema de previsão de score de crédito que seja:

- **Preciso:** Capaz de classificar corretamente os clientes em categorias de risco (Poor, Standard, Good).

- **Confiável:** Construído sobre uma base de dados limpa e um modelo robusto, com rastreabilidade completa.

- **Escalável:** Integrado em uma arquitetura que permita fácil manutenção e atualização.

- **Transparente:** Com processos documentados e automatizados para garantir a reprodutibilidade.

### 1.2. Arquitetura da Solução

A arquitetura da solução foi desenhada para ser modular, escalável e totalmente aderente aos princípios de MLOps, orquestrando um conjunto de ferramentas de ponta para garantir um ciclo de vida de Machine Learning completo e automatizado. O diagrama abaixo ilustra a interação entre os componentes chave do projeto:

![Diagrama de Arquitetura do Projeto](https://private-us-east-1.manuscdn.com/sessionFile/9w9D9DHlymp7RPTF7sCux4/sandbox/cBsrjaKVUcfEiZoLocWFUG-images_1758752977210_na1fn_L2hvbWUvdWJ1bnR1L2FycXVpdGV0dXJhX3Byb2pldG8.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOXc5RDlESGx5bXA3UlBURjdzQ3V4NC9zYW5kYm94L2NCc3JqYUtWVWNmRWlab0xvY1dGVUctaW1hZ2VzXzE3NTg3NTI5NzcyMTBfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRnljWFZwZEdWMGRYSmhYM0J5YjJwbGRHOC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=eGz-gH4VV4g6PQk56l2ntzmACmW4yM7rnKDx96jus5M8RdYRKJFk8D~6x9iIsuuZfOc~JqGGq1qN0FuiLS8hXN9UDVvVFJkUs6-baEkzduwC-pVe-oeULaMsejZKO7cRxKWulXIcK5ZwTMELERrH6Iqlh89Y4JoDDmUciLDes0usLg3RFxGQm7QPueNXyXE~cRbWZ1yKY0EJ7dJIhjCz7axH5fgnEAeRRxGI6R0l~vFDjyCIPb7fUkcEKPoe1bcOvcHRrcetPYRFNfQwlpN8qN3-W2KYYZAZPRV3QpbwX8J0Bi-xMYpvnFlPPRm56iysdqX6IElW09dJJjN3HWMO2A__)

O **Dagshub** atua como a plataforma central que unifica o versionamento de código, dados e experimentos, proporcionando uma visão 360º do projeto. A integração se dá da seguinte forma:

- **GitHub como Espelho de Código:** O código-fonte, hospedado primariamente no GitHub, é espelhado no Dagshub. Isso permite que o Dagshub tenha acesso ao código e o associe diretamente aos dados e experimentos, criando uma rastreabilidade completa.

- **DVC com Bucket S3 para Versionamento de Dados:** O **Data Version Control (DVC)** é utilizado para versionar os datasets, tratando grandes arquivos de dados da mesma forma que o Git trata o código. O DVC não armazena os dados diretamente no repositório Git; em vez disso, ele cria pequenos arquivos de metadados (ponteiros) que são versionados. Os dados reais são armazenados em um remote storage, que neste projeto é um **bucket S3** gerenciado pelo Dagshub. Essa abordagem permite:
  - **Reprodutibilidade:** Garante que qualquer versão do código possa ser restaurada com a versão exata dos dados com que foi treinada.
  - **Eficiência:** Mantém o repositório Git leve e rápido, pois os arquivos de dados pesados ficam em um armazenamento externo otimizado (S3).

- **MLflow como Central de Experimentos:** O **MLflow** é a ferramenta escolhida para o rastreamento de experimentos. O servidor MLflow, também hospedado e integrado ao Dagshub, funciona como uma central para todos os runs de treinamento. Para cada experimento, o MLflow registra:
  - **Parâmetros:** Os hiperparâmetros utilizados no treinamento do modelo.
  - **Métricas:** Os resultados de desempenho (ex: Acurácia, F1-Score, Recall).
  - **Artefatos:** Incluindo o próprio modelo treinado, gráficos de avaliação e outros arquivos relevantes.
  - **Model Registry:** O MLflow Model Registry é utilizado para gerenciar o ciclo de vida dos modelos, promovendo versões para estágios como "Staging" e "Production".

Essa tríade (DVC, S3, MLflow) sob a gestão do Dagshub cria um ambiente de MLOps coeso, onde cada componente está interligado, desde o código que gera um modelo até os dados que o alimentaram e as métricas que validaram seu desempenho.

## 2. Inicialização e Configuração do Projeto

Esta seção detalha as etapas iniciais para configurar o ambiente de desenvolvimento e estabelecer as bases para um projeto de Machine Learning robusto e versionado, seguindo as melhores práticas de MLOps.

### 2.1. Estrutura de Pastas do Projeto

A organização do projeto segue uma estrutura padronizada, inspirada no template `Cookiecutter Data Science`, que facilita a colaboração, a manutenção e a escalabilidade. Esta estrutura é fundamental para a aplicação das práticas de MLOps, separando claramente o código-fonte, dados, notebooks, modelos e relatórios.

**Descrição dos Principais Diretórios:**

- **`.dvc/`**: Contém os metadados e configurações do Data Version Control (DVC), que gerencia o versionamento de grandes arquivos de dados.

- **`.github/workflows/`**: Armazena os arquivos de configuração dos workflows do GitHub Actions, responsáveis pela automação de CI/CD.

- **`data/`**: Diretório central para todos os dados do projeto, subdividido em:
  - **`data/raw/`**: Dados brutos e originais, intocados após a ingestão.
  - **`data/processed/`**: Dados limpos e pré-processados, prontos para a modelagem.

- **`documentacao/`**: Contém a documentação do projeto, incluindo esta documentação principal e os documentos externos detalhados.

- **`models/`**: Armazena scripts relacionados à definição e registro de modelos.

- **`notebooks/`**: Contém os notebooks Jupyter utilizados para experimentação, análise exploratória de dados e desenvolvimento de modelos.

- **`reports/`**: Destinado a relatórios gerados pelo projeto, como o relatório de métricas do MLflow e figuras.

- **`tests/`**: Contém os testes unitários e de integração para o código e o modelo.



### 2.2. Criação da Estrutura do Projeto com Cookiecutter

O projeto foi iniciado utilizando o template `Cookiecutter Data Science`, uma ferramenta que padroniza a estrutura de diretórios e arquivos, promovendo organização e consistência desde o primeiro dia.

- **Ferramenta:** Cookiecutter Data Science.

- **Comando:** `cookiecutter https://github.com/drivendata/cookiecutter-data-science` (ou similar, se `ccds` for um alias configurado).

- **Informações Fornecidas:**
  - **Nome do Projeto:** `quantum-finance-credit-score`
  - **Descrição:** "Score de crédito alternativo baseado em dados recentes, com entrega via API segura e integração com app em Streamlit para empresas parceiras. Projeto completo com pipeline de ML, rastreamento e versionamento."
  - **Autor:** Daniel Estrella Couto

- **Modificações Iniciais:**
  - Ajustes pontuais na estrutura de pastas para melhor adequação ao projeto.
  - Inserção do dataset original (`credit_score_data.csv`) na pasta `data/raw`.
  - Criação de notebooks iniciais para análise exploratória e desenvolvimento do modelo na pasta `notebooks/`.
  - Atualização do arquivo `README.md` com a descrição detalhada do projeto.

### 2.2. Versionamento de Código com GitHub

O GitHub foi utilizado para versionar todo o código-fonte do projeto, garantindo rastreabilidade, colaboração e controle de alterações.

- **Repositório:** `quantum-finance-credit-score` (e outros, como `quantum-finance-app-credit-score`, `quantum-finance-api-credit-score`).

- **Processo:**
  - Criação do repositório no GitHub.
  - Inicialização do repositório local e sincronização com o remoto.
  - Criação da branch principal (`main`).
  - Primeiro commit da estrutura inicial e publicação no GitHub.

### 2.3. Configuração do Ambiente Python

Para garantir um ambiente de desenvolvimento isolado e reprodutível, o `pyenv` e ambientes virtuais Python foram utilizados.

- **`pyenv`****:** Instalado para gerenciar múltiplas versões do Python e garantir que a versão correta (e consistente) seja utilizada no projeto.

- **Ambiente Virtual:** Criado com o comando `python -m venv .venv`.

- **Ativação:** O ambiente virtual foi ativado e configurado no VS Code, isolando as dependências do projeto do ambiente global do sistema.

### 2.4. Instalação das Dependências

Todas as bibliotecas necessárias para o projeto foram listadas e instaladas, garantindo que o ambiente de execução seja consistente em diferentes máquinas.

- **Arquivo:** `requirements.txt`.

- **Bibliotecas Chave (exemplos):** `pandas`, `numpy`, `scikit-learn`, `dagshub`, `mlflow`, `dvc-s3`, `xgboost`, `lightgbm`, `catboost`, `pytest`, `python-dotenv`.

- **Comando de Instalação:** `pip install -r requirements.txt`.

- **Verificação:** A instalação foi validada através da extensão de ambientes Python no VS Code.

### 2.5. Configuração do MLOps com Dagshub: DVC, S3 e MLflow

A integração com o Dagshub é o coração da estratégia de MLOps deste projeto, centralizando o versionamento de código, dados e experimentos em uma única plataforma. Esta seção detalha a configuração do DVC para versionamento de dados com o S3 do Dagshub e a conexão com o servidor MLflow para rastreamento de experimentos.

#### 2.5.1. Versionamento de Dados com DVC e S3 do Dagshub

O Data Version Control (DVC) foi configurado para rastrear as alterações nos datasets, garantindo a reprodutibilidade dos experimentos. O Dagshub fornece um bucket S3 otimizado para atuar como o armazenamento remoto para o DVC, mantendo o repositório Git leve e ágil.

- **Inicialização do DVC:** O primeiro passo foi inicializar o DVC no repositório do projeto com o comando `dvc init`. Isso cria a estrutura `.dvc/` que armazena as configurações e metadados.

- **Configuração do Remote Storage (S3 do Dagshub):** A conexão com o bucket S3 do Dagshub foi estabelecida através dos seguintes comandos, que podem ser obtidos diretamente da interface do Dagshub:
  - **Importante:** As credenciais de acesso (`access_key_id` e `secret_access_key`) são sensíveis e foram configuradas localmente (`--local`) para não serem comitadas no Git. Em um ambiente de produção, elas devem ser gerenciadas através de variáveis de ambiente ou secrets.

- **Rastreamento e Versionamento dos Dados:**
    1. O dataset inicial (`credit_score_data.csv`) foi adicionado ao diretório `data/raw/`.
    1. O DVC foi instruído a rastrear o diretório `data/` com o comando `dvc add data`. Isso cria o arquivo `data.dvc`, um ponteiro que contém o hash MD5 dos dados e informações de localização.
    1. O arquivo `data.dvc` e o `.gitignore` (atualizado pelo DVC) foram comitados no Git: `git commit -m "Start tracking data directory with DVC"`.
    1. Finalmente, os dados foram enviados para o bucket S3 no Dagshub com o comando `dvc push`. A partir deste ponto, qualquer pessoa com acesso ao repositório pode baixar a versão correta dos dados com `dvc pull`.

#### 2.5.2. Rastreamento de Experimentos com MLflow no Dagshub

O Dagshub também fornece um servidor MLflow totalmente integrado, que atua como a central para o registro e a comparação de todos os experimentos de Machine Learning.

- **Configuração do MLflow Tracking URI:** Para que os experimentos executados localmente fossem registrados no Dagshub, o `MLFLOW_TRACKING_URI` foi configurado nos notebooks de desenvolvimento. Isso é feito através da biblioteca `dagshub`:

- **Benefícios da Integração:**
  - **Centralização:** Todos os experimentos, com seus parâmetros, métricas e artefatos, ficam visíveis na interface do Dagshub, na aba "Experiments".
  - **Comparação Facilitada:** A interface do MLflow no Dagshub permite comparar facilmente diferentes runs, identificar os melhores modelos e analisar o impacto de diferentes hiperparâmetros.
  - **Rastreabilidade Completa:** O Dagshub associa automaticamente cada experimento ao commit Git correspondente, criando um link direto entre o código, os dados (via DVC) e os resultados do modelo (via MLflow). Isso garante uma rastreabilidade de ponta a ponta, fundamental para a governança e a depuração do modelo.

## 3. Análise Exploratória e Processamento de Dados (Documentação Externa)

Para uma análise detalhada das etapas de Análise Exploratória de Dados (AED) e pré-processamento, incluindo a limpeza, padronização, engenharia de features e transformação de variáveis categóricas, consulte a documentação externa:

- **Documento:** [documentacao_tecnica_porocessamento_de_dados_credit_score_quantum_finance.md](documentacao_tecnica_porocessamento_de_dados_credit_score_quantum_finance.md)

Este documento externo oferece uma explicação célula a célula do notebook `processamento-dados.ipynb`, detalhando cada decisão e transformação aplicada aos dados brutos para prepará-los para a modelagem.

## 4. Desenvolvimento e Experimentação do Modelo (Documentação Externa)

Para uma compreensão aprofundada do desenvolvimento, treinamento e experimentação dos modelos de Machine Learning, incluindo a seleção de algoritmos, a estratégia de ensemble (Stacking), o rastreamento de experimentos com MLflow e a evolução do modelo, consulte a documentação externa:

- **Documento:** [documentacao_tecnica_experimentacao_e_desenvolvimento_de_modelo_credit_score_quantum_finance.md](documentacao_tecnica_experimentacao_e_desenvolvimento_de_modelo_credit_score_quantum_finance.md)

Este documento externo fornece uma explicação célula a célula do notebook `desenvolvimento-modelo.ipynb`, cobrindo desde a importação de bibliotecas até a seleção e registro do modelo campeão, incluindo a consideração de múltiplos fluxos de teste e a evolução do modelo.

## 5. Implementação de CI/CD e Monitoramento

Um dos pilares deste projeto MLOps é a automação do ciclo de vida do modelo através de pipelines de Integração Contínua e Entrega Contínua (CI/CD), garantindo a qualidade, rastreabilidade e observabilidade contínua do modelo em produção.

### 5.1. Geração de Relatório de Métricas (`reports/report.py`)

Um script Python (`reports/report.py`) foi desenvolvido para gerar relatórios comparativos de métricas entre o modelo em produção e a última execução experimental. Este relatório é crucial para a tomada de decisão e para o monitoramento do desempenho do modelo.

- **Funcionalidade:** Conecta-se ao MLflow, busca runs relevantes e compara as métricas da versão de produção com as da execução experimental mais recente.

- **Estrutura do Código (Resumo):**
  - **Configuração MLflow:** Define o `tracking_uri` para o Dagshub e inicializa o `MlflowClient`.
  - **Seleção da Versão de Produção:** Busca a versão mais recente do modelo no Model Registry (`credit-score-model`) e extrai suas métricas.
  - **Seleção da Execução Experimental:** Busca os runs mais recentes do experimento, filtrando por métricas relevantes (ex: `Recall_class_0_Poor > 0`), e extrai suas métricas.
  - **Formatação e Comparação:** Gera um relatório em formato Markdown (`mlflow_report.md`) que exibe as métricas de produção, as métricas experimentais e a diferença (`Δ`) entre elas. Isso fornece um feedback imediato sobre o impacto de novas alterações.

### 5.2. Testes Unitários do Modelo (`tests/model_test.py`)

Testes unitários robustos são implementados para garantir a integridade e o comportamento esperado do modelo em diferentes cenários.

- **Funcionalidade:** Garante que o modelo carrega corretamente, aceita payloads válidos e classifica casos de `golden data` de forma consistente.

- **Estrutura do Código (Resumo):**
  - **Configuração:** Carrega o modelo mais recente do MLflow Model Registry (`models:/credit-score-model/latest`) usando `mlflow.pyfunc.load_model`.
  - **Schema de Entrada:** Define a ordem exata das colunas (`COLUMNS`) e seus tipos de dados (`COLUMN_DTYPES`) para garantir a consistência do input do modelo.
  - **Pré-processamento:** Uma função `prepare_data` é implementada para converter os dados de entrada (payload) para o formato esperado pelo modelo, incluindo conversões numéricas e One-Hot Encoding manual para variáveis categóricas.
  - **Testes de ****`Golden Data`****:** Utiliza `pytest.mark.parametrize` para testar múltiplos cenários com payloads específicos (`payload_good`, `payload_poor`) e seus `expected_label` correspondentes. Isso verifica se o modelo classifica corretamente exemplos conhecidos.
  - **`Smoke Test`****:** Um teste básico (`test_model_load_call`) que verifica se o modelo carrega e retorna uma previsão válida (um inteiro entre 0, 1 ou 2), garantindo a integridade mínima do modelo.

### 5.3. Pipeline de CI/CD com GitHub Actions (`model_report.yml`)

Um workflow de GitHub Actions foi configurado para automatizar a execução dos testes e a geração do relatório de métricas a cada Pull Request.

- **Arquivo:** `.github/workflows/model_report.yml`.

- **Trigger:** `on: pull_request` para a branch `main`.

- **Permissões:** `contents: read`, `pull-requests: write` (para comentar no PR).

- **Passos Principais:**
    1. **`Checkout code`****:** Clona o repositório.
    1. **`Setup Python`****:** Configura o ambiente Python (versão 3.11).
    1. **`Install dependencies`****:** Instala as bibliotecas listadas em `requirements.txt`.
    1. **`Unit testing`****:** Executa os testes definidos em `tests/model_test.py` usando `pytest`. Se algum teste falhar, o PR é marcado como falho.
    1. **`Generate MLflow Report`****:** Executa o script `reports/report.py` para gerar o `mlflow_report.md`.
    1. **`Comment on Pull Request`****:** Utiliza a action `marocchino/sticky-pull-request-comment@v2` para publicar o conteúdo do `mlflow_report.md` como um comentário fixo no Pull Request. Isso fornece feedback imediato aos desenvolvedores e revisores.

- **Benefícios:**
  - **Qualidade Automatizada:** Testes unitários garantem que novas alterações não quebrem o modelo.
  - **Rastreabilidade de Métricas:** Comparação direta entre produção e experimental no próprio PR.
  - **Feedback Imediato:** Desenvolvedores recebem feedback sobre o impacto de suas mudanças no desempenho do modelo sem sair do GitHub.

### 5.4. Notificação e Orquestração de Deploy da API (`notify_api_deploy.yml`)

Um segundo workflow de GitHub Actions é responsável por orquestrar o deploy da API de inferência do modelo, garantindo que a API seja atualizada sempre que um novo modelo campeão for registrado.

- **Arquivo:** `.github/workflows/notify_api_deploy.yml`.

- **Objetivo:** Sincronizar o ciclo de treinamento/registro do modelo (no repositório `quantum-finance-credit-score`) com o deploy da API (no repositório `quantum-finance-api-credit-score`).

- **Trigger:** `on: push` na branch `main`, com filtro de paths para `notebooks/**`. Isso significa que o workflow é disparado apenas quando há alterações nos notebooks, indicando uma possível atualização do modelo.

- **Escopo:**
  - **Repositório:** `quantum-finance-credit-score` (modelo).
  - **Workflow:** `notify_api_deploy.yml`.
  - **Repositório Notificado:** `quantum-finance-api-credit-score` (API).

- **Integrações:** MLflow (Dagshub), GitHub Actions (OIDC/Secrets), GitHub REST API (`workflow_dispatch`).

- **Passos Principais (Resumo):**
    1. **Configuração de Ambiente:** Utiliza segredos do GitHub para autenticação no MLflow (Dagshub).
    1. **Lógica de Notificação:** O workflow contém lógica para verificar se um novo modelo foi registrado e, em caso afirmativo, dispara remotamente o workflow de deploy da API no repositório `quantum-finance-api-credit-score` via `workflow_dispatch` da GitHub REST API. Isso é feito usando um token GitHub com permissões adequadas (`ACCESS_CREDIT_SCORE`).

## 6. Conclusão e Próximos Passos

Este projeto demonstra uma implementação completa de um pipeline MLOps para um modelo de previsão de score de crédito, cobrindo desde a inicialização estruturada até a automação de CI/CD e orquestração de deploy. As práticas adotadas garantem a reprodutibilidade, rastreabilidade e a entrega contínua de valor.

### 6.1. Resumo dos Resultados

- **Estrutura de Projeto:** Utilização de Cookiecutter para padronização.

- **Versionamento:** GitHub para código, DVC/Dagshub para dados, MLflow para experimentos e modelos.

- **Automação:** Pipelines de CI/CD com GitHub Actions para testes, relatórios de métricas e orquestração de deploy.

- **Modelo:** Desenvolvimento e seleção de modelos robustos (LightGBM, CatBoost, RandomForest, Stacking).

### 6.2. Sugestões de Melhorias e Próximos Passos

- **Monitoramento de Drift:** Implementar ferramentas para monitorar o drift de dados e modelos em produção, alertando sobre degradação de desempenho.

- **Exploração de Novas Features:** Investigar fontes de dados adicionais ou técnicas avançadas de engenharia de features para melhorar ainda mais a performance do modelo.

- **Otimização de Hiperparâmetros:** Utilizar ferramentas como Optuna ou Hyperopt para otimização automatizada de hiperparâmetros de forma mais eficiente.

- **Explicação do Modelo (XAI):** Integrar técnicas de eXplainable AI (XAI) para aumentar a interpretabilidade do modelo, fornecendo insights sobre as decisões de crédito.

- **A/B Testing:** Implementar um framework para A/B testing de diferentes versões do modelo em produção.

---

**Documentos Externos Anexos:**

- [documentacao_tecnica_porocessamento_de_dados_credit_score_quantum_finance.md](documentacao_tecnica_porocessamento_de_dados_credit_score_quantum_finance.md)

- [documentacao_tecnica_experimentacao_e_desenvolvimento_de_modelo_credit_score_quantum_finance.md](documentacao_tecnica_experimentacao_e_desenvolvimento_de_modelo_credit_score_quantum_finance.md)