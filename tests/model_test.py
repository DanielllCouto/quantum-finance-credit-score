import numpy as np
import mlflow
import pandas as pd
import pytest

"""
Módulo de testes para o modelo de credit score.

Contém testes de fumaça (smoke tests) e testes de dados fixos (golden data tests)
para verificar a integridade e o comportamento esperado do modelo.
"""

# Configuração do MLflow
mlflow.set_tracking_uri("https://dagshub.com/estrellacouto05/quantum-finance-credit-score.mlflow")

MODEL_URI = "models:/credit-score-model/latest"
model = mlflow.pyfunc.load_model(MODEL_URI)


# Define a lista de colunas esperadas pelo modelo
COLUMNS = [
    'idade', 'renda_anual', 'salario_liquido_mensal',
    'qtd_contas_bancarias', 'qtd_cartoes_credito', 'taxa_juros',
    'qtd_emprestimos', 'dias_atraso_pagamento', 'qtd_pagamentos_atrasados',
    'variacao_limite_credito', 'qtd_consultas_credito', 'divida_pendente',
    'percentual_utilizacao_credito', 'total_emprestimos_mensal',
    'valor_investido_mensal', 'saldo_mensal',
    'tempo_historico_credito_meses', 'ocupacao_Architect',
    'ocupacao_Developer', 'ocupacao_Doctor', 'ocupacao_Engineer',
    'ocupacao_Entrepreneur', 'ocupacao_Journalist', 'ocupacao_Lawyer',
    'ocupacao_Manager', 'ocupacao_Mechanic', 'ocupacao_Media_Manager',
    'ocupacao_Musician', 'ocupacao_Not Informed', 'ocupacao_Scientist',
    'ocupacao_Teacher', 'ocupacao_Writer',
    'pagamento_valor_minimo_Not Informed', 'pagamento_valor_minimo_Yes',
    'comportamento_pagamento_High_spent_Medium_value_payments',
    'comportamento_pagamento_High_spent_Small_value_payments',
    'comportamento_pagamento_Low_spent_Large_value_payments',
    'comportamento_pagamento_Low_spent_Medium_value_payments',
    'comportamento_pagamento_Low_spent_Small_value_payments',
    'tipos_emprestimos_Credit-Builder Loan',
    'tipos_emprestimos_Debt Consolidation Loan',
    'tipos_emprestimos_Home Equity Loan', 'tipos_emprestimos_Mortgage Loan',
    'tipos_emprestimos_Not Specified', 'tipos_emprestimos_Payday Loan',
    'tipos_emprestimos_Personal Loan', 'tipos_emprestimos_Student Loan',
    'tipos_emprestimos_Two or More Types of Loan',
]

# Definição dos tipos de dados para a conversão no DataFrame
COLUMN_DTYPES = {
    'qtd_cartoes_credito': 'int64',
    'dias_atraso_pagamento': 'int64',
    'tempo_historico_credito_meses': 'int64',
    'ocupacao_Architect': 'bool',
    'ocupacao_Developer': 'bool',
    'ocupacao_Doctor': 'bool',
    'ocupacao_Engineer': 'bool',
    'ocupacao_Entrepreneur': 'bool',
    'ocupacao_Journalist': 'bool',
    'ocupacao_Lawyer': 'bool',
    'ocupacao_Manager': 'bool',
    'ocupacao_Mechanic': 'bool',
    'ocupacao_Media_Manager': 'bool',
    'ocupacao_Musician': 'bool',
    'ocupacao_Not Informed': 'bool',
    'ocupacao_Scientist': 'bool',
    'ocupacao_Teacher': 'bool',
    'ocupacao_Writer': 'bool',
    'pagamento_valor_minimo_Not Informed': 'bool',
    'pagamento_valor_minimo_Yes': 'bool',
    'comportamento_pagamento_High_spent_Medium_value_payments': 'bool',
    'comportamento_pagamento_High_spent_Small_value_payments': 'bool',
    'comportamento_pagamento_Low_spent_Large_value_payments': 'bool',
    'comportamento_pagamento_Low_spent_Medium_value_payments': 'bool',
    'comportamento_pagamento_Low_spent_Small_value_payments': 'bool',
    'tipos_emprestimos_Credit-Builder Loan': 'bool',
    'tipos_emprestimos_Debt Consolidation Loan': 'bool',
    'tipos_emprestimos_Home Equity Loan': 'bool',
    'tipos_emprestimos_Mortgage Loan': 'bool',
    'tipos_emprestimos_Not Specified': 'bool',
    'tipos_emprestimos_Payday Loan': 'bool',
    'tipos_emprestimos_Personal Loan': 'bool',
    'tipos_emprestimos_Student Loan': 'bool',
    'tipos_emprestimos_Two or More Types of Loan': 'bool'
}


def prepare_data(data: dict) -> list:
    """
    Processa um dicionário de dados brutos para o formato de lista adequado
    para a criação do DataFrame de entrada do modelo.

    Args:
        data (dict): Dicionário com os dados de um único cliente.

    Returns:
        list: Lista de valores com tipos de dados mistos (numéricos e booleanos).
    """
    data_processed = []

    # Conversões de colunas numéricas
    data_processed.append(float(data["idade"]))
    data_processed.append(float(data["renda_anual"]))
    data_processed.append(float(data["salario_liquido_mensal"]))
    data_processed.append(float(data["qtd_contas_bancarias"]))
    data_processed.append(int(data["qtd_cartoes_credito"]))
    data_processed.append(float(data["taxa_juros"]))
    data_processed.append(float(data["qtd_emprestimos"]))
    data_processed.append(int(data["dias_atraso_pagamento"]))
    data_processed.append(float(data["qtd_pagamentos_atrasados"]))
    data_processed.append(float(data["variacao_limite_credito"]))
    data_processed.append(float(data["qtd_consultas_credito"]))
    data_processed.append(float(data["divida_pendente"]))
    data_processed.append(float(data["percentual_utilizacao_credito"]))
    data_processed.append(float(data["total_emprestimos_mensal"]))
    data_processed.append(float(data["valor_investido_mensal"]))
    data_processed.append(float(data["saldo_mensal"]))
    data_processed.append(int(data["tempo_historico_credito_meses"]))

    # Dicionário de categorias para one-hot encoding manual
    conditions = {
        "ocupacao": [
            "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
            "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager",
            "Musician", "Not Informed", "Scientist", "Teacher", "Writer"
        ],
        "pagamento_valor_minimo": ["Not Informed", "Yes"],
        "comportamento_pagamento": [
            "High_spent_Medium_value_payments",
            "High_spent_Small_value_payments",
            "Low_spent_Large_value_payments",
            "Low_spent_Medium_value_payments",
            "Low_spent_Small_value_payments"
        ],
        "tipos_emprestimos": [
            "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
            "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan",
            "Student Loan", "Two or More Types of Loan"
        ],
    }

    # Adiciona os valores booleanos para as colunas categóricas
    for key, values in conditions.items():
        for value in values:
            data_processed.append(data[key] == value)

    return data_processed


@pytest.mark.parametrize(
    "payload, expected_label, case_name",
    [
        # Caso 'bom pagador' -> esperado classe 2 (Good)
        (
            {
                "idade": 40, "renda_anual": 95000.0, "salario_liquido_mensal": 6200.0,
                "qtd_contas_bancarias": 3, "qtd_cartoes_credito": 2, "taxa_juros": 2.1,
                "qtd_emprestimos": 1, "dias_atraso_pagamento": 0, "qtd_pagamentos_atrasados": 0,
                "variacao_limite_credito": 750.0, "qtd_consultas_credito": 2, "divida_pendente": 1000.0,
                "percentual_utilizacao_credito": 25.0, "total_emprestimos_mensal": 500.0,
                "valor_investido_mensal": 1500.0, "saldo_mensal": 4500.0,
                "tempo_historico_credito_meses": 96, "ocupacao": "Manager",
                "pagamento_valor_minimo": "Not Informed",
                "comportamento_pagamento": "Low_spent_Small_value_payments",
                "tipos_emprestimos": "Not Specified",
            },
            2,
            "good_case",
        ),
        # Caso 'mau pagador' -> esperado classe 0 (Poor)
        (
            {
                "idade": 28, "renda_anual": 45000.0, "salario_liquido_mensal": 3200.0,
                "qtd_contas_bancarias": 1, "qtd_cartoes_credito": 1, "taxa_juros": 6.8,
                "qtd_emprestimos": 3, "dias_atraso_pagamento": 25, "qtd_pagamentos_atrasados": 2,
                "variacao_limite_credito": 100.0, "qtd_consultas_credito": 5, "divida_pendente": 8900.0,
                "percentual_utilizacao_credito": 85.0, "total_emprestimos_mensal": 1500.0,
                "valor_investido_mensal": 50.0, "saldo_mensal": 450.0,
                "tempo_historico_credito_meses": 15, "ocupacao": "Mechanic",
                "pagamento_valor_minimo": "Not Informed",
                "comportamento_pagamento": "High_spent_Small_value_payments",
                "tipos_emprestimos": "Payday Loan",
            },
            0,
            "poor_case",
        ),
    ],
)
def test_golden_data(payload: dict, expected_label: int, case_name: str):
    """
    Testa o modelo usando payloads fixos ("golden data") com resultados esperados.

    Verifica se a predição do modelo corresponde exatamente à classe esperada
    para um "Good" (2) e um "Poor" (0).
    """
    data_processed = prepare_data(payload)
    df_input = pd.DataFrame([data_processed], columns=COLUMNS)

    # Convertendo tipos de dados para corresponder ao schema do modelo no MLflow.
    df_input = df_input.astype(COLUMN_DTYPES)

    y_pred = model.predict(df_input)
    pred_label = int(y_pred[0])

    assert pred_label == expected_label, (
        f"[{case_name}] esperado={expected_label}, previsto={pred_label}"
    )


def test_model_load_call():
    """
    Smoke test: verifica se o modelo carrega corretamente e retorna uma predição válida.
    """
    payload = {
        "idade": 40, "renda_anual": 95000.0, "salario_liquido_mensal": 6200.0,
        "qtd_contas_bancarias": 3, "qtd_cartoes_credito": 2, "taxa_juros": 2.1,
        "qtd_emprestimos": 1, "dias_atraso_pagamento": 0, "qtd_pagamentos_atrasados": 0,
        "variacao_limite_credito": 750.0, "qtd_consultas_credito": 2, "divida_pendente": 1000.0,
        "percentual_utilizacao_credito": 25.0, "total_emprestimos_mensal": 500.0,
        "valor_investido_mensal": 1500.0, "saldo_mensal": 4500.0,
        "tempo_historico_credito_meses": 96, "ocupacao": "Manager",
        "pagamento_valor_minimo": "Not Informed",
        "comportamento_pagamento": "Low_spent_Small_value_payments",
        "tipos_emprestimos": "Not Specified",
    }

    data_processed = prepare_data(payload)
    df_input = pd.DataFrame([data_processed], columns=COLUMNS)

    # Convertendo tipos de dados para corresponder ao schema do modelo no MLflow.
    df_input = df_input.astype(COLUMN_DTYPES)

    y_pred = model.predict(df_input)
    pred = int(y_pred[0])

    # Verifica se a predição é um inteiro e se está dentro das classes esperadas.
    assert isinstance(pred, int), "A predição deve ser um inteiro"
    assert pred in {0, 1, 2}, f"Classe inválida: {pred} (esperado uma de {{0,1,2}})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([
        __file__, "-k", "test_golden_data or test_model_load_call", "-q"
    ]))