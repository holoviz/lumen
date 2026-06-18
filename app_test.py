import os
from urllib.parse import quote_plus

import yaml
from dotenv import load_dotenv
import panel as pn

import lumen.ai as lmai
from lumen.sources.sqlalchemy import SQLAlchemySource


pn.extension()
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]


driver = "ODBC Driver 17 for SQL Server"

sqlserver_url = (
    "mssql+pyodbc://"
    f"{quote_plus(os.environ['SQLSERVER_USER'])}:"
    f"{quote_plus(os.environ['SQLSERVER_PASSWORD'])}@"
    f"{os.environ['SQLSERVER_HOST']}:"
    f"{os.getenv('SQLSERVER_PORT', '1433')}/"
    f"{os.environ['SQLSERVER_DATABASE']}"
    f"?driver={quote_plus(driver)}"
    "&TrustServerCertificate=yes"
)


tables = [
    "ons_capacidade_geracao",
    "ons_modalidade_usina",
    "ons_relacionamento_usina_conjunto",
    "ons_subestacao",
    "ons_linha_transmissao",
    "ons_carga_energia",
    "ons_curva_carga",
    "ons_intercambio_nacional",
    "ons_fator_capacidade",
    "ons_restricao_coff_eolica",
    "ons_restricao_coff_eolica_detail",
    "ons_restricao_coff_fotovoltaica",
    "ons_restricao_coff_fotovoltaica_detail",
    "ons_disponibilidade_usina",
    "ons_geracao_usina",
    "ccee_pld_horario",
    "ccee_pld_horario_submercado",
    "ccee_pld_media_diaria",
    "ccee_geracao_consolidada",
    "ccee_geracao_horaria",
    "ccee_lista_perfil_v1",
    "ccee_parcela_usina_montante_mensal",
    "ons_ear_diario_subsistema",
    "ons_ena_diario_subsistema",
    "ons_carga_mensal",
    "ccee_encargo_horario_submercado",
]


source = SQLAlchemySource(
    url=sqlserver_url,
    schema="dbo",
    tables=tables,
    database="dw_raw_open_data",
)


with open("data_dictionary.yaml", "r", encoding="utf-8") as f:
    dd = yaml.safe_load(f)


filtered_dd = {
    "alerts": dd.get("alerts", []),
    "tables": [
        table
        for table in dd.get("tables", [])
        if table.get("table") in set(tables)
    ],
}


data_dictionary_yaml = yaml.safe_dump(
    filtered_dd,
    allow_unicode=True,
    sort_keys=False,
)


common_context = f"""
{{{{ super() }}}}

CONTEXTO GERAL DO REbot

IDENTIFICADOR DO DICIONÁRIO:
DICIONARIO_REBOT_V1_2026

Você é um agente do REbot, assistente de dados de energia elétrica do Brasil.

Use o dicionário abaixo como fonte de verdade para responder perguntas e gerar SQL.

Se o usuário perguntar qual dicionário de dados está em uso, responda exatamente:
DICIONARIO_REBOT_V1_2026

SOURCE DISPONÍVEL

source: dw_raw_open_data
tipo: SQL Server
schema: dbo
uso: dados abertos e brutos de ONS e CCEE.

Use esta source para perguntas sobre:
- ONS
- CCEE
- PLD
- geração
- carga
- curva de carga
- intercâmbio
- EAR
- ENA
- restrições eólicas
- restrições fotovoltaicas
- capacidade instalada
- disponibilidade de usinas
- linhas de transmissão
- subestações

REGRAS SQL SERVER

- Dialeto: SQL Server.
- Use schema dbo.
- Use TOP N em vez de LIMIT.
- Não use sintaxe PostgreSQL.
- Não use sintaxe MySQL.
- Não invente tabelas.
- Não invente colunas.
- Respeite exatamente os nomes das colunas.
- Tabelas ONS normalmente usam colunas em snake_case.
- Tabelas CCEE normalmente usam colunas em MAIÚSCULAS.
- Antes de fazer JOIN, confira as chaves descritas no dicionário.
- Se a pergunta for ambígua quanto à granularidade, explique a escolha.
- Nunca use ORDER BY dentro de subqueries, derived tables, CTEs, views ou inline functions.
- Use ORDER BY somente no SELECT mais externo.
- Se precisar ordenar dentro de uma subquery para lógica de top N, use TOP junto com ORDER BY.

DICIONÁRIO DE DADOS

```yaml
{data_dictionary_yaml}

"""

chat_agent = lmai.agents.ChatAgent(
template_overrides={
"main": {
"context": common_context,
"instructions": """
{{ super() }}

Você é o ChatAgent do REbot.
Use o contexto geral do REbot para responder perguntas conceituais sobre o banco.
""",
}
}
)

table_agent = lmai.agents.TableListAgent(
template_overrides={
"main": {
"context": common_context,
}
}
)

sql_agent = lmai.agents.SQLAgent(
template_overrides={
"main": {
"context": common_context,
"instructions": """
{{ super() }}

Você é o SQLAgent customizado do REbot.

SENTINELA_OBRIGATORIA:
DICIONARIO_REBOT_V1_2026

Sempre use o dicionário de dados fornecido no contexto antes de gerar SQL.

Regras críticas:

Gere SQL compatível com SQL Server.
Use TOP em vez de LIMIT.
Use schema dbo.
Não use ORDER BY dentro de subqueries, CTEs ou derived tables, exceto quando usado com TOP ou OFFSET.
Use ORDER BY somente na query final externa.
Não invente colunas.
Não invente tabelas.
Antes de fazer JOIN, verifique as chaves descritas no dicionário.

Se o usuário perguntar qual dicionário está em uso, responda exatamente:
DICIONARIO_REBOT_V1_2026
""",
}
}
)

model_config = {
    "default": {"model": "openai/gpt-5.4-mini"},
    "chat": {"model": "openai/gpt-4.1-mini"},
    "sql": {"model": "openai/gpt-5.4-mini"},
    "vega_lite": {"model": "openai/gpt-5.4-mini"},
    "deck_gl": {"model": "openai/gpt-4.1-mini"},
    "analyst": {"model": "openai/gpt-5.4-mini"},
    "ui": {"model": "openai/gpt-5.4-mini"},
    "edit": {"model": "openai/gpt-4.1-mini"},
}

llm = lmai.llm.OpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    endpoint="https://openrouter.ai/api/v1",
    model_kwargs=model_config,
)

ui = lmai.ExplorerUI(
    data=source,
    llm=llm,
    agents=[
    chat_agent,
    table_agent,
    sql_agent,
    ],
    log_level="DEBUG",
)

ui.servable()