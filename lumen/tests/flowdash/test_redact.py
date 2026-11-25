import json
import logging

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("duckdb")
pytest.importorskip("sqlglot")

from lumen.flowdash.redact import REDACTED, redact_spec

S = "S3CR3TXYZ"
ENV = {"MY_DB_PW": "envsecret99", "HOME": "/home/user", "SNOWFLAKE_ACCOUNT_LOCATOR": "acctlocator1",
       "SERVICE_PASSCODE": "passcode777", "OPENAI_API_KEY": "sk-abcdefghij", "PGPASS": "pgpass0000",
       "SHORT_TOKEN": "tok123"}
# Built in parts so the repository's private key check does not flag the test.
PEM_BEGIN, PEM_END = "-----BEGIN " + "PRIVATE KEY-----", "-----END " + "PRIVATE KEY-----"
CANARIES = [S, *(value for name, value in ENV.items() if name != "HOME")]

cases = {
    "uri_basic": {"url": f"postgresql://alice:{S}@host/db"},
    "uri_pct40": {"url": f"postgresql://alice:p%40{S}@host/db"},
    "uri_at_in_pw": {"url": f"postgresql://alice:p@{S}@host/db"},
    "uri_slash_in_pw": {"url": f"postgresql://alice:p/{S}@host/db"},
    "uri_hash_in_pw": {"url": f"postgresql://alice:p#{S}@host/db"},
    "uri_space_in_pw": {"url": f"postgresql://alice:p {S}@host/db"},
    "uri_token_as_user": {"url": f"https://{S}@github.com/org/repo.git"},
    "uri_query_password": {"url": f"postgresql://host/db?user=a&password={S}"},
    "uri_query_sslpassword": {"url": f"postgresql://host/db?sslmode=require&sslpassword={S}"},
    "uri_query_token": {"url": f"https://api.example.com/data.csv?token={S}"},
    "uri_azure_sas_sig": {"tables": {"t": f"SELECT * FROM 'https://acct.blob.core.windows.net/c/f.parquet?sv=2022&se=2030&sp=r&sig={S}'"}},
    "uri_amz_presigned": {"tables": {"t": f"SELECT * FROM 'https://b.s3.amazonaws.com/f.csv?X-Amz-Credential=AKIA/x&X-Amz-Signature={S}'"}},
    "uri_in_list": {"uris": ["sqlite:///a.db", f"mysql://u:{S}@h/d"]},
    "uri_nested": {"a": {"b": [{"c": f"mssql+pyodbc://u:{S}@h/d?driver=x"}]}},
    "uri_percent_query": {"url": f"postgresql://host/db?password%3D{S}"},
    "uri_jdbc": {"url": f"jdbc:postgresql://host/db;user=u;password={S}"},
    "odbc": {"url": f"Driver={{ODBC}};Server=h;UID=u;PWD={S};"},
    "upper_key": {"PASSWORD": S, "Api_Key": S},
    "api_key": {"api_key": S, "apiKey": S, "private_key_path": S, "aws_secret_access_key": S},
    "access_key_id": {"aws_access_key_id": S},
    "jwt": {"jwt": S},
    "bearer_header": {"headers": {"Authorization": f"Bearer {S}"}},
    "header_list_pairs": {"headers": [["Authorization", f"Bearer {S}"]]},
    "header_string": {"header": f"Authorization: Bearer {S}"},
    "header_string_basic": {"opts": f"Authorization: Basic {S}"},
    "x_api_key_header": {"headers": {"X-Api-Key": S}},
    "cookie": {"headers": {"Cookie": f"session={S}"}},
    "session_id": {"session_id": S},
    "sas": {"sas": S},
    "account_key_param": {"account_key": S},
    "connection_string_az": {"connection_string": f"DefaultEndpointsProtocol=https;AccountName=a;AccountKey={S};EndpointSuffix=core.windows.net"},
    "connection_string_sas": {"connection_string": f"BlobEndpoint=https://a.blob;SharedAccessSignature=sv=2022&ss=b&sig={S}"},
    "connect_args": {"connect_args": {"sslrootcert": "/x", "opt": S, "nested": {"x": S}, "lst": [S]}},
    "connection_args": {"connection_args": {"opt": S}},
    "kwargs": {"kwargs": {"opt": S}},
    "engine_kwargs": {"engine_kwargs": {"connect_args": {"x": S}}},
    "engine_kwargs_plain": {"engine_kwargs": {"x": S}},
    "query_params": {"query_params": {"sslpassword": S, "options": f"-c password={S}"}},
    "storage_options": {"storage_options": {"anon": False, "client_kwargs": {"x": S}}},
    "snowflake_pk_str": {"private_key": f"{PEM_BEGIN}\n{S}\n{PEM_END}"},
    "snowflake_pk_bytes": {"blob": S.encode()},
    "pem_unkeyed": {"x": f"{PEM_BEGIN}\n{S}\n{PEM_END}"},
    "bq_creds_dict": {"credentials_info": {"type": "service_account", "private_key": S, "private_key_id": S, "client_email": "x@y"}},
    "bq_creds_dict_other_key": {"service_account_info": {"private_key": S}},
    "bq_creds_json_string": {"service_account_info": json.dumps({"type": "service_account", "private_key": S, "private_key_id": S})},
    "bq_creds_json_string_spaces": {"info": json.dumps({"private_key": S}, indent=2)},
    "init_create_secret": {"initializers": [f"CREATE SECRET (TYPE s3, KEY_ID 'AKIA{S}', SECRET '{S}', REGION 'us')"]},
    "init_create_secret_case": {"initializers": [f"create  persistent secret s (type S3,key_id\n'A{S}',secret\t'{S}')"]},
    "init_create_secret_dollar": {"initializers": [f"CREATE SECRET (TYPE s3, KEY_ID 'a', SECRET $${S}$$)"]},
    "init_create_secret_estring": {"initializers": [f"CREATE SECRET (TYPE s3, KEY_ID 'a', SECRET E'{S}')"]},
    "init_secret_session_token": {"initializers": [f"CREATE SECRET (TYPE s3, SESSION_TOKEN '{S}')"]},
    "init_secret_hf": {"initializers": [f"CREATE SECRET hf (TYPE huggingface, TOKEN '{S}')"]},
    "init_secret_azure_cs": {"initializers": [f"CREATE SECRET a (TYPE azure, CONNECTION_STRING 'AccountName=x;AccountKey={S}')"]},
    "init_secret_azure_cs_sas": {"initializers": [f"CREATE SECRET a (TYPE azure, CONNECTION_STRING 'BlobEndpoint=https://x;SharedAccessSignature=sv=1&sig={S}')"]},
    "init_secret_azure_client": {"initializers": [f"CREATE SECRET a (TYPE azure, PROVIDER service_principal, TENANT_ID 't', CLIENT_ID 'c', CLIENT_SECRET '{S}', ACCOUNT_NAME 'n')"]},
    "init_set_eq": {"initializers": [f"SET s3_secret_access_key='{S}'"]},
    "init_set_to": {"initializers": [f"SET s3_secret_access_key TO '{S}'"]},
    "init_set_global_to": {"initializers": [f"SET GLOBAL s3_session_token TO '{S}'"]},
    "init_set_unquoted": {"initializers": [f"SET s3_secret_access_key={S}"]},
    "init_pragma": {"initializers": [f"PRAGMA s3_secret_access_key='{S}'"]},
    "init_attach_pg": {"initializers": [f"ATTACH 'dbname=d user=u password={S} host=h' AS db (TYPE postgres)"]},
    "init_attach_pg_quoted": {"initializers": [f"ATTACH 'dbname=d password=''{S}'' host=h' AS db (TYPE postgres)"]},
    "init_attach_mysql": {"initializers": [f"ATTACH 'host=h user=u passwd={S}' AS m (TYPE mysql)"]},
    "init_motherduck": {"uri": f"md:my_db?motherduck_token={S}"},
    "init_install_ext": {"extensions": ["httpfs", f"x{S}"]},
    "sql_where_literal_pw_col": {"tables": {"t": f"SELECT * FROM users WHERE password = '{S}'"}},
    "sql_with_pw_concat": {"sql_expr": f"SELECT '{S}' AS password"},
    "env_whole": {"user": "envsecret99"},
    "env_inside_named_pw": {"dsn": "host=h user=u envsecret99"},
    "env_inside_short_token": {"note": "x tok123 y"},
    "env_short_whole": {"user": "tok123"},
    "env_passcode": {"note": "abc passcode777 def"},
    "env_openai_inside": {"note": "use sk-abcdefghij here"},
    "env_pgpass_inside": {"note": "use pgpass0000 here"},
    "env_template_syntax": {"user": "{{ env('MY_DB_PW') }}"},
    "dollar_var": {"user": "$MY_DB_PW"},
    "int_password": {"password": 12345678},
    "secret_in_key_name": {S: "value"},
    "tuple_mirror": {"mirrors": {"t": ({"type": "sqlalchemy", "url": f"pg://u:{S}@h/d"}, "t")}},
    "mirror_table_named_secret": {"mirrors": {"secret": f"pg://u:{S}@h/d"}},
    "tables_list": {"tables": [f"read_csv('s3://AKIA:{S}@b/f.csv')"]},
    "table_named_password": {"tables": {"password": "SELECT 1"}},
    "excluded_tables_secret_name": {"excluded_tables": ["api_keys"]},
}
cases.update({
    "duckdb_http_headers_map": {"initializers": [
        f"CREATE SECRET http (TYPE http, EXTRA_HTTP_HEADERS MAP {{'Authorization': 'Bearer {S}'}})"
    ]},
    "duckdb_set_global_to": {"initializers": [f"SET GLOBAL s3_session_token TO '{S}'"]},
    "duckdb_reset_then_set": {"initializers": ["RESET s3_region", f"SET VARIABLE token = '{S}'"]},
    "duckdb_create_secret_lower": {"initializers": [f"create or replace secret (type gcs, key_id 'x', secret '{S}')"]},
    "duckdb_quoted_escape": {"initializers": [f"SELECT 'it''s {S}'"]},
    "duckdb_dollar_tag": {"initializers": [f"SELECT $tag${S}$tag$"]},
    "duckdb_comment": {"initializers": [f"SELECT 1 -- {S}", f"SELECT 2 /* {S} */"]},
    "duckdb_unparseable": {"initializers": [f"FROBNICATE {S} ((("]},
    "table_from_quoted_url": {"tables": {"t": f"SELECT * FROM 'https://{S}@host/f.csv'"}},
    "table_identifier_secret": {"tables": {"t": f'SELECT * FROM "x?token={S}"'}},
    "table_url_with_query": {"tables": {"t": f"https://host/data.parquet?sig={S}"}},
    "table_dict_spec": {"tables": {"t": {"url": f"https://u:{S}@h/f.csv", "token": S}}},
    "json_service_account_quoted_key": {"x": f'{{"private_key": "{S}"}}'},
    "bearer_in_string": {"options": f"Authorization: Bearer {S}"},
    "azure_sas_url": {"url": f"https://acct.blob.core.windows.net/c?sv=1&sig={S}"},
    "token_userinfo": {"uri": f"https://{S}@host/repo"},
    "storage_options_named": {"storage_options": {"key": S, "secret": S, "token": S, "endpoint_url": S}},
    "engine_kwargs_named": {"engine_kwargs": {"pool_size": 3, "creator": S}},
    "host_with_secret": {"host": f"{S}@db"},
    "database_with_query": {"database": f"/data/db.sqlite?password={S}"},
    "type_with_spaces": {"type": f"duck {S}"},
    "port_string": {"port": f"5432;password={S}"},
    "table_definition_text": {"tables": {"t": S + " x"}},
    "env_pw_suffix": {"note": "uses envsecret99"},
    "env_secret_in_url_path": {"url": "https://host/acctlocator1/f.csv"},
    "env_secret_in_table_path": {"tables": {"t": "/data/pgpass0000/f.csv"}},
    "env_short_in_sql": {"tables": {"t": "SELECT * FROM tok123"}},
    "bytes_anywhere": {"tables": {"t": S.encode()}, "initializers": [S.encode()]},
    "set_of_strings": {"x": {S}},
})


AWS = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
cases.update({
    "host_token": {"host": "ghp_S3CR3TXYZ"},
    "database_token": {"database": f"tok{S}"},
    "database_md": {"database": f"md:my_db?motherduck_token={S}"},
    "uri_md": {"uri": f"md:?motherduck_token={S}"},
    "host_userinfo": {"host": f"u:{S}@h"},
    "cache_dir_userinfo": {"cache_dir": f"s3://u:{S}@bucket/cache"},
    "cache_dir_query": {"cache_dir": f"/tmp/c?token={S}"},
    "root_path_secret": {"root": f"/home/u/{S}"},
    "url_path_token": {"url": f"https://hooks.example.com/services/{S}/data.csv"},
    "url_slash_pw": {"url": f"postgresql://u:pa/{S}@host/db"},
    "url_hash_pw": {"url": f"postgresql://u:p#{S}@host/db"},
    "url_digits_pw": {"url": f"postgresql://u:1234/{S}@host/db"},
    "url_fragment": {"url": f"https://h/f.csv#{S}"},
    "url_ipv6": {"url": f"postgresql://u:{S}@[::1]:5432/db"},
    "table_key_url_creds": {"tables": {f"s3://AKIA:{S}@b/f.csv": "SELECT 1"}},
    "table_key_secret_name": {"tables": {f"tok_{S}": "SELECT 1"}},
    "table_list_url_creds": {"tables": [f"https://u:{S}@h/f.csv"]},
    "table_value_url_query": {"tables": {"t": f"https://h/f.csv?sig={S}"}},
    "table_value_path_aws": {"tables": {"t": AWS}},
    "excluded_tables_url": {"excluded_tables": [f"https://u:{S}@h/x"]},
    "table_params": {"table_params": {"t": {"token": S}}},
    "mirror_key": {"mirrors": {f"https://u:{S}@h/x": "t"}},
    "sql_comment_line": {"tables": {"t": f"SELECT 1 -- token {S}\n"}},
    "sql_comment_block": {"tables": {"t": f"SELECT /* {S} */ 1"}},
    "sql_hint": {"tables": {"t": f"SELECT /*+ {S} */ 1"}},
    "sql_quoted_ident": {"tables": {"t": f'SELECT * FROM "AKIA{S}"'}},
    "sql_quoted_ident_alias": {"tables": {"t": f'SELECT 1 AS "{S}"'}},
    "sql_unquoted_ident": {"tables": {"t": f"SELECT 1 AS tok_{S}"}},
    "sql_read_csv_url": {"tables": {"t": f"SELECT * FROM read_csv('https://u:{S}@h/f.csv?token={S}')"}},
    "sql_read_csv_kw_aws": {"tables": {"t": f"SELECT * FROM read_csv('s3://b/f.csv', s3_secret_access_key='{AWS}')"}},
    "sql_literal_aws": {"tables": {"t": f"SELECT '{AWS}' AS k"}},
    "sql_literal_file": {"tables": {"t": f"SELECT 'x{S}.csv' AS k"}},
    "sql_insert_aws": {"initializers": [f"INSERT INTO creds VALUES ('{AWS}')"]},
    "sql_create_secret_aws": {"initializers": [f"CREATE SECRET (TYPE s3, KEY_ID 'AKIA', SECRET '{AWS}')"]},
    "sql_create_persistent_secret_aws": {"initializers": [f"CREATE PERSISTENT SECRET s (TYPE s3, SECRET '{AWS}')"]},
    "sql_create_table_as": {"initializers": [f"CREATE TABLE c AS SELECT '{AWS}' AS s"]},
    "sql_create_default": {"initializers": [f"CREATE TABLE c (s VARCHAR DEFAULT '{S}')"]},
    "sql_national": {"tables": {"t": f"SELECT N'{S}'"}},
    "sql_hex": {"tables": {"t": f"SELECT x'{S.encode().hex()}'"}},
    "sql_raw_dollar": {"tables": {"t": f"SELECT $${S}$$"}},
    "sql_raw_dollar_tag": {"tables": {"t": f"SELECT $tag${S}$tag$"}},
    "sql_estring": {"tables": {"t": f"SELECT E'{S}'"}},
    "sql_map_header": {"initializers": [f"CREATE SECRET h (TYPE http, EXTRA_HTTP_HEADERS MAP {{'Authorization': 'Bearer {S}'}})"]},
    "sql_struct_key": {"tables": {"t": f"SELECT {{'{S}': 1}} AS s"}},
    "sql_attach_conn": {"initializers": [f"ATTACH 'host=h password={S}' AS db (TYPE postgres)"]},
    "sql_attach_path": {"initializers": [f"ATTACH 's3://u:{S}@b/x.duckdb' AS db"]},
    "sql_set_variable": {"initializers": [f"SET VARIABLE tok = '{S}'"]},
    "sql_call": {"initializers": [f"CALL load_aws_credentials('{S}')"]},
    "sql_select_fn": {"initializers": [f"SELECT set_config('s3_secret', '{S}')"]},
    "sql_multi": {"initializers": [f"SELECT 1; SET s3_secret_access_key='{S}'; SELECT 2"]},
    "sql_semicolon_in_literal": {"initializers": [f"SELECT 'a;{S}'"]},
    "sql_expr_param": {"sql_expr": f"SELECT * FROM {{table}} WHERE token = '{S}'"},
    "sql_cast_ident": {"tables": {"t": f"SELECT CAST('{S}' AS VARCHAR)"}},
    "unparseable": {"tables": {"t": f"SELEC '{S}' FRM"}},
    "key_lower_secret": {f"tok{S.lower()}": "x"},
    "safe_setting_secret_val": {"warehouse": f"tok_{S}"},
    "schema_secret": {"schema": S},
    "nested_source": {"source": {"type": "sqlalchemy", "url": f"pg://u:{S}@h/d"}},
    "array": {"api_key": np.array([1, 2])},
    "frame": {"x": pd.DataFrame({"a": [1]})},
    "env_whole_short": {"host": "tok123"},
})
CANARIES += [S.lower(), AWS, S.encode().hex()]


@pytest.mark.parametrize("name", sorted(cases))
def test_no_secret_survives(name, caplog):
    with caplog.at_level(logging.DEBUG):
        text = yaml.safe_dump(redact_spec(cases[name], environ=ENV), sort_keys=False)
    for canary in CANARIES:
        assert canary not in text
        assert canary not in caplog.text


def test_safe_values_stay_readable():
    spec = {
        "type": "duckdb",
        "uri": "/data/warehouse.duckdb",
        "read_only": True,
        "initializers": ["INSTALL httpfs", "LOAD httpfs", "SET s3_region = 'eu-west-1'"],
        "tables": {
            "orders": "/data/orders.csv",
            "events": "s3://bucket/events/*.parquet",
            "by_region": "SELECT region, SUM(amount) AS total FROM read_parquet('/data/o.parquet') GROUP BY region",
            "nums": "SELECT i\nFROM range(500) t(i)  -- note\nWHERE i > 2",
        },
        "excluded_tables": ["tmp_scratch"],
    }
    redacted = redact_spec(spec, environ={})
    assert redacted["type"] == "duckdb"
    assert redacted["uri"] == "/data/warehouse.duckdb"
    assert redacted["read_only"] is True
    assert redacted["initializers"] == ["INSTALL httpfs", "LOAD ***", "SET ***"]
    assert redacted["tables"]["orders"] == "/data/orders.csv"
    assert redacted["tables"]["events"] == "s3://bucket/events/*.parquet"
    assert redacted["tables"]["by_region"] == spec["tables"]["by_region"]
    # SQL is shown as written, not as sqlglot writes it again, with comments dropped.
    assert redacted["tables"]["nums"] == "SELECT i\nFROM range(500) t(i)\nWHERE i > 2"
    assert redacted["excluded_tables"] == ["tmp_scratch"]


def test_connection_settings_show_only_where_they_are():
    spec = {
        "type": "sqlalchemy",
        "url": "postgresql+psycopg://alice:pw@db.example.com:5432/sales?sslmode=require",
        "host": "db.example.com",
        "port": 5432,
        "database": "sales",
        "schema": "public",
        "username": "alice",
        "schema_timeout_seconds": 600,
    }
    redacted = redact_spec(spec, environ={})
    assert redacted["url"] == "postgresql+psycopg://db.example.com:5432/sales"
    assert redacted["host"] == "db.example.com"
    assert redacted["port"] == 5432
    assert redacted["database"] == "sales"
    assert redacted["schema"] == "public"
    assert redacted["username"] == REDACTED
    assert redacted["schema_timeout_seconds"] == 600


@pytest.mark.parametrize("value", [np.array(["x", "y"]), pd.DataFrame({"a": [1]}), pd.Series([1]), object()])
@pytest.mark.parametrize("key", ["private_key", "tables", "anything", "initializers"])
def test_objects_yaml_cannot_show_are_named_not_printed(key, value):
    redacted = redact_spec({key: value}, environ={})
    assert redacted == {key: f"<{type(value).__name__}>"}
    yaml.safe_dump(redacted)


def test_masked_keys_stay_unique():
    redacted = redact_spec({"headers": {"Authorization": "a", "X-Api-Key": "b"}}, environ={})
    assert len(redacted["headers"]) == 2
    assert set(redacted["headers"].values()) == {REDACTED}
