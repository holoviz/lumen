from lumen.auth import YamlAuthMapperPlugin
from lumen.dashboard import AuthSpec

auth_mapper = """
group:
  admins:
    email:
      - lumen@holoviz.org
      - panel@holoviz.org
      - param@holoviz.org
  users:
    email:
      - philipp@holoviz.org
"""


def test_yaml_auth_mapper_plugin(yaml_file):
    yaml_file.write(auth_mapper)
    yaml_file.seek(0)
    yaml_file.flush()

    mapper = YamlAuthMapperPlugin(yaml_file=yaml_file.name)

    original = {
        'group': ['admins', 'users']
    }
    transformed = mapper.transform(original)

    assert transformed == {
        'email': [
            'lumen@holoviz.org',
            'panel@holoviz.org',
            'param@holoviz.org',
            'philipp@holoviz.org'
        ]
    }

def test_auth(state_userinfo):
    auth = AuthSpec(spec={'email': 'lumen@holoviz.org'})
    auth2 = AuthSpec(spec={'email': 'panel@holoviz.org'})
    assert auth.authorized
    assert not auth2.authorized


def test_auth_case_sensitive(state_userinfo):
    auth = AuthSpec(spec={'email': 'Lumen@holoviz.org'})
    auth2 = AuthSpec(spec={'email': ' Lumen@holoviz.org  '})
    auth3 = AuthSpec(spec={'email': 'Lumen@holoviz.org'}, case_sensitive=True)
    assert auth.authorized
    assert auth2.authorized
    assert not auth3.authorized
