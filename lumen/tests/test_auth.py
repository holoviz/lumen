from lumen.auth import YamlAuthMapperPlugin


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
