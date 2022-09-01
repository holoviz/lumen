from panel.template.base import BasicTemplate

from lumen.dashboard import Config


class CustomTemplate(BasicTemplate):
    pass


def test_config_template():
    config = Config.validate({'template': 'lumen.tests.test_config.CustomTemplate'})
    assert config['template'] is CustomTemplate
