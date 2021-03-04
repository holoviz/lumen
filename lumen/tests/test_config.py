from lumen.dashboard import Config

from panel.template.base import BasicTemplate

class TestTemplate(BasicTemplate):
    pass

def test_config_template():
    config = Config.from_spec({'template': 'lumen.tests.test_config.TestTemplate'})
    assert config.template is TestTemplate
