import param

from ..variables import Variable, Variables
from .base import WizardItem


class VariablesEditor(WizardItem):
    """
    Configure variables to use in your dashboard.
    """

    disabled = param.Boolean(default=True)

    ready = param.Boolean(default=True)

    variable_name = param.String(doc="Enter a name for the variable")

    variable_type = param.Selector(doc="Select the type of variable")

    variables = param.Dict(default={})

    layout = param.Selector(precedence=1)

    _template = """
      <span style="font-size: 2em"><b>Variable Editor</b></span>
      <p>{{ __doc__ }}</p>
      <fast-divider></fast-divider>
      <div style="display: flex;">
      <form role="form" style="flex: 20%; max-width: 250px; line-height: 2em;">
        <div style="display: grid;">
          <label for="variable-name-${id}"><b>{{ param.variable_name.label }}</b></label>
          <fast-text-field id="variable-name" placeholder="{{ param.variable_name.doc }}" value="${variable_name}">
          </fast-text-field>
        </div>
        <div style="display: flex;">
          <div style="display: grid; flex: auto;">
            <label for="type-${id}">
              <b>{{ param.variable_type.label }}</b>
            </label>
            <fast-select id="variable-select" style="min-width: 150px;" value="${variable_type}">
              {% for stype in param.variable_type.objects %}
              <fast-option id="variable-option-{{ loop.index0 }}" value="{{ stype }}">{{ stype.title() }}</fast-option>
              {% endfor %}
            </fast-select>
            <fast-tooltip anchor="type-${id}">{{ param.variable_type.doc }}</fast-tooltip>
          </div>
          <fast-button id="submit" appearance="accent" style="margin-top: auto; margin-left: 1em; width: 20px;" onclick="${_add_variable}" disabled="${disabled}">
            <b style="font-size: 2em;">+</b>
          </fast-button>
        </div>
      </form>
      <div id="variables" style="display: flex; flex-wrap: wrap; flex: 75%; margin-left: 1em;">
        {% for variable in variables.values() %}
        <div id="variable-container" style="margin: 0.5em">${variable}</div>
        {% endfor %}
      </div>
    </div>
    """

    _dom_events = {'variable-name': ['keyup']}

    def __init__(self, **params):
        super().__init__(**params)
        variables = param.concrete_descendents(Variable)
        self.param.variable_type.objects = types = [
            variable.variable_type for variable in variables.values()
        ]
        if self.variable_type is None and types:
            self.variable_type = types[0]

    @param.depends('variable_name', watch=True)
    def _enable_add(self):
        self.disabled = not bool(self.variable_name)

    def _add_variable(self, event):
        self.spec[self.variable_name] = spec = {
            'type': self.variable_type, 'name': self.variable_name
        }
        self.variables[self.variable_name] = Variable.from_spec(spec)
        self.param.trigger('variables')
        self.variable_name = ''
