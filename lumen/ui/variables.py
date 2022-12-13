import pathlib

import panel as pn
import param  # type: ignore
import yaml

from panel import Param
from panel.widgets import PasswordInput

from ..state import state
from ..util import expand_spec
from ..variables import Variable, Variables
from .base import WizardItem


class VariablesEditor(WizardItem):
    """
    Configure variables to use in your dashboard.
    """

    disabled = param.Boolean(default=True)

    enabled = param.List()

    ready = param.Boolean(default=True)

    secure = param.Boolean(default=False)

    variable_name = param.String(doc="Enter a name for the variable")

    variable_type = param.Selector(doc="Select the type of variable")

    variables = param.Dict(default={})

    layout = param.Selector(precedence=1)

    path = param.Foldername()

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
        <div style="display: grid;">
          <label for="secure-${id}"><b>{{ param.secure.label }}</b></label>
          <fast-switch id="secure" checked="${secure}"></fast-switch>
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
        <div id="variable-container" style="margin: 0.5em; position: relative;">
        <fast-switch id="selected-{{ variable.name }}" {% if variable.name in enabled %}checked{% endif %} style="position: absolute; right: 0px; z-index: 100;" onchange="${script('toggle')}"></fast-switch>
        ${variable}
        </div>
        {% endfor %}
      </div>
    </div>
    """ # noqa

    _dom_events = {'variable-name': ['keyup']}

    _scripts = {
        'toggle': """
          var variable = event.target.id.split('-')[1]
          var enabled = [...data.enabled]
          if (enabled.indexOf(variable) > -1) {
            if (!event.target.checked) {
              enabled.splice(enabled.indexOf(variable), 1)
            }
          } else if (event.target.checked) {
            enabled.push(variable)
          }
          data.enabled = enabled
        """
    }

    _glob_pattern = '*.y*ml'

    def __init__(self, **params):
        super().__init__(**params)
        variables = param.concrete_descendents(Variable)
        self.param.variable_type.objects = types = [
            variable.variable_type or f'{variable.__module__}.{variable.__name__}'
            for variable in variables.values()
        ]
        if self.variable_type is None and types:
            self.variable_type = types[0]
        self._variables = state._variables[pn.state.curdoc] = Variables()
        path = params.get('path', self.path)
        components = pathlib.Path(path).glob(self._glob_pattern)
        for source in components:
            with open(source, encoding='utf-8') as f:
                yaml_spec = f.read()
                spec = yaml.safe_load(expand_spec(yaml_spec))
            if not spec:
                continue
            for varname, varspec in spec.items():
                self._add_from_spec(dict(name=varname, **varspec))

    def _get_varspec(self, var):
        variable = self._variables._vars[var]
        if variable.materialize:
            variable = variable.as_materialized()
        varspec = {
            k: v for k, v in variable.param.values().items()
            if k != 'value' and v != variable.param[k].default
        }
        vartype = type(variable)
        if variable.variable_type and vartype.__module__.startswith('lumen.'):
            varspec['type'] = variable.variable_type
        else:
            varspec['type'] = f'{vartype.__module__}.{vartype.__name__}'
        return varspec

    @param.depends('enabled', watch=True)
    def _update_enabled(self, *events):
        self.spec.clear()
        var_specs = {var: self._get_varspec(var) for var in self.enabled}
        self.spec.update(var_specs)
        ready = True
        for name in self.enabled:
            var = self._variables._vars[name]
            if var.required and not var.value:
                ready = False
        self.ready = ready

    @param.depends('variable_name', watch=True)
    def _enable_add(self):
        self.disabled = not bool(self.variable_name)

    def _add_variable(self, event=None):
        spec = {
            'type': self.variable_type,
            'name': self.variable_name,
            'secure': self.secure
        }
        self._add_from_spec(spec)
        self.variable_name = ''

    def _add_from_spec(self, spec, enable=False):
        varname, vartype, secure = spec['name'], spec['type'], spec.get('secure', False)
        enabled = spec.pop('enabled', enable)
        variable = Variable.from_spec(spec, self._variables)
        self._variables.add_variable(variable)
        if 'key' in variable.param and not variable.key:
            variable.key = variable.name
        variable.default = variable.value
        params = [
            p for p in variable.param if p not in (
                'name', 'materialize', 'required', 'secure', 'value'
            )
        ]
        widgets = {}
        if secure:
            widgets['default'] = PasswordInput
        var_editor = Param(variable, parameters=params, widgets=widgets).layout
        var_editor[0].value = f'{varname} ({vartype})'
        self.variables[varname] = var_editor
        variable.param.watch(self._update_enabled, list(variable.param))
        if enabled:
            self.enabled = self.enabled + [varname]
        self.param.trigger('variables')
