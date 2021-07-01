import param

from lumen.config import _TEMPLATES

from .base import WizardItem


class ConfigEditor(WizardItem):

    title = param.String(doc="Enter a title")

    logo = param.String(doc="Path or URL pointing to a logo.")

    template = param.Selector(precedence=1, objects=list(_TEMPLATES),
                              doc="Select the template")

    theme = param.Selector(default='default', doc="Select the theme")

    layout = param.Selector(precedence=1)

    _template = """
      <span style="font-size: 2em"><b>Config Editor</b></span>
      <p>Declare the high-level configuration of the dashboard.</p>
      <fast-divider></fast-divider>
      <form style="display: grid; line-height: 2em;">
        <label for="title"><b>Title</b></label>
        <fast-text-field id="title" placeholder="{{ param.title.doc }}" value="${title}"></fast-text-field>
        <label for="Logo"><b>Logo</b></label>
        <fast-text-field id="logo" placeholder="{{ param.logo.doc }}" value="${logo}"></fast-text-field>
        <label for="template-${id}"><b>Template</b></label>
        <fast-select id="template" value="${template}">
          {% for tname in param.template.objects %}
          <fast-option value="{{ tname.lower() }}">{{ tname }}</fast-option>
          {% endfor %}
        </fast-select>
        <fast-tooltip anchor="template-${id}">{{ param.template.doc }}</fast-tooltip>
        <label for="theme-${id}"><b>Theme</b></label>
        <fast-select id="theme" value="${theme}">
          <fast-option value="default">Default</fast-option>
          <fast-option value="dark">Dark</fast-option>
        </fast-select>
        <fast-tooltip anchor="theme-${id}">{{ param.theme.doc }}</fast-tooltip>
      </form>
    """

    _dom_events = {'title': ['change']}

    @param.depends('spec', watch=True)
    def _update_params(self):
        update = {key: value for key, value in self.spec.items()
                  if key in self.param}
        self.param.set_param(**update)

    def _update_spec(self, *events):
        for event in events:
            if event.new:
                self.spec[event.name] = event.new
