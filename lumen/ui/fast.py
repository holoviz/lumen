import param

from panel.io.server import init_doc, state
from panel.layout.base import ListLike, NamedListLike
from panel.reactive import ReactiveHTML
from panel.widgets.select import SelectBase


class FastDesignProvider(ListLike, ReactiveHTML):

    _template = '<fast-design-system-provider id="fdsp" use-defaults>${objects}</fast-design-system-provider>'


class FastComponent(ReactiveHTML):

    sizing_mode = param.ObjectSelector(default='stretch_width', objects=[
        'fixed', 'stretch_width', 'stretch_height', 'stretch_both',
        'scale_width', 'scale_height', 'scale_both', None])

    __js_modules__ = [
        'https://unpkg.com/@microsoft/fast-components@1.13.0'
    ]

    __abstract = True

    def get_root(self, doc=None, comm=None, preprocess=True):
        doc = init_doc(doc)
        root_obj = FastDesignProvider()
        root_obj.append(self)
        root = root_obj.get_root(doc, comm, False)
        if preprocess:
            root_obj._preprocess(root)
        ref = root.ref['id']
        state._views[ref] = (self, root, doc, comm)
        return root


class FastNumberInput(FastComponent):

    appearance = param.Selector(default='outline', objects=['outline', 'filled'])

    autofocus = param.Boolean(default=False)

    placeholder = param.String(default='Type here')

    step = param.Number(default=0)

    start = param.Number(default=0)

    end = param.Number(default=1)

    value = param.Number(default=0)

    _child_config = {'name': 'template'}

    _template= '<fast-number-field id="fast-number" autofocus="${autofocus}" placeholder="${placeholder}" step="${step}" value="${value}" min="${start}" max="${end}" appearance="${appearance}"></fast-number-field>' # noqa

    _dom_events = {'fast-number': ['change']}


class FastSlider(FastComponent):

    height = param.Integer(default=50)

    step = param.Number(default=0.1)

    start = param.Number(default=0)

    end = param.Number(default=1)

    value = param.Number(default=0)

    _child_config = {'name': 'template'}

    _template = '<fast-slider id="fast-slider" value="${value}" min="${start}" max="${end}" # noqa step="${step}"></fast-slider>'

class FastSelect(FastComponent, SelectBase):

    options = param.ClassSelector(default=[], class_=(dict, list))

    value = param.Parameter()

    _template = """
    <fast-select value="${value}" id="fast-select">
    {% for option in options %}
      <fast-option value="{{ option }}">${option}</fast-option>
    {% endfor %}
    </fast-select>
    """

    def _process_property_change(self, msg):
        msg = super()._process_property_change(msg)
        if 'value' in msg:
            msg['value'] = self._items[msg['value']]
        return msg


class FastButton(FastComponent):

    clicks = param.Integer(default=0, bounds=(0, None))

    value = param.Event()

    _child_config = {'name': 'template'}

    _template = '<fast-button onclick="${_on_click}" appearance="outline" id="fast-button">${name}</fast-button>'

    def _on_click(self, event):
        self.clicks += 1
        self.param.trigger('value')


class FastCheckbox(FastComponent):

    value = param.Boolean(default=True)

    _child_config = {'name': 'template'}

    _template = '<fast-checkbox id="fast-checkbox" value="${value}">${name}</fast-checkbox>'


class FastRadioGroup(FastComponent, SelectBase):

    options = param.ClassSelector(default=[], class_=(dict, list))

    value = param.Parameter()

    _child_config = {'options': 'literal'}

    _template = """
    <fast-radio-group value="${value}" id="fast-radio-group">
      <label style="color: var(--neutral-foreground-rest);" slot="label" slot="label">${title}</label>
      {% for option in options %}
      <fast-radio id="fast-radio-{{ loop.index0 }}">${options[{{ loop.index0 }}]}</fast-radio>
      {% endfor %}
      </fast-radio-group>
    """

    def _on_change(self, event):
        self.value = self._items[event.data['value']]

    def _process_children(self, doc, root, model, comm, children):
        children['fast-options'] = self.labels
        return children


class FastToggle(FastComponent):

    value = param.Boolean(default=False)

    _template = '<fast-switch id="fast-toggle" checked="${value}">${name}</fast-switch>'


class FastTextArea(FastComponent):

    value = param.String()

    _template = '<fast-text-area id="fast-text-area" value="${value}"></fast-text-area>'


class FastTextInput(FastComponent):

    placeholder = param.String()

    value = param.String()

    _template = '<fast-text-field id="fast-text-field" placeholder="${placeholder}" value="${value}"></fast-text-field>'


class FastProgress(FastComponent):

    _template = '<fast-progress></fast-progress>'


class FastProgressRing(FastComponent):

    _template = '<fast-progress-ring></fast-progress-ring>'



class FastDivider(FastComponent):

    _template = '<fast-divider></fast-divider>'


class FastCard(FastComponent, ListLike):

    _template = '<fast-card id="fast-card">${objects}</fast-card>'

    def __init__(self, *objects, **params):
        super().__init__(objects=list(objects), **params)


class FastDialog(FastComponent, ListLike):

    hidden = param.Boolean(default=False)

    _template = """
    <fast-dialog id="fast-dialog" hidden=${hidden} style="z-index: 100; --dialog-width: 80%; --dialog-height: 80%">
      <fast-button id="close-button" onclick="${_close}" style="float: right">X</fast-button>
      ${objects}
    </fast-dialog>
    """

    def __init__(self, *objects, **params):
        super().__init__(objects=list(objects), **params)

    def _close(self, event):
        self.hidden = True

class FastTabs(FastComponent, NamedListLike):

    active = param.Integer(default=0, bounds=(0, None), doc="""
        Index of the currently displayed objects.""")

    _activeid = param.String(default='')

    _template = """
    <fast-tabs id="fast-tabs" activeid="${_activeid}">
    {% for obj_name in objects_names %}
      <fast-tab slot="tab" slot="tab">{{ obj_name }}</fast-tab>
    {% endfor %}
    {% for object in objects %}
      <fast-tab-panel id="fast-tab-panel" slot="tabpanel" slot="tabpanel">${object}</fast-tab-panel>
    {% endfor %}
    </fast-tabs>
    """

    @property
    def _child_names(self):
        return {'objects': self._names}

    def __init__(self, *objects, **params):
        NamedListLike.__init__(self, *objects, **params)
        FastComponent.__init__(self, objects=self.objects, **params)

    @param.depends('_activeid', watch=True)
    def _update_active(self):
        self.active = int(self._activeid.split('-')[-1])-1


class FastAccordion(FastComponent, NamedListLike):

    active = param.List()

    _scripts = {'on_change': """
      const active = []
      for (let i=0; i < fast_accordion.children.length; i++) {
        if (fast_accordion.children[i].expanded)
          active.push(i)
      }
      data.active = active;
      """
    }

    _template = """
    <fast-accordion id="fast-accordion" onchange="${script('on_change')}">
      {% for object in objects %}
      <fast-accordion-item slot="item" slot="item" {% if loop.index0 in active %} expanded {% endif %}>
          <div slot="heading" slot="heading">{{ objects_names[loop.index0] }}</div>
          <svg style="stroke: #e62f63;" width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg" slot="collapsed-icon" slot="collapsed-icon">
            <path d="M15.2222 1H2.77778C1.79594 1 1 1.79594 1 2.77778V15.2222C1 16.2041 1.79594 17 2.77778 17H15.2222C16.2041 17 17 16.2041 17 15.2222V2.77778C17 1.79594 16.2041 1 15.2222 1Z" stroke-linecap="round" stroke-linejoin="round"></path>
            <path d="M9 5.44446V12.5556" stroke-linecap="round" stroke-linejoin="round"></path>
            <path d="M5.44446 9H12.5556" stroke-linecap="round" stroke-linejoin="round"></path>
          </svg>
          <svg style="stroke: #e62f63;" width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg" slot="expanded-icon" slot="expanded-icon">
            <path d="M15.2222 1H2.77778C1.79594 1 1 1.79594 1 2.77778V15.2222C1 16.2041 1.79594 17 2.77778 17H15.2222C16.2041 17 17 16.2041 17 15.2222V2.77778C17 1.79594 16.2041 1 15.2222 1Z" stroke-linecap="round" stroke-linejoin="round"></path>
            <path d="M5.44446 9H12.5556" stroke-linecap="round" stroke-linejoin="round"></path>
          </svg>
          <div id="accordion-content">${object}</div>
      </fast-accordion-item>
      {% endfor %}
    </fast-accordion>
    """ # noqa

    def __init__(self, *objects, **params):
        NamedListLike.__init__(self, *objects, **params)
        FastComponent.__init__(self, objects=self.objects, **params)

    @property
    def _child_names(self):
        return {'objects': self._names}
