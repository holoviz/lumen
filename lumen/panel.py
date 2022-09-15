import param

from panel import panel
from panel.reactive import ReactiveHTML
from panel.widgets import FileDownload

from .util import catch_and_notify

try:
    # Backward compatibility for panel 0.12.6
    import bokeh.core.properties as bp

    from panel.links import PARAM_MAPPING

    # The Bokeh Color property has `_default_help` set which causes
    # an error to be raise when Nullable is called on it. This converter
    # overrides the Bokeh _help to set it to None and avoid the error.
    # See https://github.com/holoviz/panel/issues/3058
    def color_param_to_ppt(p, kwargs):
        ppt = bp.Color(**kwargs)
        ppt._help = None
        return ppt

    PARAM_MAPPING[param.Color] = color_param_to_ppt
except Exception:
    pass


class DownloadButton(ReactiveHTML):

    callback = param.Callable(precedence=-1)

    color = param.Color(default='grey', allow_None=True)

    data = param.String()

    filename = param.String()

    hide = param.Boolean(default=False)

    size = param.Integer(default=20)

    _template = """
    <style>
    .download-button {
      position: absolute;
      top: 0px;
      right: 0px;
      width: {{ size }}px;
      height: {{ size }}px;
      z-index: 10000;
      opacity: {% if hide %}0{% else %}1{% endif %};
      transition-delay: 0.5s;
      transition: 0.5s;
      cursor: pointer;
      font-size: {{ size }}px;
      {% if color %}color: {{ color }};{% endif %}
    }

    .download-button:hover {
      transition: 0.5s;
      opacity: 1;
    }

    .download-button:focus {
      opacity: 1;
    }
    </style>
    <span id="download-button" onclick="${_on_click}" class="download-button">
      <i class="fas fa-download"></i>
    </span>
    """

    _scripts = {
        'data': """
          if (data.data == null || !data.data.length)
            return
          const byteString = atob(data.data.split(',')[1]);

          // separate out the mime component
          const mimeString = data.data.split(',')[0].split(':')[1].split(';')[0];

          // Reset data

          data.data = '';

          // write the bytes of the string to an ArrayBuffer
          const ab = new ArrayBuffer(byteString.length);
          const ia = new Uint8Array(ab);
          for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
          }

          // write the ArrayBuffer to a blob, and you're done
          var bb = new Blob([ab], { type: mimeString });

          var link = document.createElement('a');

          link.href = URL.createObjectURL(bb)
          link.download = data.filename
          link.click()
        """
    }

    def __init__(self, object=None, **params):
        params['sizing_mode'] = 'stretch_width'
        if object is not None:
            object = panel(object)
            params['object'] = object
        super().__init__(**params)

    @catch_and_notify("Download failed")
    def _on_click(self, event=None):
        file_input = FileDownload(callback=self.callback, filename=self.filename)
        file_input._transfer()
        self.data = file_input.data


class IconButton(ReactiveHTML):

    disabled = param.Boolean(default=False)

    color = param.Color(default=None)

    icon = param.String(default=None, doc="""
      The FontAwesome icon to use.""")

    size = param.Integer(default=12, bounds=(0, None))

    _template = """
      <i id="icon-button" class="fas ${icon}" style="font-size: ${size}px; color: ${color}" onclick=${script('clicked')}></i>
    """

    _scripts = {
        'clicked': """
          if (data.disabled)
            return
          data.disabled = true;
          view._send_event('button', 'click', state.event)
        """,
        'disabled': """
          icon_button.style.cursor = data.disabled ? "not-allowed": "inherit";
        """
    }

    _event = 'dom_event'

    def __init__(self, **params):
        super().__init__(**params)
        self._callbacks = []
        self._disabled_watcher = None

    def _enable_button(self, event):
        if self._disabled_watcher:
            self.param.unwatch(self._disabled_watcher)
        self.disabled = False

    @param.depends('size', watch=True, on_init=True)
    def _update_height(self):
        self.height = self.size

    def on_click(self, callback):
        self._callbacks.append(callback)

    def js_on_click(self, args={}, code=""):
        from panel.links import Callback
        return Callback(self, code={'evegnt:'+self._event: code}, args=args)

    def _button_click(self, event=None):
        for cb in self._callbacks:
            cb(event)
        self._disabled_watcher = self.param.watch(self._enable_button, ['disabled'])
