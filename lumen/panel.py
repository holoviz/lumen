import param

from panel import panel
from panel.reactive import ReactiveHTML
from panel.widgets import FileDownload


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

    def _on_click(self, event=None):
        file_input = FileDownload(callback=self.callback, filename=self.filename)
        file_input._transfer()
        self.data = file_input.data
