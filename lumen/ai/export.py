import base64
import json
import os
import re
import shutil
import subprocess
import sys

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import Any

import nbformat
import yaml

from panel.pane import Markdown
from panel.pane.image import ImageBase
from panel.viewable import Viewable

from ..config import config
from ..pipeline import Pipeline
from ..util import as_pandas
from ..views import View
from .editors import LumenEditor


def make_md_cell(text: str):
    return nbformat.v4.new_markdown_cell(source=text)

def make_preamble(preamble: str, extensions: list[str] | None = None, title: str | None = None):
    if extensions is None:
        extensions = []
    if 'tabulator' not in extensions:
        extensions = ['tabulator'] + extensions
    exts = ', '.join([repr(ext) for ext in extensions])
    source = (preamble + dedent(
        f"""
        import yaml

        import lumen as lm
        import panel as pn

        pn.extension({exts})
        """
    )).strip()
    imports = nbformat.v4.new_code_cell(source=source)
    if title:
        return [make_md_cell(f'# {title}'), imports]
    return [imports]

def serialize_avatar(avatar: str | bytes | BytesIO | ImageBase, size: int = 45) -> str:
    """
    Process different types of avatar inputs into HTML img tag or text span.

    Args:
        avatar: Avatar source (URL, BytesIO, PIL Image, or text)
        size: Desired width and height of the avatar in pixels

    Returns:
        HTML string representing the avatar
    """
    if isinstance(avatar, ImageBase):
        avatar = avatar.object # type: ignore
    if isinstance(avatar, BytesIO):
        avatar = avatar.getvalue()

    if isinstance(avatar, str):
        if avatar.startswith(('http://', 'https://')):
            return f'<img src="{avatar}" width={size} height={size} alt="avatar" style="border-radius: 50%;"></img>'
        elif os.path.exists(avatar):
            avatar_path = os.path.abspath(avatar)
            return f'<img src="{avatar_path}" width={size} height={size} alt="avatar" style="border-radius: 50%;"></img>'
        else:
            return f'<span class="text-avatar" style="width: {size}px; height: {size}px; display: inline-flex; align-items: center; justify-content: center; background-color: #f0f0f0; border-radius: 50%;">{avatar[:2].upper()}</span>'
    elif isinstance(avatar, bytes):
        img_data = base64.b64encode(avatar).decode()
        return f'<img src="data:image/png;base64,{img_data}" width={size} height={size} alt="avatar" style="border-radius: 50%;"></img>'
    else:
        return f'<span class="text-avatar" style="width: {size}px; height: {size}px; display: inline-flex; align-items: center; justify-content: center; background-color: #f0f0f0; border-radius: 50%;">{str(avatar)[:2].upper()}</span>'

def format_markdown(md: Markdown):
    return [nbformat.v4.new_markdown_cell(source=md.object)]

def format_output(output: Any, spec: dict | None = None, load_spec: str = "yaml.safe_load(yaml_spec)"):
    """
    A notebook cell that rebuilds an output's Pipeline or View from its spec.

    ``spec`` replaces the component's own spec, and ``load_spec`` is the
    expression the cell uses to turn ``yaml_spec`` back into a dict.
    """
    ext = None
    code = []

    if hasattr(output, 'component') and hasattr(output.component, 'to_spec'):
        component = output.component
    elif hasattr(output, 'to_spec'):
        component = output
    else:
        return None, None

    with config.param.update(serializer='csv'):
        try:
            spec = yaml.dump(component.to_spec() if spec is None else spec, sort_keys=False)
        except (TypeError, yaml.YAMLError):
            return None, None
    read_code = [
        # A repr, not a triple-quoted string, so no value in the data or SQL
        # can end the string and run as code
        f'yaml_spec = {spec!r}',
        f'spec = {load_spec}',
    ]
    if isinstance(component, Pipeline):
        code.extend([
            *read_code,
            'pipeline = lm.Pipeline.from_spec(spec)',
            'pipeline'
        ])
    elif isinstance(component, View):
        ext = getattr(component, '_extension', None)
        code.extend([
            *read_code,
            'view = lm.View.from_spec(spec)',
            'view'
        ])
    return nbformat.v4.new_code_cell(source='\n'.join(code)), ext

def render_cells(outputs: list[Viewable]) -> tuple[Any, list[str]]:
    cells, extensions = [], []
    for out in outputs:
        if isinstance(out, Markdown):
            cells += format_markdown(out)
        elif isinstance(out, LumenEditor):
            cell, ext = format_output(out)
            if cell is not None:
                cells.append(cell)
            if ext and ext not in extensions:
                extensions.append(ext)
    return cells, extensions

def write_notebook(cells):
    nb = nbformat.v4.new_notebook(cells=cells)
    return nbformat.v4.writes(nb)

def export_notebook(outputs: list[Viewable], preamble: str = ""):
    cells, extensions = render_cells(outputs)
    cells = make_preamble(preamble, extensions=extensions) + cells
    return write_notebook(cells)


def _docx_add_runs(paragraph, text: str):
    """Add text to a paragraph, rendering ``**bold**`` spans as bold runs."""
    for i, part in enumerate(re.split(r'\*\*(.+?)\*\*', text)):
        if not part:
            continue
        run = paragraph.add_run(part)
        if i % 2 == 1:
            run.bold = True


def docx_add_markdown(doc, text: str):
    """Render basic markdown (headings, bullets, bold) into a docx document."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            level = len(stripped) - len(stripped.lstrip('#'))
            doc.add_heading(stripped.lstrip('# ').strip(), level=min(level, 4))
        elif stripped.startswith(('- ', '* ')):
            _docx_add_runs(doc.add_paragraph(style='List Bullet'), stripped[2:])
        else:
            _docx_add_runs(doc.add_paragraph(), stripped)


def docx_add_table(doc, df, max_rows: int = 50):
    """Render a DataFrame as a native Word table (header + capped rows)."""
    df = as_pandas(df)  # iterrows is pandas only
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = 'Table Grid'
    for cell, col in zip(table.rows[0].cells, df.columns, strict=False):
        cell.text = str(col)
    for _, record in df.head(max_rows).iterrows():
        cells = table.add_row().cells
        for cell, value in zip(cells, record, strict=False):
            cell.text = '' if value is None else str(value)
    if len(df) > max_rows:
        doc.add_paragraph(f'... {len(df) - max_rows} more rows')


def docx_add_chart(doc, editor) -> bool:
    """Embed a chart editor as a PNG image; return True if one was added."""
    from docx.shared import Inches

    if 'png' not in editor.export_formats:
        return False
    try:
        png = editor.export('png')
    except Exception:
        # Best-effort: a chart we cannot render should not abort the document.
        return False
    png.seek(0)
    doc.add_picture(png, width=Inches(6))
    return True


#: What a Lumen dashboard imports, installed in the browser. Lumen itself
#: goes in as files, since its own requirements include packages with no
#: browser build, such as tiktoken.
PYODIDE_REQUIREMENTS = [
    "chardet", "duckdb", "holoviews", "hvplot", "jinja2", "markdown-it-py", "narwhals", "panel-material-ui",
    "pillow", "platformdirs", "pyarrow", "pyyaml", "requests", "sqlglot", "tqdm",
]

_PYODIDE_APP = """\
import panel_material_ui  # noqa: F401  its components must be on the page before Lumen draws any
import yaml

from lumen.dashboard import Dashboard

Dashboard(yaml.safe_load({spec!r})).servable()
"""


_CONVERT = "import json, sys; from panel.io.convert import convert_apps; convert_apps([sys.argv[1]], **json.loads(sys.argv[2]))"


def to_pyodide_html(spec: str, title: str) -> str:
    """
    A Lumen dashboard spec as one HTML page that runs it in the browser
    with Pyodide: no Python server, only the Pyodide and package downloads
    on first load. Lumen's own files are inside the page.
    """
    with TemporaryDirectory() as tmp:
        build = Path(tmp)
        # The dashboard needs neither the AI modules nor the tests
        shutil.copytree(Path(__file__).parents[1], build / "lumen", ignore=shutil.ignore_patterns("ai", "tests", "__pycache__"))
        (build / "app.py").write_text(_PYODIDE_APP.format(spec=spec), encoding="utf-8")
        files = [str(path) for path in (build / "lumen").rglob("*") if path.is_file()]
        options = dict(
            dest_path=str(build / "out"), runtime="pyodide", title=title, requirements=PYODIDE_REQUIREMENTS,
            resources=files, build_pwa=False, build_index=False,
        )
        # Its own process: converting runs the app, which would change this
        # session's Panel config, and starts a process pool that a script
        # read from stdin cannot spawn from
        subprocess.run(
            [sys.executable, "-c", _CONVERT, str(build / "app.py"), json.dumps(options)],
            check=True, capture_output=True, cwd=build,
        )
        html = (build / "out" / "app.html").read_text(encoding="utf-8")
        archive = base64.b64encode((build / "out" / "app.resources.zip").read_bytes()).decode()
    # The page fetches its files as a zip; inlined, the page is the whole app
    reference = "'app.resources.zip'"
    if html.count(reference) != 1:
        raise RuntimeError("Panel's converted page no longer loads its files as Lumen expects; save the spec as .yaml instead.")
    return html.replace(reference, f"'data:application/zip;base64,{archive}'")
