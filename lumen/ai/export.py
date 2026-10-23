import ast
import base64
import inspect
import os
import re

from collections.abc import Callable
from importlib.machinery import PathFinder
from io import BytesIO
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
from .tools.base import FunctionCall


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

def format_output(output: Any):
    ext = None
    code = []

    if hasattr(output, 'component') and hasattr(output.component, 'to_spec'):
        component = output.component
    elif hasattr(output, 'to_spec'):
        component = output
    else:
        return None, None

    with config.param.update(serializer='csv'):
        # replace |2- |3- |4-... etc with | for a cleaner look
        try:
            spec = re.sub(r'(\|[-\d]*)', '|', yaml.dump(component.to_spec(), sort_keys=False))
        except (TypeError, yaml.YAMLError):
            return None, None
    read_code = [
        f'yaml_spec = """\n{spec}"""',
        'spec = yaml.safe_load(yaml_spec)',
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

def _assigned_names(source: str) -> set[str]:
    """Names a code cell defines at its top level."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split('.')[0] for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
    return names


def _is_literal(value: Any) -> bool:
    """Whether ``repr(value)`` reads back as the same value."""
    try:
        return ast.literal_eval(repr(value)) == value
    except Exception:
        return False


def _function_definition(function: Callable) -> str | None:
    """
    The code that makes ``function`` available in the notebook: an import
    when its module can be imported, otherwise its source.
    """
    name = function.__name__
    module = getattr(function, '__module__', None) or ''
    if module != '__main__' and getattr(function, '__qualname__', name) == name:
        if PathFinder.find_spec(module.split('.')[0]) is not None:
            return f"from {module} import {name}"
    try:
        source = dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return None
    # Drop decorators, the notebook may not have them
    node = ast.parse(source).body[0]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
        source = '\n'.join(source.splitlines()[node.lineno - 1:])
    imports = _file_imports(function, source)
    return '\n'.join([*imports, '', source.rstrip()] if imports else [source.rstrip()])


def _file_imports(function: Callable, source: str) -> list[str]:
    """
    The top level imports of the file defining ``function`` that its
    source uses, so the inlined function can run on its own.
    """
    try:
        with open(inspect.getsourcefile(function), encoding='utf-8') as f:
            text = f.read()
        tree = ast.parse(text)
    except (OSError, TypeError, SyntaxError):
        return []
    used = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
    imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound = {(alias.asname or alias.name).split('.')[0] for alias in node.names}
            if bound & used:
                imports.append(ast.get_source_segment(text, node))
    return imports


def format_function_call(call: FunctionCall, cells: list) -> Any | None:
    """
    A code cell that repeats a FunctionTool call, after the given cells.

    Arguments that can't be written as literals, such as those filled in
    from Lumen's context, are passed by name, with a note when no earlier
    cell defines them.
    """
    function = call.function
    name = getattr(function, '__name__', '<lambda>')
    if name == '<lambda>':
        return None
    defined = set()
    for cell in cells:
        if cell.cell_type == 'code':
            defined |= _assigned_names(cell.source)
    lines = []
    if name not in defined:
        definition = _function_definition(function)
        if definition is None:
            return None
        lines += [definition, '']
    arguments, missing = [], []
    for key, value in call.arguments.items():
        if _is_literal(value):
            arguments.append(f"{key}={value!r}")
            continue
        arguments.append(f"{key}={key}")
        if key not in defined:
            missing.append(key)
    for key in call.context:
        arguments.append(f"{key}={key}")
        if key not in defined:
            missing.append(key)
    if missing:
        names = ', '.join(missing)
        lines.append(f"# Lumen passed in {names}; define {'it' if len(missing) == 1 else 'them'} before running this cell")
    expression = f"{name}({', '.join(arguments)})"
    if inspect.iscoroutinefunction(function):
        expression = f"await {expression}"
    if call.provides and not call.unpack:
        lines.append(f"{call.provides[0]} = {expression}")
    elif call.provides:
        lines.append(f"result = {expression}")
        lines += [f"{key} = result[{key!r}]" for key in call.provides]
    else:
        lines.append(expression)
    return nbformat.v4.new_code_cell(source='\n'.join(lines))


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
        elif isinstance(out, FunctionCall):
            cell = format_function_call(out, cells)
            if cell is not None:
                cells.append(cell)
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
