import ast
import base64
import inspect
import keyword
import os
import re
import sys

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

# Argument names whose values are left out of exported code, since they
# may hold credentials
_SECRET_NAME = re.compile(
    r'(?:^|_)(?:pass(?:word|wd)?|pwd|secret|credentials?|auth)(?:_|$)'
    r'|(?:api|access|private|secret)_?key$|token$',
    re.IGNORECASE,
)

def _literal_repr(value: Any) -> str | None:
    """Return repr(value) if it evaluates back to a Python literal."""
    try:
        text = repr(value)
        ast.literal_eval(text)
    except Exception:
        return None
    return text

def assigned_names(source: str) -> set[str]:
    """Return the top level names a code cell assigns or imports."""
    try:
        tree = compile(
            source, '<cell>', 'exec',
            flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
    except SyntaxError:
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(
                    sub.id for sub in ast.walk(target)
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
                )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split('.')[0] for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names

def _function_name(function) -> str | None:
    """The name ``function`` can be called by, if it has a usable one."""
    name = getattr(function, '__name__', None)
    if isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name):
        return name
    return None

def _function_import(function, name: str) -> str | None:
    """An import statement for ``function``, if it can be imported by name."""
    module = getattr(function, '__module__', None)
    if module in (None, '__main__') or getattr(function, '__qualname__', name) != name:
        return None
    if getattr(sys.modules.get(module), name, None) is not function:
        return None
    return f'from {module} import {name}'

def format_function_call(call: FunctionCall, defined: set[str] | None = None):
    """
    Render a FunctionTool call as a code cell that repeats the call,
    assigning the returned values to the context keys it provides.

    Arguments
    ---------
    call: FunctionCall
        The recorded call.
    defined: set[str] | None
        Names defined by earlier cells of the notebook. Arguments that are
        not Python literals are passed by name, and any name not defined
        earlier is reported by a placeholder that stops the cell with a
        clear error instead of a bare NameError.
    """
    defined = set(defined or ())
    function = call.function
    name = _function_name(function)
    if name is None:
        # e.g. a functools.partial, which has no name to call it by
        return nbformat.v4.new_code_cell(source=(
            f'# The call to {function!r} cannot be repeated here, since it has no name to call it by.'
        ))

    lines, notes, missing = [], [], []
    import_line = _function_import(function, name)
    if import_line:
        lines.append(import_line)
    elif name not in defined:
        # The source is not copied, since that would drop its imports and closures
        missing.append(name)
        notes.append(f'# {name} cannot be imported, so it must be defined before this cell.')

    args = []
    for key, value in call.arguments.items():
        literal = _literal_repr(value)
        if literal is not None and not (isinstance(value, str) and _SECRET_NAME.search(key)):
            args.append(f'{key}={literal}')
            continue
        args.append(f'{key}={key}')
        if key in defined:
            continue
        missing.append(key)
        if literal is None:
            notes.append(f'# {key} came from the Lumen context and is not defined by an earlier cell.')
        else:
            notes.append(f'# {key} was left out because it may hold a secret.')

    if missing:
        names = ', '.join(missing)
        lines.extend(notes)
        lines.append(f'raise NameError("Define {names} before running this cell")')
    if lines:
        lines.append('')

    is_async = inspect.iscoroutinefunction(function)
    call_str = f"{'await ' if is_async else ''}{name}({', '.join(args)})"
    if len(call_str) > 80 and args:
        arg_lines = ''.join(f'\n    {arg},' for arg in args)
        call_str = f"{'await ' if is_async else ''}{name}({arg_lines}\n)"

    if not call.provides:
        lines.append(call_str)
    elif not call.unpack:
        lines.append(f'{call.provides[0]} = {call_str}')
    else:
        lines.append(f'result = {call_str}')
        lines.extend(f'{key} = result[{key!r}]' for key in call.provides)
    return nbformat.v4.new_code_cell(source='\n'.join(lines))

def render_cells(outputs: list[Viewable]) -> tuple[Any, list[str]]:
    cells, extensions, defined = [], [], set()
    for out in outputs:
        if isinstance(out, FunctionCall):
            cells.append(format_function_call(out, defined))
        elif isinstance(out, Markdown):
            cells += format_markdown(out)
        elif isinstance(out, LumenEditor):
            cell, ext = format_output(out)
            if cell is not None:
                cells.append(cell)
            if ext and ext not in extensions:
                extensions.append(ext)
        if cells and cells[-1].cell_type == 'code':
            defined |= assigned_names(cells[-1].source)
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
