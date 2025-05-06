import base64
import datetime as dt
import os
import re

from io import BytesIO
from textwrap import dedent
from typing import Any

import nbformat
import yaml

from panel import Column
from panel.chat import ChatMessage, ChatStep
from panel.pane.image import ImageBase

from ..config import config
from ..pipeline import Pipeline
from ..views import View
from .views import LumenOutput


def make_md_cell(text: str):
    return nbformat.v4.new_markdown_cell(source=text)

def make_preamble(preamble: str, extensions: list[str]):
    now = dt.datetime.now()
    header = make_md_cell(f'# Lumen.ai - Chat Logs {now}')
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
    return [header, imports]


def serialize_avatar(avatar: str | BytesIO, size: int = 45) -> str:
    """
    Process different types of avatar inputs into HTML img tag or text span.

    Args:
        avatar: Avatar source (URL, BytesIO, PIL Image, or text)
        size: Desired width and height of the avatar in pixels

    Returns:
        HTML string representing the avatar
    """
    if isinstance(avatar, ImageBase):
        avatar = avatar.object
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


def format_markdown(msg: ChatMessage):
    avatar_html = serialize_avatar(msg.avatar)
    header = f'<div style="display: flex; flex-direction: row; font-weight: bold; font-size: 2em;">{avatar_html}<span style="margin-left: 0.5em">{msg.user}</span></div>'
    prefix = '\n' if msg.user == 'User' else '\n> '
    content = prefix.join(msg.serialize().split('\n'))
    return [nbformat.v4.new_markdown_cell(source=f'{header}\n{prefix}{content}')]


def format_output(msg: ChatMessage):
    output = msg.object
    ext = None
    code = []
    with config.param.update(serializer='csv'):
        # replace |2- |3- |4-... etc with | for a cleaner look
        spec = re.sub(r'(\|[-\d]*)', '|', yaml.dump(output.component.to_spec(), sort_keys=False))
    read_code = [
        f'yaml_spec = """\n{spec}"""',
        'spec = yaml.safe_load(yaml_spec)',
    ]
    if isinstance(output.component, Pipeline):
        code.extend([
            *read_code,
            'pipeline = lm.Pipeline.from_spec(spec)',
            'pipeline'
        ])
    elif isinstance(output.component, View):
        ext = output.component._extension
        code.extend([
            *read_code,
            'view = lm.View.from_spec(spec)',
            'view'
        ])
    return [nbformat.v4.new_code_cell(source='\n'.join(code))], ext

def render_cells(messages: list[ChatMessage]) -> tuple[Any, list[str]]:
    cells, extensions = [], []
    for msg in messages:
        if msg.user in ("Help", " "):
            continue
        elif isinstance(msg.object, str):
            cells += format_markdown(msg)
        elif isinstance(msg.object, LumenOutput):
            out, ext = format_output(msg)
            cells += out
            if ext and ext not in extensions:
                extensions.append(ext)
        elif isinstance(msg.object, Column):
            for obj in msg.object:
                if isinstance(obj, ChatStep):
                    continue
                cells += format_output(obj)
    return cells, extensions

def write_notebook(cells):
    nb = nbformat.v4.new_notebook(cells=cells)
    return nbformat.v4.writes(nb)

def export_notebook(messages: list[ChatMessage], preamble: str = ""):
    cells, extensions = render_cells(messages)
    cells = make_preamble(preamble, extensions=extensions) + cells
    return write_notebook(cells)
