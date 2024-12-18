import datetime as dt
import json

from textwrap import dedent

import nbformat

from panel import Column
from panel.chat import ChatMessage, ChatStep

from lumen.ai.views import LumenOutput
from lumen.config import config
from lumen.pipeline import Pipeline
from lumen.views import View


def make_md_cell(text: str):
    return nbformat.v4.new_markdown_cell(source=text)

def make_preamble(preamble: str):
    now = dt.datetime.now()
    header = make_md_cell(f'# Lumen.ai - Chat Logs {now}')
    source = (preamble + dedent(
        """
        import lumen as lm
        import panel as pn

        pn.extension('tabulator')
        """
    )).strip()
    imports = nbformat.v4.new_code_cell(source=source)
    return [header, imports]

def format_markdown(msg: ChatMessage):
    if msg.avatar.startswith('https://'):
        avatar = f'<img src="{msg.avatar}" width=45 height=45></img>'
    else:
        avatar = f'<span>{msg.avatar}</span>'
    header = f'<div style="display: flex; flex-direction: row; font-weight: bold; font-size: 2em;">{avatar}<span style="margin-left: 0.5em">{msg.user}</span></div>'
    prefix = '\n' if msg.user == 'User' else '\n> '
    content = prefix.join(msg.serialize().split('\n'))
    return [nbformat.v4.new_markdown_cell(source=f'{header}\n{prefix}{content}')]

def format_output(msg: ChatMessage):
    output = msg.object
    code = []
    with config.param.update(serializer='csv'):
        spec = json.dumps(output.component.to_spec(), indent=2).replace('true', 'True').replace('false', 'False')
    if isinstance(output.component, Pipeline):
        code.extend([
            f'pipeline = lm.Pipeline.from_spec({spec})',
            'pipeline'
        ])
    elif isinstance(output.component, View):
        code.extend([
            f'view = lm.View.from_spec({spec})',
            'view'
        ])
    return [nbformat.v4.new_code_cell(source='\n'.join(code))]

def render_cells(messages: list[ChatMessage]):
    cells = []
    for msg in messages:
        if msg.user == 'Help':
            continue
        elif isinstance(msg.object, str):
            cells += format_markdown(msg)
        elif isinstance(msg.object, LumenOutput):
            cells += format_output(msg)
        elif isinstance(msg.object, Column):
            for obj in msg.object:
                if isinstance(obj, ChatStep):
                    continue
                cells += format_output(obj)
    return cells

def write_notebook(cells):
    nb = nbformat.v4.new_notebook(cells=cells)
    return nbformat.v4.writes(nb)

def export_notebook(messages: list[ChatMessage], preamble: str = ""):
    cells = make_preamble(preamble) + render_cells(messages)
    return write_notebook(cells)
