import datetime as dt
import json

import nbformat

from lumen.ai.views import LumenOutput
from lumen.pipeline import Pipeline
from lumen.views import View


def make_preamble():
    now = dt.datetime.now()
    header = nbformat.v4.new_markdown_cell(source=f'# Lumen.ai - Chat Logs {now}')
    imports = nbformat.v4.new_code_cell(source="""\
import lumen as lm
import panel as pn

pn.extension('tabulator')""")
    return [header, imports]

def format_markdown(msg):
    if msg.avatar.startswith('https://'):
        avatar = f'<img src="{msg.avatar}" width=45 height=45></img>'
    else:
        avatar = f'<span>{msg.avatar}</span>'
    header = f'<div style="display: flex; flex-direction: row; font-weight: bold; font-size: 2em;">{avatar}<span style="margin-left: 0.5em">{msg.user}</span></div>'
    prefix = '\n' if msg.user == 'User' else '\n> '
    content = prefix.join(msg.serialize().split('\n'))
    return [nbformat.v4.new_markdown_cell(source=f'{header}\n{prefix}{content}')]

def format_output(msg):
    output = msg.object
    code = []
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

def export_notebook(assistant):
    cells = make_preamble()
    for msg in assistant.interface.objects:
        if msg.user == 'Help':
            continue
        elif isinstance(msg.object, str):
            cells += format_markdown(msg)
        elif isinstance(msg.object, LumenOutput):
            cells += format_output(msg)

    nb = nbformat.v4.new_notebook(cells=cells)
    return nbformat.v4.writes(nb)
