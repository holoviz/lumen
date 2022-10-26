import pathlib
import textwrap

import param

param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

from param import concrete_descendents
from rst_to_myst import rst_to_myst

import lumen
import lumen.sources.intake
import lumen.sources.intake_sql

from lumen.dashboard import Auth, Config, Defaults
from lumen.filters import Filter
from lumen.pipeline import Pipeline
from lumen.sources import Source
from lumen.transforms import Transform
from lumen.variables import Variable
from lumen.views import View

bases = [Auth, Config, Defaults, Pipeline]
multi_bases = [Filter, Source, Transform, Variable, View]

BASE_PATH = pathlib.Path(lumen.__file__).parent.parent
REFERENCE_PATH = BASE_PATH / 'docs' / 'reference'
EXAMPLES_PATH = BASE_PATH / 'examples' / 'reference'

def escape_table_docs(doc):
    doc = textwrap.dedent(doc or '').strip().replace('|', '&#124;')
    fenced = False
    lines = []
    for line in doc.split('\n'):
        if '```' in line:
            fenced = not fenced
        if not fenced:
            line += '<br>'
        lines.append(line)
    return ''.join(lines)

def generate_param_docs(base, component):
    if base is component:
        params = list(base.param)
    else:
        params = sorted([p for p in component.param if p not in base.param])
    rows =  [
        '| Name | Type | Constraint | Default | Documentation |',
        '| ---- | ---- | ---------- | ------- | ------------- |'
    ]
    for pname in params:
        if pname.startswith('_'):
            continue
        pobj = component.param[pname]
        ptype = type(pobj)
        pdoc = escape_table_docs(pobj.doc)
        default = repr(pobj.default)
        if isinstance(pobj, param.ClassSelector):
            if isinstance(pobj.class_, tuple):
                constraint = ' | '.join([cls.__name__ for cls in pobj.class_])
            else:
                constraint = pobj.class_.__name__
        elif hasattr(pobj, 'bounds'):
            constraint = repr(pobj.bounds)
        elif hasattr(pobj, 'get_range'):
            constraint = ' | '.join(pobj.get_range())
        else:
            constraint = ''
        rows.append(f'| **`{pname}`** | {ptype.__name__} | {constraint} | {default} | {pdoc} |')
    return '\n'.join(rows) if len(rows) > 2 else ''

def generate_page(base, component):
    title = f'# {component.__name__}'
    if component.__doc__:
        try:
            lines = []
            for line in rst_to_myst(component.__doc__).text.split('\n'):
                if line.startswith('>'):
                    line = line[1:].lstrip()
                lines.append(line)
            doc = '\n'.join(lines)
        except Exception as e:
            print(f'Failed to convert {title[2:]} docstring: {e}')
            doc = ""
    else:
        doc = ""
    params = generate_param_docs(base, component)
    examples = generate_examples(base, component)
    return '\n\n'.join([title, doc, '---', params, '---', examples])

def generate_examples(base, component):
    component_path = str(EXAMPLES_PATH / base.__name__.lower() / component.__name__)
    yaml_path = pathlib.Path(component_path + '.yaml')
    examples = '## Examples\n\n'
    if yaml_path.is_file():
        examples += f'```yaml\n{yaml_path.read_text()}\n\n```'
    py_path = pathlib.Path(component_path + '.py')
    if py_path.is_file():
        examples += f'```{{pyodide}}\n{yaml_path.read_text()}\n```'
    return '' if examples == '## Examples\n\n' else examples

def generate_grid(base, gridspec='2 2 3 3', gutter=3, margin=0, rel=''):
    page = f'::::{{grid}} {gridspec}\n:gutter: {gutter}\n:margin: {margin}\n\n'
    for component in concrete_descendents(base).values():
        name = component.__name__
        if name.startswith('_'):
            continue
        if component.__doc__:
            description = component.__doc__.split('\n')[1].strip()
        else:
            description = f'{name} {base.__name__}'
        page += f':::{{grid-item-card}} {name}\n:link: {rel}{name}.html\n:shadow: md\n\n{description}\n:::\n\n'
    page += '::::'
    return page

def generate_component_index(base):
    page = generate_page(base, base)
    grid = generate_grid(base)
    return '\n'.join([page, grid])

def generate_multi_component_pages(base):
    index = generate_component_index(base)
    with open(REFERENCE_PATH /  base.__name__.lower() / 'index.md', 'w') as f:
        f.write(index)
    for component in concrete_descendents(base).values():
        if component.__name__.startswith('_'):
            continue
        page = generate_page(base, component)
        path = REFERENCE_PATH /  base.__name__.lower() / f'{component.__name__}.md'
        print(f'Writing {path}.')
        with open(path, 'w') as f:
            f.write(page)
    return page

def generate_single_component_page(component):
    page = generate_page(component, component)
    path = REFERENCE_PATH / f'{component.__name__}.md'
    with open(path, 'w') as f:
        f.write(page)

def write_index(bases, multi_bases):
    page = '# Reference\n\n'
    for base in bases:
        page += f'## [`{base.__name__}`]({base.__name__})\n\n'
    for base in multi_bases:
        page += f'## [`{base.__name__}`]({base.__name__})\n\n'
        page += generate_grid(base, rel=f'{base.__name__}/')
        page += '\n\n'
    path = REFERENCE_PATH / f'index.md'
    with open(path, 'w') as f:
        f.write(page)

if __name__ == '__main__':
    for component in bases:
        generate_single_component_page(component)
    for base in multi_bases:
        generate_multi_component_pages(base)
    write_index(bases, multi_bases)
