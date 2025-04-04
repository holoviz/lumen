import numbers
import pathlib
import sys
import textwrap
import types
import typing

import numpy as np
import pandas as pd
import param

param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

from param import concrete_descendents

import lumen
import lumen.sources.duckdb
import lumen.sources.intake
import lumen.sources.intake_sql

from lumen.ai.actor import Actor
from lumen.ai.agents import Agent
from lumen.ai.coordinator import Coordinator
from lumen.ai.tools import Tool
from lumen.ai.ui import UI
from lumen.base import MultiTypeComponent
from lumen.dashboard import Auth, Config, Defaults
from lumen.filters import Filter
from lumen.layout import Layout
from lumen.pipeline import Pipeline
from lumen.sources import Source
from lumen.transforms import Transform
from lumen.variables import Variable
from lumen.views import View

sections = {
    'AI': [Agent, Coordinator, Tool, UI],
    'Spec': [
        Config, Variable, Pipeline, Source, Filter,
        Transform, View, Layout, Auth, Defaults
    ]
}

BASE_PATH = pathlib.Path(lumen.__file__).parent.parent
REFERENCE_PATH = BASE_PATH / 'doc' / 'reference'
EXAMPLES_PATH = BASE_PATH / 'examples' / 'reference'

def escape_param_doc(doc):
    return textwrap.dedent(doc or '').strip().replace('\n', '')

def _search_module(obj):
    path = obj.__module__.split('.')
    current = path[0]
    for p in path[1:]:
        if getattr(sys.modules[current], obj.__name__, None) is obj:
            return current
        current += f'.{p}'
    return current

def _format_typing(objtype):
    if hasattr(objtype, '__args__'):
        args = ', '.join([format_type(t) for t in objtype.__args__])
    else:
        return str(objtype).replace('typing.', '')
    origin = objtype.__origin__
    if origin is typing.Union:
        return ' | '.join([format_type(t) for t in objtype.__args__])
    out = format_type(origin)
    if args:
        return f'{out}[{args}]'
    return out

def format_type(objtype):
    if hasattr(objtype, '__module__'):
        if objtype.__module__.startswith('builtins'):
            return objtype.__name__
        elif objtype.__module__.startswith('typing'):
            return _format_typing(objtype)
        elif objtype is numbers.Number:
            return 'Number'
        module = _search_module(objtype)
        return f'{module}.{objtype.__name__}'
    return objtype.__name__

def format_object(obj):
    if isinstance(obj, type):
        return format_type(obj)
    elif isinstance(obj, param.Parameterized):
        return f'{type(obj).__name__}()'
    return repr(obj)

def extract_type(pobj):
    if isinstance(pobj, param.Path):
        return typing.Union[str, pathlib.Path]
    elif isinstance(pobj, param.String):
        return str
    if isinstance(pobj, param.Integer):
        return int
    elif isinstance(pobj, param.Boolean):
        return bool
    elif isinstance(pobj, param.Number):
        return numbers.Number
    elif isinstance(pobj, param.NumericTuple):
        if isinstance(pobj.length):
            return tuple[(numbers.Number,)*pobj.length]
        else:
            return tuple[numbers.Number, ...]
    elif isinstance(pobj, param.Tuple):
        if isinstance(pobj.length):
            return tuple[(typing.Any,)*pobj.length]
        else:
            return tuple[typing.Any, ...]
    elif isinstance(pobj, param.List):
        item_type = pobj.item_type or pobj.class_
        if item_type is None:
            return list[typing.Any]
        if isinstance(item_type, tuple):
            item_type = typing.Union[item_type]
        return list[item_type]
    elif isinstance(pobj, param.Dict):
        return dict[typing.Hashable, typing.Any]
    elif isinstance(pobj, param.Array):
        return np.ndarray
    elif isinstance(pobj, param.DataFrame):
        return pd.DataFrame
    elif isinstance(pobj, param.Series):
        return pd.Series
    return typing.Any

def generate_param_docs(base, component, gutter=3, margin=0):
    if base is component:
        params = sorted(base.param)
    else:
        params = sorted([p for p in component.param if p not in base.param])
    grid_items = []
    for pname in params:
        if pname.startswith('_'):
            continue
        pobj = component.param[pname]
        ptype = type(pobj)
        pbadge = f'<span style="font-size: 1.25em;">{{bdg-dark}}`{pname}`'
        if pname in getattr(component, '_required_keys', []):
            pbadge += '&emsp;&emsp;&emsp;{bdg-warning}`required`'
        pbadge += '</span>'
        pitems = []
        pdefault = format_object(pobj.default)
        if isinstance(pobj, param.ClassSelector):
            if isinstance(pobj.class_, tuple):
                types = ' | '.join([format_object(cls) for cls in pobj.class_])
            else:
                types = format_object(pobj.class_)
            pitems.append(f'**type**: `{types}`')
        else:
            ptype = format_type(extract_type(pobj))
            pitems.append(f'**type**: `{ptype}`')
        pitems.append(f'**default**: `{pdefault}`')
        constraint = None
        if isinstance(pobj, param.Number):
            constraint = f'**bounds**: {pobj.bounds}'
        elif isinstance(pobj, param.Selector):
            objects = pobj.get_range()
            if isinstance(objects, dict):
                objects= objects.values()
            objects = ' | '.join([format_object(o) for o in objects])
            constraint = f'**Possible values**: `{objects}`'
        if constraint:
            pitems.append(constraint)
        if pobj.doc:
            pdoc = escape_param_doc(pobj.doc)
            pitems.append(f'<span style="vertical-align: bottom;">{pdoc}</span>')
        param_item = '<br>'.join(pitems)
        grid_item = f':::{{grid-item-card}} {pbadge}\n:shadow: md\n\n{param_item}\n:::'
        grid_items.append(grid_item)
    if not grid_items:
        return None
    grid_items = '\n\n'.join(grid_items)
    grid = f'::::{{grid}} 1 1 2 2\n:gutter: {gutter}\n:margin: {margin}\n\n{grid_items}\n\n::::'
    return f'## Parameters\n\n{grid}'

PARAM_DEPRECATED = [
    'debug', 'defaults', 'message', 'pprint', 'print_param_values', 'script_repr',
    'state_pop', 'state_push', 'script_repr', 'verbose', 'warning'
]

def generate_automethods(component):
    methods = [
        member for member in dir(component)
        if not member.startswith('_') and isinstance(getattr(component, member), types.FunctionType)
        and member not in PARAM_DEPRECATED
    ]
    if not methods:
        return None
    automethods = ""
    for method in methods:
        automethods += f"""
```{{eval-rst}}
.. automethod:: {component.__module__}.{component.__name__}.{method}
```"""
    return f'## Methods\n\n{automethods}'

def generate_page(base, component):
    title = f'# {component.__name__}'
    if base is not component and issubclass(component, MultiTypeComponent):
        ctype = getattr(component, f'{base.__name__.lower()}_type')
        title += f'&nbsp;&nbsp;{{bdg-primary}}`type: {ctype}`'
    title += '\n'
    page_items = [title]
    page_items.append(f"""
```{{eval-rst}}
.. autoclass:: {component.__module__}.{component.__name__}
```\n\n""")
    params = generate_param_docs(base, component)
    if params:
        page_items.extend(['\n---', params])
    methods = generate_automethods(component)
    if methods:
        page_items.extend(['\n---', methods])
    examples = generate_examples(base, component)
    if examples:
        page_items.extend(['\n---', examples])
    return '\n'.join(page_items)

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
        ctype = getattr(component, f'{base.__name__.lower()}_type', None)
        badge = f'   {{bdg-primary}}`type: {ctype}`' if ctype else ''
        page += f':::{{grid-item-card}} {name}{badge}\n:link: {rel}{name}.html\n:shadow: md\n\n{description}\n:::\n\n'
    page += '::::'
    return page

def generate_component_index(base):
    page = generate_page(base, base)
    grid = generate_grid(base)
    return '\n\n'.join([page, '## Types', grid])

def generate_multi_component_pages(base):
    index = generate_component_index(base)
    component_path = REFERENCE_PATH /  base.__name__.lower()
    component_path.mkdir(exist_ok=True, parents=True)
    toc = ''
    for component in concrete_descendents(base).values():
        if component.__name__.startswith('_'):
            continue
        page = generate_page(base, component)
        name = component.__name__
        path = REFERENCE_PATH /  base.__name__.lower() / f'{name}.md'
        print(f'Writing {path}.')
        with open(path, 'w') as f:
            f.write(page)
        toc += f'{name}\n'
    index += f"""

```{{toctree}}
---
hidden: true
---
{toc[:-1]}
```
"""
    with open(component_path / 'index.md', 'w') as f:
        f.write(index)
    return page

def generate_single_component_page(component):
    page = generate_page(component, component)
    path = REFERENCE_PATH / f'{component.__name__}.md'
    with open(path, 'w') as f:
        f.write(page)

def write_index(sections):
    page = '# Reference\n\n'
    for section, bases in sections.items():
        toc = ''
        page += f'## {section}\n\n'
        for base in bases:
            name = base.__name__
            if base.__doc__:
                description = base.__doc__.split("\n")[1].strip()
            else:
                description = None
            if issubclass(base, (Actor, MultiTypeComponent, UI)):
                page += f'### [`{name}`]({name.lower()}/index)\n\n'
                if description:
                    page += f'{description}\n\n'
                page += generate_grid(base, rel=f'{base.__name__.lower()}/')
                page += '\n\n'
            else:
                page += f'### [`{name}`]({name})\n\n'
                if description:
                    page += f'{description}\n\n'
            toc += f'{name} <{name.lower()}/index>\n'
        page += f"""
```{{toctree}}
---
hidden: true
---
{toc[:-1]}
```
"""
    path = REFERENCE_PATH / 'index.md'
    with open(path, 'w') as f:
        f.write(page)

if __name__ == '__main__':
    REFERENCE_PATH.mkdir(exist_ok=True, parents=True)
    for bases in sections.values():
        for component in bases:
            if issubclass(component, (Actor, MultiTypeComponent, UI)):
                generate_multi_component_pages(component)
            else:
                generate_single_component_page(component)
    write_index(sections)
