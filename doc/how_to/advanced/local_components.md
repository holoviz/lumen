# Access custom components

```{admonition} What does this guide solve?
---
class: important
---
This guide will show you how to access local files that build custom components.
```

## Overview
While Lumen ships with a wide range of components, users may also define custom components in files which live alongside the YAML file. There are two approaches to reference these local custom components.

The first approach is to save your custom components to files with specific names that Lumen will automatically import. With this approach, you can just refer to your components in the YAML specification using their simple string name (for example, `texteditor`).

The second approach is to save your custom components to your own folder and filenames. With this approach, you can refer to your components using their module path (for example, `my_library.my_module.TextEditor`).

In the custom component examples below, we will add a transform for [stable sorting](https://en.wikipedia.org/wiki/Category:Stable_sorts), and a view for a [rich text editor](https://panel.holoviz.org/reference/widgets/TextEditor.html#widgets-gallery-texteditor).

## Approach 1: Using imported files

Lumen will automatically import `filters.py`, `sources.py`, `transforms.py`, and `views.py` if these files exist alongside the YAML dashboard specification.

For example, if you created a custom transform and a custom view, all you have to do is save the custom transform to `transforms.py` and the custom view to `views.py`, and then place them in the same directory as your YAML specification. Now you can reference these local components within your specification using their simple string name, and they will be applied when you launch your dashboard.

Here are our example custom component files:

::::{tab-set}

:::{tab-item} transforms.py
```{code-block} python
import param
from lumen import Transform

class StableSort(Transform):
    """
    Uses a stable sorting algorithm on one or more columns.

    See `pandas.DataFrame.sort_values` with kind='stable'

    df.sort_values(<by>, ascending=<ascending>, kind='stable')
    """

    by = param.ListSelector(default=[], doc="""
       Columns or indexes to sort by.""")

    ascending = param.ClassSelector(default=True, class_=(bool, list), doc="""
       Sort ascending vs. descending. Specify list for multiple sort
       orders. If this is a list of bools, must match the length of
       the by.""")

    transform_type = 'stablesort'

    _field_params = ['by']

    def apply(self, table):
        return table.sort_values(self.by, ascending=self.ascending, kind='stable')
```
:::

:::{tab-item} views.py
```{code-block} python
from lumen import View
import panel as pn

class TextEditor(View):
    """
    Provides a rich text editor.

    See https://panel.holoviz.org/reference/widgets/TextEditor.html#widgets-gallery-texteditor
    """

    view_type = 'texteditor'

    _extension = 'texteditor'

    def get_panel(self):
        return pn.widgets.TextEditor(**self._get_params())

    def _get_params(self):
        return dict(**self.kwargs, sizing_mode='stretch_width', placeholder='Enter some text')
```
:::

::::

And here is our example YAML specification (dashboard.yaml) that references these local custom component files:

```{code-block} yaml
---
emphasize-lines: 22-23, 47-48
---
config:
  title: Palmer Penguins

sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv

pipelines:
  penguin_pipeline:
    source: penguin_source
    table: penguin_table
    filters:
      - type: widget
        field: sex
      - type: widget
        field: island
    transforms:
      - type: columns
        columns: ['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']
      - type: my_module.StableSor
        by: island

layouts:
  - title: Penguins
    pipeline: penguin_pipeline
    layout: [[0, 1], [2, 3]]
    sizing_mode: stretch_width
    height: 800
    views:
      - type: hvplot
        x: bill_length_mm
        y: bill_depth_mm
        kind: scatter
        color: species
        responsive: true
        height: 400
      - type: hvplot
        kind: hist
        y: bill_length_mm
        responsive: true
        height: 300
      - type: table
        show_index: false
        height: 300
      - type: texteditor
        height: 250
```

Now launch the dashboard with:

```{code-block} bash
lumen serve dashboard.yaml --show
```


![](../_static/how_to/local_components/local_components_app.png)


## Approach 2: Using module paths

Lumen will import custom components from your own modules (folders and files) if referenced in the YAML specification using dot notation (for example, `my_library.my_module.TextEditor`). If using this approach, the root of the module path (for example, `my_library`) needs to be in the same folder as the YAML specification.

Here is our example custom component file. In this case, we combined the components into one file, but they could have been in different module paths:

**my_library/my_module.py**
```{code-block} python

import param
from lumen import Transform, View
import panel as pn

class StableSort(Transform):
    """
    Uses a stable sorting algorithm on one or more columns.

    See `pandas.DataFrame.sort_values` with kind='stable'

    df.sort_values(<by>, ascending=<ascending>, kind='stable')
    """

    by = param.ListSelector(default=[], doc="""
       Columns or indexes to sort by.""")

    ascending = param.ClassSelector(default=True, class_=(bool, list), doc="""
       Sort ascending vs. descending. Specify list for multiple sort
       orders. If this is a list of bools, must match the length of
       the by.""")

    transform_type = 'stablesort'

    _field_params = ['by']

    def apply(self, table):
        return table.sort_values(self.by, ascending=self.ascending, kind='stable')


class TextEditor(View):
    """
    Provides a rich text editor.

    See https://panel.holoviz.org/reference/widgets/TextEditor.html#widgets-gallery-texteditor
    """

    view_type = 'texteditor'

    _extension = 'texteditor'

    def get_panel(self):
        return pn.widgets.TextEditor(**self._get_params())

    def _get_params(self):
        return dict(**self.kwargs, sizing_mode='stretch_width', placeholder='Enter some text')

```

And here is our example YAML specification (dashboard.yaml) that references these local custom components:

```{code-block} yaml
:emphasize-lines: 22-23, 47-48
config:
  title: Palmer Penguins

sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv

pipelines:
  penguin_pipeline:
    source: penguin_source
    table: penguin_table
    filters:
      - type: widget
        field: sex
      - type: widget
        field: island
    transforms:
      - type: columns
        columns: ['species', 'island', 'sex', 'year', 'bill_length_mm', 'bill_depth_mm']
      - type: my_library.my_module.StableSort
        by: island

layouts:
  - title: Penguins
    pipeline: penguin_pipeline
    layout: [[0, 1], [2, 3]]
    sizing_mode: stretch_width
    height: 800
    views:
      - type: hvplot
        x: bill_length_mm
        y: bill_depth_mm
        kind: scatter
        color: species
        responsive: true
        height: 400
      - type: hvplot
        kind: hist
        y: bill_length_mm
        responsive: true
        height: 300
      - type: table
        show_index: false
        height: 300
      - type: my_library.my_module.TextEditor
        height: 250
```

Now launch the dashboard with:

```{code-block} bash
lumen serve dashboard.yaml --show
```

![](../_static/how_to/local_components/local_components_app.png)


## Related Resources
* See [how to](index.md) guides on building custom components
