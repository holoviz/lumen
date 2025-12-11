# Access custom components

!!! important
    This guide shows how to use local files to build custom components.

## Overview

While Lumen includes many built-in components, you can define custom components in local files. There are two approaches to reference them.

**Approach 1: Automatic imports** — Save to specific filenames (`filters.py`, `sources.py`, `transforms.py`, `views.py`) and reference by simple name.

**Approach 2: Module paths** — Save to custom folders and reference using dot notation (e.g., `my_library.my_module.TextEditor`).

## Approach 1: Automatic imports

Lumen automatically imports `filters.py`, `sources.py`, `transforms.py`, and `views.py` if they exist alongside your YAML file.

Create `transforms.py`:

```python
import param
from lumen import Transform

class StableSort(Transform):
    """Uses stable sorting algorithm on columns."""

    by = param.ListSelector(default=[], doc="Columns to sort by")

    ascending = param.ClassSelector(
        default=True,
        class_=(bool, list),
        doc="Sort ascending vs descending"
    )

    transform_type = 'stablesort'
    _field_params = ['by']

    def apply(self, table):
        return table.sort_values(
            self.by,
            ascending=self.ascending,
            kind='stable'
        )
```

Create `views.py`:

```python
from lumen import View
import panel as pn

class TextEditor(View):
    """Provides a rich text editor."""

    view_type = 'texteditor'
    _extension = 'texteditor'

    def get_panel(self):
        return pn.widgets.TextEditor(**self._get_params())

    def _get_params(self):
        return dict(
            **self.kwargs,
            sizing_mode='stretch_width',
            placeholder='Enter some text'
        )
```

Reference in your YAML (dashboard.yaml):

```yaml
config:
  title: Palmer Penguins

sources:
  penguin_source:
    type: file
    tables:
      penguin_table: https://datasets.holoviz.org/penguins/v1/penguins.csv

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
      - type: stablesort
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

Launch with:

```bash
lumen serve dashboard.yaml --show
```

## Approach 2: Module paths

Create custom modules in a folder next to your YAML file. Reference using dot notation (e.g., `my_library.my_module.StableSort`).

Create `my_library/my_module.py`:

```python
import param
from lumen import Transform, View
import panel as pn

class StableSort(Transform):
    """Uses stable sorting algorithm on columns."""

    by = param.ListSelector(default=[], doc="Columns to sort by")

    ascending = param.ClassSelector(
        default=True,
        class_=(bool, list),
        doc="Sort ascending vs descending"
    )

    transform_type = 'stablesort'
    _field_params = ['by']

    def apply(self, table):
        return table.sort_values(
            self.by,
            ascending=self.ascending,
            kind='stable'
        )


class TextEditor(View):
    """Provides a rich text editor."""

    view_type = 'texteditor'
    _extension = 'texteditor'

    def get_panel(self):
        return pn.widgets.TextEditor(**self._get_params())

    def _get_params(self):
        return dict(
            **self.kwargs,
            sizing_mode='stretch_width',
            placeholder='Enter some text'
        )
```

Reference in your YAML using the full module path:

```yaml
pipelines:
  penguin_pipeline:
    source: penguin_source
    table: penguin_table
    transforms:
      - type: my_library.my_module.StableSort
        by: island

layouts:
  - title: Penguins
    views:
      - type: my_library.my_module.TextEditor
        height: 250
```

Launch with:

```bash
lumen serve dashboard.yaml --show
```
