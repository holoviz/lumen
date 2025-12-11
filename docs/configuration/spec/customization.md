# Extending Lumen with custom components

Build custom components and callbacks to extend Lumen's functionality.

## Customization overview

Lumen provides two main extension points:

| Extension Type | Purpose | Examples |
|----------------|---------|----------|
| **Custom components** | New sources, transforms, filters, or views | Custom data source, special transform |
| **Callbacks** | Actions triggered by events | Logging, notifications, custom workflows |

## Custom components

Create custom sources, transforms, filters, or views when built-in components don't meet your needs.

### Component types

You can customize these component types:

| File | Component Type | Base Class |
|------|----------------|------------|
| `sources.py` | Data sources | `lumen.sources.Source` |
| `transforms.py` | Data transforms | `lumen.transforms.Transform` |
| `filters.py` | Data filters | `lumen.filters.Filter` |
| `views.py` | Visualizations | `lumen.views.View` |

### Two approaches

**Approach 1: Automatic imports** — Save to specific filenames next to your YAML file. Reference by simple name.

**Approach 2: Module paths** — Save to custom folders. Reference using dot notation.

## Approach 1: Automatic imports

Lumen automatically imports these files if they exist alongside your YAML:

- `sources.py`
- `transforms.py`
- `filters.py`
- `views.py`

### Example: Custom transform

Create `transforms.py`:

```python
import param
from lumen.transforms import Transform

class StableSort(Transform):
    """Sort using stable algorithm."""
    
    by = param.ListSelector(
        default=[],
        doc="Columns to sort by"
    )
    
    ascending = param.ClassSelector(
        default=True,
        class_=(bool, list),
        doc="Sort ascending vs descending"
    )
    
    transform_type = 'stablesort'  # Name for YAML
    _field_params = ['by']         # Parameters that accept field names
    
    def apply(self, table):
        """Apply the transform to data."""
        return table.sort_values(
            self.by,
            ascending=self.ascending,
            kind='stable'
        )
```

Reference in YAML:

```yaml
pipelines:
  sorted_data:
    source: my_source
    table: my_table
    transforms:
      - type: stablesort          # Uses your custom transform
        by: [date, category]
        ascending: [true, false]
```

### Example: Custom view

Create `views.py`:

```python
from lumen.views import View
import panel as pn

class TextEditor(View):
    """Rich text editor view."""
    
    view_type = 'texteditor'       # Name for YAML
    _extension = 'texteditor'      # Panel extension to load
    
    def get_panel(self):
        """Return the Panel component."""
        return pn.widgets.TextEditor(**self._get_params())
    
    def _get_params(self):
        """Get parameters for the widget."""
        return dict(
            **self.kwargs,
            sizing_mode='stretch_width',
            placeholder='Enter some text'
        )
```

Reference in YAML:

```yaml
layouts:
  - title: Dashboard
    views:
      - type: texteditor          # Uses your custom view
        height: 250
        placeholder: "Enter notes here"
```

### Example: Custom filter

Create `filters.py`:

```python
import param
from lumen.filters import Filter

class RangeFilter(Filter):
    """Filter for numeric ranges."""
    
    min_value = param.Number(default=0)
    max_value = param.Number(default=100)
    
    filter_type = 'range'
    
    def apply(self, table):
        """Apply the filter."""
        return table[
            (table[self.field] >= self.min_value) &
            (table[self.field] <= self.max_value)
        ]
```

Reference in YAML:

```yaml
pipelines:
  filtered:
    source: my_source
    table: my_table
    filters:
      - type: range
        field: price
        min_value: 10
        max_value: 1000
```

### Example: Custom source

Create `sources.py`:

```python
import param
import pandas as pd
from lumen.sources import Source

class APISource(Source):
    """Load data from a REST API."""
    
    api_url = param.String(doc="API endpoint URL")
    api_key = param.String(doc="API authentication key")
    
    source_type = 'api'
    
    def get(self, table, **query):
        """Fetch data from API."""
        import requests
        
        response = requests.get(
            f"{self.api_url}/{table}",
            headers={'Authorization': f'Bearer {self.api_key}'},
            params=query
        )
        response.raise_for_status()
        return pd.DataFrame(response.json())
```

Reference in YAML:

```yaml
sources:
  my_api:
    type: api
    api_url: https://api.example.com/v1
    api_key: {{ env("API_KEY") }}
    tables:
      users: users
      orders: orders
```

### Complete example

File structure:

```
project/
├── dashboard.yaml
├── transforms.py
└── views.py
```

**transforms.py:**

```python
import param
from lumen.transforms import Transform

class StableSort(Transform):
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

**views.py:**

```python
from lumen.views import View
import panel as pn

class TextEditor(View):
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

**dashboard.yaml:**

```yaml
config:
  title: Custom Components Demo

sources:
  penguins:
    type: file
    tables:
      data: https://datasets.holoviz.org/penguins/v1/penguins.csv

pipelines:
  processed:
    source: penguins
    table: data
    filters:
      - type: widget
        field: species
    transforms:
      - type: columns
        columns: [species, island, bill_length_mm, bill_depth_mm]
      - type: stablesort          # Custom transform
        by: [species, island]

layouts:
  - title: Analysis
    pipeline: processed
    layout: [[0, 1], [2]]
    sizing_mode: stretch_width
    views:
      - type: hvplot
        kind: scatter
        x: bill_length_mm
        y: bill_depth_mm
        color: species
        height: 400
      - type: table
        show_index: false
        height: 400
      - type: texteditor          # Custom view
        height: 200
```

Launch with:

```bash
lumen serve dashboard.yaml --show
```

## Approach 2: Module paths

Organize custom components in a module structure and reference using dot notation.

### Module structure

Create a package next to your YAML file:

```
project/
├── dashboard.yaml
└── my_library/
    ├── __init__.py
    └── custom.py
```

**my_library/custom.py:**

```python
import param
from lumen.transforms import Transform
from lumen.views import View
import panel as pn

class StableSort(Transform):
    by = param.ListSelector(default=[])
    ascending = param.ClassSelector(default=True, class_=(bool, list))
    transform_type = 'stablesort'
    _field_params = ['by']
    
    def apply(self, table):
        return table.sort_values(self.by, ascending=self.ascending, kind='stable')


class TextEditor(View):
    view_type = 'texteditor'
    _extension = 'texteditor'
    
    def get_panel(self):
        return pn.widgets.TextEditor(**self._get_params())
    
    def _get_params(self):
        return dict(**self.kwargs, sizing_mode='stretch_width')
```

### Reference with module path

Use full module path in YAML:

```yaml
pipelines:
  processed:
    source: my_source
    table: my_table
    transforms:
      - type: my_library.custom.StableSort  # Full module path
        by: [date]

layouts:
  - title: Dashboard
    views:
      - type: my_library.custom.TextEditor  # Full module path
        height: 250
```

### When to use module paths

Use module paths when:

- You have many custom components
- You want to organize components by functionality
- You're building a reusable library
- You need namespace separation

Use automatic imports when:

- You have few custom components
- You want simple, quick customization
- Components are specific to one dashboard

## Callbacks

Callbacks perform actions when specific events occur.

### Available callback hooks

The `Config` object provides these hooks:

| Hook | Trigger | Use case |
|------|---------|----------|
| `on_session_created` | User session starts | Logging, initialization |
| `on_session_destroyed` | User session ends | Cleanup, analytics |
| `on_loaded` | Frontend fully loaded | Ready notifications |
| `on_error` | Dashboard callback errors | Error handling |
| `on_update` | Pipeline updates | Change tracking, logging |

### Defining callbacks

Callbacks must be importable functions. Create `callbacks.py` next to your YAML:

```python
import panel as pn

def session_created():
    """Called when a user session starts."""
    print(f'Session created for user {pn.state.user}')
    print(f'Session ID: {pn.state.session_id}')

def session_destroyed():
    """Called when a user session ends."""
    print(f'Session ended for user {pn.state.user}')

def frontend_loaded():
    """Called when the frontend finishes loading."""
    print('Dashboard loaded successfully')

def pipeline_updated(pipeline):
    """Called when a pipeline updates."""
    print(f'Pipeline {pipeline.name} was updated')
    print(f'Current data shape: {pipeline.data.shape}')

def error_occurred(error):
    """Called when an error occurs."""
    print(f'Error: {error}')
    # Send to error tracking service
```

Reference in YAML:

```yaml
config:
  on_session_created: callbacks.session_created
  on_session_destroyed: callbacks.session_destroyed
  on_loaded: callbacks.frontend_loaded
  on_update: callbacks.pipeline_updated
  on_error: callbacks.error_occurred
```

### Callback examples

#### Logging user activity

```python
# callbacks.py
import logging
import panel as pn

logger = logging.getLogger(__name__)

def log_session_start():
    logger.info(f'User {pn.state.user} started session at {pn.state.curdoc.session_context.id}')

def log_pipeline_update(pipeline):
    logger.info(f'Pipeline {pipeline.name} updated. Rows: {len(pipeline.data)}')
```

```yaml
config:
  on_session_created: callbacks.log_session_start
  on_update: callbacks.log_pipeline_update
```

#### Send notifications

```python
# callbacks.py
import requests

def notify_error(error):
    """Send error to Slack webhook."""
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    message = {
        "text": f"Dashboard error: {error}",
        "username": "Lumen Bot"
    }
    requests.post(webhook_url, json=message)
```

```yaml
config:
  on_error: callbacks.notify_error
```

#### Track analytics

```python
# callbacks.py
import panel as pn
from datetime import datetime

sessions = []

def track_session_start():
    sessions.append({
        'user': pn.state.user,
        'start_time': datetime.now(),
        'session_id': pn.state.session_id
    })

def track_session_end():
    for session in sessions:
        if session['session_id'] == pn.state.session_id:
            session['end_time'] = datetime.now()
            session['duration'] = session['end_time'] - session['start_time']
```

```yaml
config:
  on_session_created: callbacks.track_session_start
  on_session_destroyed: callbacks.track_session_end
```

#### Initialize resources

```python
# callbacks.py
import panel as pn

def initialize_session():
    """Set up session-specific resources."""
    if not hasattr(pn.state, 'cache'):
        pn.state.cache = {}
    
    print(f'Initialized cache for {pn.state.user}')

def cleanup_session():
    """Clean up session resources."""
    if hasattr(pn.state, 'cache'):
        pn.state.cache.clear()
        print(f'Cleaned up resources for {pn.state.user}')
```

```yaml
config:
  on_session_created: callbacks.initialize_session
  on_session_destroyed: callbacks.cleanup_session
```

### Callback limitations

!!! warning "Serialization"
    Callbacks defined inline in Python cannot be serialized to YAML:
    
    ```python
    # ❌ Won't work - cannot serialize to YAML
    def created():
        print('Session created')
    
    config = lm.Config(on_session_created=created)
    config.to_spec()  # Fails!
    ```
    
    Define callbacks in external modules instead:
    
    ```python
    # ✅ Works - importable from module
    import lumen as lm
    import callbacks  # External module
    
    config = lm.Config(on_session_created=callbacks.created)
    config.to_spec()  # Succeeds
    ```

## Best practices

### Component design

**Keep components focused**: Each component should do one thing well.

```python
# ✅ Good - focused responsibility
class UpperCaseTransform(Transform):
    def apply(self, table):
        return table.apply(lambda x: x.str.upper() if x.dtype == 'object' else x)

# ❌ Bad - too many responsibilities
class MegaTransform(Transform):
    def apply(self, table):
        # Uppercase, sort, filter, aggregate...
        pass
```

**Document parameters**: Use clear docstrings.

```python
class MyTransform(Transform):
    field = param.String(doc="Column name to transform")
    method = param.Selector(
        default='mean',
        objects=['mean', 'median', 'mode'],
        doc="Aggregation method to apply"
    )
```

**Handle errors gracefully**: Validate inputs and provide helpful error messages.

```python
def apply(self, table):
    if self.field not in table.columns:
        raise ValueError(
            f"Column '{self.field}' not found. "
            f"Available columns: {list(table.columns)}"
        )
    return table[self.field].mean()
```

### Callback design

**Keep callbacks fast**: Don't block the main thread.

```python
# ✅ Good - quick operation
def log_update(pipeline):
    print(f'Updated: {pipeline.name}')

# ❌ Bad - slow operation
def slow_update(pipeline):
    time.sleep(10)  # Blocks dashboard!
```

**Use async for slow operations**:

```python
import asyncio

async def notify_update(pipeline):
    await asyncio.sleep(1)  # Doesn't block
    print(f'Updated: {pipeline.name}')
```

**Handle exceptions**: Don't let callbacks crash the dashboard.

```python
def safe_callback(pipeline):
    try:
        # Your logic
        risky_operation()
    except Exception as e:
        print(f'Callback error: {e}')
        # Dashboard continues running
```

## Troubleshooting

### Component not found

**Problem**: `Unknown type 'mycomponent'`

**Solutions**:
- Check filename is correct (`transforms.py`, `views.py`, etc.)
- Verify `transform_type` or `view_type` matches YAML
- Ensure file is in same directory as YAML
- Check for Python syntax errors in component file

### Import errors

**Problem**: `ImportError` or `ModuleNotFoundError`

**Solutions**:
- Install required packages
- Check module path is correct
- Verify `__init__.py` exists in package directories
- Check Python path includes your modules

### Callback not firing

**Problem**: Callback function doesn't execute

**Solutions**:
- Verify function is importable: `import callbacks; callbacks.my_function`
- Check function signature matches hook requirements
- Look for errors in terminal/logs
- Ensure function name in YAML matches actual function

### Parameters not working

**Problem**: Component parameters don't affect behavior

**Solutions**:
- Check parameter names match between YAML and Python
- Verify parameters are used in `apply()` or `get_panel()` methods
- Check parameter types match declarations
- Use `_field_params` for column-name parameters

## Next steps

Now that you can extend Lumen:

- **[Python API guide](python-api.md)** - Build complete custom applications
- **[Deployment guide](deployment.md)** - Deploy dashboards with custom components
- **Panel documentation** - Learn more about [Panel widgets](https://panel.holoviz.org/reference/index.html)
- **Param documentation** - Understand [parameter declarations](https://param.holoviz.org/)
