# Define callbacks

Perform custom actions when sessions launch, end, or when users interact with pipelines.

## Available hooks

The `Config` object provides these hooks:

- `on_session_created`: Fires when a user session starts
- `on_session_destroyed`: Fires when a user session ends
- `on_loaded`: Fires when the frontend is fully loaded
- `on_error`: Fires when a dashboard callback encounters an error
- `on_update`: Fires when a pipeline updates

## Reference callbacks

Callbacks must reference importable functions. They can be in installed packages or in a module alongside your YAML file.

Create a `callbacks.py` file:

```python
import panel as pn

def created():
    print(f'Session created for user {pn.state.user}')

def updated(pipeline):
    print(f'Pipeline {pipeline.name} was updated.')
```

Reference them in your YAML:

```yaml
config:
  on_session_created: callbacks.created
  on_update: callbacks.updated
```

## Serializing callbacks

When defining callbacks programmatically, they can only be serialized if defined in an external module.

This **won't work**:

```python
import lumen as lm

def created():
    print('Session created.')

lm.Config(on_session_created=created).to_spec()
```

Instead, define the callback in a separate module and import it.
