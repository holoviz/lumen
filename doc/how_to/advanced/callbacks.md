# How to define callbacks

Often times you want to perform custom actions when a session launches or ends or when a user interacts with a data pipeline. Since Lumen applications are specified as a declarative specification the easiest way to to do this is to define hooks.

The [`Config`](lumen.dashboard.Config) object provides a number of hooks that can be defined:

- `on_session_created`: Callback that fires when a user session is created.
- `on_session_destroyed`: Callback that fires when a user session is destroyed.
- `on_loaded`: Callback that fires when a user frontend session is fully loaded.
- `on_error`: Callback that fires if an error occurs in a dashboard callback.
- `on_update`: Callback that fires when a pipeline is updated.

## Reference callbacks in external module

Since we have to specify callbacks as a reference to a specific module the function or method must be in an importable location. This can either be a reference to a function in an installed a package or a module shipped alongside the YAML file, e.g. if we declare a file called `callbacks.py`:

```python
import panel as pn

def created():
   print(f'Session created for user {pn.state.user}')

def updated(pipeline):
   print(f'Pipeline {pipeline.name} was updated.')
```

We can now reference these callbacks from the YAML:

```yaml
config:
  on_session_created: callbacks.created
  on_update: callbacks.updated
```

## Serializing callbacks

The fact that the functions have to be importable also means that when defining the callbacks programmatically they can only be serialized if they were defined in an external module.

As an example we cannot do something like this:

```python
import lumen as lm

def created():
   print('Session created.')

lm.Config(on_session_created=created).to_spec()
```

We must move the callback into a module that can be imported.
