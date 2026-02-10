# FileDropper Configuration Suggestions

## Summary
Add configurable size limits to the FileDropper widget in both `controls.py` and `ui.py` to provide reasonable defaults while allowing users to customize upload limits.

---

## 1. Configuration in `controls.py`

### Location: DownloadConfig class (lines 38-51)

**Current code:**
```python
class DownloadConfig:
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    TIMEOUT_SECONDS = 300     # 5 minutes
    PROGRESS_UPDATE_INTERVAL = 50  # Update every 50 chunks
    DEFAULT_HASH_MODULO = 10000
    UNKNOWN_SIZE_MAX = 1000000000  # 1GB max for unknown file sizes

    # Connection settings
    CONNECTION_LIMIT = 100
    CONNECTION_LIMIT_PER_HOST = 30
    KEEPALIVE_TIMEOUT = 30
```

**Suggested addition:**
```python
class DownloadConfig:
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    TIMEOUT_SECONDS = 300     # 5 minutes
    PROGRESS_UPDATE_INTERVAL = 50  # Update every 50 chunks
    DEFAULT_HASH_MODULO = 10000
    UNKNOWN_SIZE_MAX = 1000000000  # 1GB max for unknown file sizes

    # Connection settings
    CONNECTION_LIMIT = 100
    CONNECTION_LIMIT_PER_HOST = 30
    KEEPALIVE_TIMEOUT = 30


class UploadConfig:
    """Configuration for FileDropper upload limits."""
    MAX_FILE_SIZE = '100MB'           # Maximum size per file
    MAX_TOTAL_FILE_SIZE = '500MB'     # Maximum total upload size
    MAX_FILES = 20                    # Maximum number of files
    CHUNK_SIZE = 10000000             # 10MB chunks (FileDropper default)
```

---

## 2. Configuration in `UploadControls.__init__` (lines 650-665)

**Current code:**
```python
def __init__(self, **params):
    super().__init__(**params)

    self._file_input = FileDropper(
        layout="compact",
        multiple=self.param.multiple,
        margin=1,
        sizing_mode="stretch_width",
        disabled=self.param.disabled,
        stylesheets=[".bk-input.filepond--root { box-shadow: unset; cursor: grab; } .bk-input.filepond--root:not([disabled]):hover { box-shadow: unset; }"],
        visible=self._upload_cards.param.visible.rx.not_()
    )
```

**Suggested modification:**
```python
def __init__(self, **params):
    super().__init__(**params)

    self._file_input = FileDropper(
        layout="compact",
        multiple=self.param.multiple,
        max_file_size=UploadConfig.MAX_FILE_SIZE,
        max_total_file_size=UploadConfig.MAX_TOTAL_FILE_SIZE,
        max_files=UploadConfig.MAX_FILES,
        chunk_size=UploadConfig.CHUNK_SIZE,
        margin=1,
        sizing_mode="stretch_width",
        disabled=self.param.disabled,
        stylesheets=[".bk-input.filepond--root { box-shadow: unset; cursor: grab; } .bk-input.filepond--root:not([disabled]):hover { box-shadow: unset; }"],
        visible=self._upload_cards.param.visible.rx.not_()
    )
```

---

## 3. Optional: Add parameters to UploadControls class

If you want to make these configurable per-instance:

**Location: UploadControls class definition (around line 641)**

```python
class UploadControls(BaseSourceControls):
    """
    Controls for uploading files from the local filesystem.
    """

    max_file_size = param.String(default=UploadConfig.MAX_FILE_SIZE, doc="""
        Maximum size per file (e.g., '100MB', '1GB')""")
    
    max_total_file_size = param.String(default=UploadConfig.MAX_TOTAL_FILE_SIZE, doc="""
        Maximum total size of all uploads (e.g., '500MB', '2GB')""")
    
    max_files = param.Integer(default=UploadConfig.MAX_FILES, doc="""
        Maximum number of files that can be uploaded""")
    
    chunk_size = param.Integer(default=UploadConfig.CHUNK_SIZE, doc="""
        Size in bytes per chunk transferred across WebSocket""")

    label = '<span class="material-icons" style="vertical-align: middle;">upload</span> Upload Data'

    def __init__(self, **params):
        super().__init__(**params)

        self._file_input = FileDropper(
            layout="compact",
            multiple=self.param.multiple,
            max_file_size=self.max_file_size,
            max_total_file_size=self.max_total_file_size,
            max_files=self.max_files,
            chunk_size=self.chunk_size,
            margin=1,
            sizing_mode="stretch_width",
            disabled=self.param.disabled,
            stylesheets=[".bk-input.filepond--root { box-shadow: unset; cursor: grab; } .bk-input.filepond--root:not([disabled]):hover { box-shadow: unset; }"],
            visible=self._upload_cards.param.visible.rx.not_()
        )
```

---

## 4. Configuration in `ui.py`

### Location: UI class (around line 361)

If you want to expose upload config at the UI level:

```python
class UI(Viewer):
    """
    UI provides a baseclass and high-level entrypoint to start chatting with your data.
    """

    # ... existing parameters ...
    
    upload_config = param.Dict(default={}, doc="""
        Configuration for file upload limits. Keys can include:
        - max_file_size: str (e.g., '100MB')
        - max_total_file_size: str (e.g., '500MB') 
        - max_files: int
        - chunk_size: int (bytes)""")
```

Then pass this config when initializing UploadControls in `_render_main`:

```python
# In _render_main method (around line 1252)
for control in self.source_controls:
    control_params = {
        'context': self.context,
        'source_catalog': self._source_catalog,
        'upload_handlers': self.upload_handlers
    }
    
    # Apply upload config if this is UploadControls
    if control is UploadControls and self.upload_config:
        control_params.update(self.upload_config)
    
    control_inst = control(**control_params)
    # ... rest of code
```

---

## Recommended Values

Based on the discussion:

- **Conservative (default):** 
  - `max_file_size='100MB'`
  - `max_total_file_size='500MB'`
  - `max_files=20`

- **Moderate (data science use):**
  - `max_file_size='500MB'`
  - `max_total_file_size='2GB'`
  - `max_files=50`

- **Aggressive (specialized):**
  - `max_file_size='5GB'`
  - `max_total_file_size='10GB'`
  - `max_files=100`

---

## Implementation Priority

1. **High priority:** Add `UploadConfig` class and apply to `FileDropper` in `UploadControls.__init__`
2. **Medium priority:** Add parameters to `UploadControls` class for per-instance configuration
3. **Low priority:** Expose config at UI level (only if users need to customize per deployment)
