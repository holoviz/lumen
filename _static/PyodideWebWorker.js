importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

const QUEUE = [];

const REQUIRES = {}

function sendPatch(patch, buffers, cell_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers,
    id: cell_id
  })
}

function sendStdout(cell_id, stdout) {
  self.postMessage({
    type: 'stdout',
    content: stdout,
    id: cell_id
  })
}
function sendStderr(cell_id, stderr) {
  self.postMessage({
    type: 'stderr',
    content: stderr,
    id: cell_id
  })
}

async function loadApplication(cell_id, path) {
  console.log("Loading pyodide!");
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  self.pyodide.globals.set("sendStdout", sendStdout);
  self.pyodide.globals.set("sendStderr", sendStderr);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const packages = ['https://cdn.holoviz.org/panel/1.7.1/dist/wheels/bokeh-3.7.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/1.7.1/dist/wheels/panel-1.7.1-py3-none-any.whl', 'lumen==0.10.1', 'requests', 'pandas', 'pyodide-http', 'holoviews>=1.15.1'];
  if (path != null) {
    for (const key of Object.keys(REQUIRES)) {
      if (path.replace('.html', '').endsWith(key.replace('.md', ''))) {
        for (const req of REQUIRES[key]) {
          packages.push(req)
        }
      }
    }
  }

  await self.pyodide.runPythonAsync("")
  self.pyodide.runPython("import micropip")
  for (const pkg of packages) {
    self.postMessage({
      type: 'loading',
      msg: `Loading ${pkg}`,
      id: cell_id
    });
    await self.pyodide.runPythonAsync(`
      await micropip.install('${pkg}', keep_going=True);
    `);
  }
  console.log("Packages loaded!");
}

const autodetect_deps_code = `
import json
try:
    from panel.io.mime_render import find_requirements
except Exception:
    from panel.io.mime_render import find_imports as find_requirements
json.dumps(find_requirements(msg.to_py()['code']))`

const exec_code = `
from functools import partial
from panel.io.pyodide import pyrender
from js import console

msg = msg.to_py()
code = msg['code']
stdout_cb = partial(sendStdout, msg['id'])
stderr_cb = partial(sendStderr, msg['id'])
target = f"output-{msg['id']}"
pyrender(code, stdout_cb, stderr_cb, target)`

const onload_code = `
msg = msg.to_py()
if msg['mime'] == 'application/bokeh':
    from panel.io.pyodide import _link_docs_worker
    from panel.io.state import state
    doc = state.cache[f"output-{msg['id']}"]
    _link_docs_worker(doc, sendPatch, msg['id'], 'js')`

const patch_code = `
from panel import state

try:
    from pane.io.pyodide import _convert_json_patch
    patch = _convert_json_patch(msg.patch)
except:
    patch = msg.patch.to_py()
doc = state.cache[f"output-{msg.id}"]
doc.apply_json_patch(patch, setter='js')`

const MESSAGES = {
  patch: patch_code,
  execute: exec_code,
  rendered: onload_code
}

self.onmessage = async (event) => {
  let resolveExecution, rejectExecution;
   const executing = new Promise(function(resolve, reject){
    resolveExecution = resolve;
    rejectExecution = reject;
  });

  const prev_msg = QUEUE[0]
  const msg = {...event.data, executing}
  QUEUE.unshift(msg)

  if (prev_msg) {
    self.postMessage({
      type: 'loading',
      msg: 'Awaiting previous cells',
      id: msg.id
    });
    await prev_msg.executing
  }

  // Init pyodide
  if (self.pyodide == null) {
    self.postMessage({
      type: 'loading',
      msg: 'Loading pyodide',
      id: msg.id
    });
    await loadApplication(msg.id, msg.path)
    self.postMessage({
      type: 'loaded',
      id: msg.id
    });
  }

  // Handle message
  if (!MESSAGES.hasOwnProperty(msg.type)) {
    console.warn(`Service worker received unknown message type '${msg.type}'.`)
    resolveExecution()
    self.postMessage({
      type: 'idle',
      id: msg.id
    });
    return
  }

  if (msg.type === 'execute') {
    let deps
    try {
      self.pyodide.globals.set('msg', msg)
      deps = self.pyodide.runPython(autodetect_deps_code)
    } catch(e) {
      deps = '[]'
      console.warn(`Auto-detection of dependencies failed with error: ${e}`)
    }
    for (const pkg of JSON.parse(deps)) {
      self.postMessage({
        type: 'loading',
        msg: `Loading ${pkg}`,
        id: msg.id
      });
      try {
        await self.pyodide.runPythonAsync(`await micropip.install('${pkg}', keep_going=True)`)
      } catch(e) {
        console.log(`Auto-detected dependency ${pkg} could not be installed.`)
      }
    }
  }

  try {
    self.pyodide.globals.set('msg', msg)
    let out = await self.pyodide.runPythonAsync(MESSAGES[msg.type])
    resolveExecution()
    if (out == null) {
      out = new Map()
    }
    if (out.has('content')) {
      self.postMessage({
        type: 'render',
        id: msg.id,
        content: out.get('content'),
        mime: out.get('mime_type')
      });
    }
    if (out.has('stdout') && out.get('stdout').length) {
      self.postMessage({
        type: 'stdout',
        content: out.get('stdout'),
        id: msg.id
      });
    }
    if (out.has('stderr') && out.get('stderr').length) {
      self.postMessage({
        type: 'stderr',
        content: out.get('stderr'),
        id: msg.id
      });
    }
    self.postMessage({
      type: 'idle',
      id: msg.id,
      uuid: msg.uuid
    });
  } catch (e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'error',
      traceback: traceback,
      msg: tblines[tblines.length-2],
      id: msg.id
    });
    resolveExecution()
    throw(e)
  }
}