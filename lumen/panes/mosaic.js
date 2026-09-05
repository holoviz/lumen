import {InstantiateContext, astToDOM, parseSpec} from "@uwdata/mosaic-spec"
import {tableFromIPC} from "@uwdata/flechette"

/**
 * Decode a base64 string into the bytes of an Arrow IPC stream.
 *
 * Panel's ESM bridge carries custom messages as JSON with no binary channel,
 * so query results arrive base64-encoded rather than as a separate buffer.
 */
function decodeBase64(b64) {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes
}

/**
 * A Param is a Selection when it can produce a SQL predicate.
 *
 * Duck-typed rather than using mosaic-core's `isSelection`, so this keeps
 * working even when the Selection was built by a different copy of
 * mosaic-core than the one an `instanceof` check would compare against.
 */
function isSelection(param) {
  return typeof param?.predicate === "function"
}

/**
 * Reduce a value to something the Bokeh protocol can serialize.
 *
 * A Param's value can be `undefined` (an unset selection) or carry
 * non-serializable internals; either would fail the sync to Python and leave
 * `params` stale, so anything that will not survive JSON becomes null.
 */
function jsonSafe(value) {
  try {
    return JSON.parse(JSON.stringify(value ?? null))
  } catch {
    return null
  }
}

function predicateSQL(selection) {
  const predicate = selection.predicate(undefined) ?? []
  const parts = (Array.isArray(predicate) ? predicate : [predicate]).map(String)
  return parts.length > 1 ? parts.map((s) => `(${s})`).join(" AND ") : parts[0] ?? ""
}

export function render({model, el}) {
  el.classList.add("mosaic-pane")

  // Each pane builds its own Mosaic context instead of the module-wide
  // `coordinator()` singleton, so several panes can share a page without the
  // last one rendered stealing the others' database connector.
  const ctx = new InstantiateContext()
  const {coordinator} = ctx

  const pending = new Map()
  let counter = 0

  // Mosaic's browser runtime pushes its SQL down to the database rather than
  // shipping rows to the client; every query is forwarded to Python, which
  // answers over `msg:custom` keyed by `uuid`.
  coordinator.databaseConnector({
    query(query) {
      return new Promise((resolve, reject) => {
        const uuid = `${++counter}`
        pending.set(uuid, {resolve, reject})
        model.send_msg({...query, uuid})
      })
    },
  })

  model.on("msg:custom", (msg) => {
    const query = pending.get(msg.uuid)
    if (query === undefined) {
      return
    }
    pending.delete(msg.uuid)
    if (msg.error) {
      query.reject(new Error(msg.error))
    } else if (msg.type === "arrow") {
      query.resolve(tableFromIPC(decodeBase64(msg.data), {useDate: true}))
    } else if (msg.type === "json") {
      query.resolve(msg.result)
    } else {
      query.resolve({})
    }
  })

  let applied = null

  async function updateSpec() {
    const spec = model.spec
    const json = JSON.stringify(spec)
    if (json === applied) {
      return
    }
    applied = json
    coordinator.clear()
    if (spec == null || Object.keys(spec).length === 0) {
      el.replaceChildren()
      return
    }

    let dom
    try {
      dom = await astToDOM(parseSpec(spec), {api: ctx.api})
    } catch (e) {
      // Surface the failure in the pane; a spec that only errors in the
      // console looks identical to a chart that legitimately drew nothing.
      const error = document.createElement("pre")
      error.className = "mosaic-pane-error"
      error.textContent = `Could not render Mosaic spec:\n${e.message ?? e}`
      el.replaceChildren(error)
      throw e
    }
    el.replaceChildren(dom.element)

    // Mirror every Param/Selection back to Python so `Mosaic.params` tracks the
    // chart's live interaction state, including each selection's SQL predicate.
    let params = {}
    const snapshot = (param, value) => ({
      value: jsonSafe(value),
      ...(isSelection(param) ? {predicate: predicateSQL(param)} : {}),
    })
    const publish = () => {
      model.params = params
    }
    for (const [name, param] of dom.params) {
      params[name] = snapshot(param, param.value)
      param.addEventListener("value", (value) => {
        params = {...params, [name]: snapshot(param, value)}
        publish()
      })
    }
    publish()
  }

  function configureCoordinator() {
    // Empty leaves Mosaic's own default schema in place.
    const schema = model.preagg_schema
    if (schema) {
      coordinator.preaggregator.schema = schema
    }
  }

  model.on("spec", () => updateSpec())
  model.on("preagg_schema", () => configureCoordinator())
  configureCoordinator()
  updateSpec()

  return () => coordinator.clear()
}

export default {render}
