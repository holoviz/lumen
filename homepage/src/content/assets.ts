/**
 * Screenshots converted from docs/assets to webp. Importing them rather than serving them from
 * public/ puts them under the hashed /_home/ prefix, so they never collide with docs paths.
 */
const files = import.meta.glob<string>('../assets/**/*.webp', {eager: true, query: '?url', import: 'default'})

export function asset(path: string): string {
  const url = files[`../assets/${path}`]
  // Fail the build rather than ship a blank tile when a name and a file drift apart.
  if (!url) throw new Error(`No picture at src/assets/${path}`)
  return url
}

const clips = import.meta.glob<string>('../assets/**/*.{mp4,webm}', {eager: true, query: '?url', import: 'default'})

/** A screen recording shipped as MP4 and WebM, which share one name. */
export function video(name: string): {mp4: string; webm: string} {
  const mp4 = clips[`../assets/${name}.mp4`]
  const webm = clips[`../assets/${name}.webm`]
  if (!mp4 || !webm) throw new Error(`Need both src/assets/${name}.mp4 and .webm`)
  return {mp4, webm}
}

const charts = import.meta.glob<string>('../assets/hero/*.svg', {eager: true, query: '?raw', import: 'default'})

/** A chart rendered from a real Vega-Lite spec, as inline SVG markup. */
export function chart(name: string): string {
  const svg = charts[`../assets/hero/${name}.svg`]
  if (!svg) throw new Error(`No chart at src/assets/hero/${name}.svg; run scripts/hero_charts.py`)
  return svg
}

const specs = import.meta.glob<Record<string, unknown>>('../assets/hero/*.json', {eager: true, import: 'default'})

/** The finished Vega-Lite spec behind a hero chart, data included, for the browser to embed. */
export function chartSpec(name: string): Record<string, unknown> {
  const spec = specs[`../assets/hero/${name}.json`]
  if (!spec) throw new Error(`No spec at src/assets/hero/${name}.json; run scripts/hero_charts.py`)
  return spec
}
