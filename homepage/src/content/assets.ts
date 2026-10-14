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
