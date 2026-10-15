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
