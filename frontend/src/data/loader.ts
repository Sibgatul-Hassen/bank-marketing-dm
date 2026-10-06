import { useEffect, useState } from 'react'

const cache = new Map<string, Promise<unknown>>()

export class NotBuiltError extends Error {
  path: string
  constructor(path: string) { super(`not built yet: ${path}`); this.path = path }
}

/** Fetch an artifact under /results. 404 → NotBuiltError (rendered as a "not built yet" placeholder). */
export function loadArtifact<T>(path: string): Promise<T> {
  const url = `/results/${path}`
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then(async (r) => {
      if (r.status === 404) throw new NotBuiltError(path)
      if (!r.ok) throw new Error(`${r.status} ${url}`)
      const text = await r.text()
      try { return JSON.parse(text) } catch { throw new NotBuiltError(path) }
    }).catch((e) => { cache.delete(url); throw e }))
  }
  return cache.get(url) as Promise<T>
}

export interface AsyncState<T> { data?: T; error?: Error; loading: boolean }

export function useArtifact<T>(path: string | null | undefined): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ loading: !!path })
  useEffect(() => {
    if (!path) { setState({ loading: false }); return }
    let alive = true
    setState((s) => ({ ...s, loading: true }))
    loadArtifact<T>(path).then(
      (data) => alive && setState({ data, loading: false }),
      (error) => alive && setState({ error, loading: false }),
    )
    return () => { alive = false }
  }, [path])
  return state
}

/** Load several artifacts at once; missing ones resolve to undefined rather than failing the group. */
export function useArtifacts<T>(paths: string[]): { data: (T | undefined)[]; loading: boolean } {
  const key = paths.join('|')
  const [state, setState] = useState<{ data: (T | undefined)[]; loading: boolean }>({ data: [], loading: true })
  useEffect(() => {
    let alive = true
    Promise.all(paths.map((p) => loadArtifact<T>(p).catch(() => undefined))).then((data) => alive && setState({ data, loading: false }))
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return state
}
