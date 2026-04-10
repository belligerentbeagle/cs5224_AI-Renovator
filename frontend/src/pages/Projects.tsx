import { useState, useEffect } from 'react'
import { listProjects } from '../api'
import { useAuth } from '../context/AuthContext'
import { Project } from '../types'
import { Link } from 'react-router-dom'

export default function Projects() {
  const { token } = useAuth()
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!token) return
    setLoading(true)
    listProjects(token)
      .then(setProjects)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [token])

  if (!token) {
    return (
      <main className="max-w-2xl mx-auto px-4 py-20 text-center">
        <p className="text-4xl mb-3">🔒</p>
        <p className="text-stone-500">Sign in to view your projects.</p>
      </main>
    )
  }

  return (
    <main className="max-w-5xl mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-5">
        <h1 className="text-2xl font-bold text-rs-dark">My Projects</h1>
        <Link to="/" className="btn-primary text-sm">+ New Design</Link>
      </div>

      {loading && (
        <div className="text-center py-20 text-stone-400">Loading…</div>
      )}
      {error && <p className="text-red-500 text-sm">{error}</p>}

      {!loading && projects.length === 0 && (
        <div className="text-center py-20 text-stone-400">
          <p className="text-4xl mb-3">🏠</p>
          <p className="mb-4">No projects yet. Start your first design!</p>
          <Link to="/" className="btn-primary">Create Design</Link>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {projects.map((p) => (
          <div key={p.project_id} className="card p-4 hover:shadow-md transition-shadow">
            <div className="w-full h-28 bg-cream rounded-xl flex items-center justify-center text-4xl mb-3">🛋️</div>
            <h3 className="font-semibold text-sm truncate">{p.name}</h3>
            <p className="text-xs text-stone-400 mt-0.5">
              {new Date(p.created_at).toLocaleDateString('en-SG', {
                day: 'numeric', month: 'short', year: 'numeric',
              })}
            </p>
            <div className="flex gap-3 mt-2 text-xs text-stone-500">
              <span>{p.photo_ids.length} photo{p.photo_ids.length !== 1 ? 's' : ''}</span>
              <span>{p.generation_ids.length} generation{p.generation_ids.length !== 1 ? 's' : ''}</span>
              {p.budget_limit && <span>S${p.budget_limit} budget</span>}
            </div>
          </div>
        ))}
      </div>
    </main>
  )
}
