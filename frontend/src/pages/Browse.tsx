import { useState, useEffect } from 'react'
import { searchProducts } from '../api'
import ProductCard from '../components/ProductCard'
import { Product } from '../types'

const STYLES = ['', 'Modern', 'Scandinavian', 'Cozy Warm', 'Futuristic', 'Nature', 'Industrial']

export default function Browse() {
  const [query, setQuery] = useState('')
  const [style, setStyle] = useState('')
  const [minPrice, setMinPrice] = useState('')
  const [maxPrice, setMaxPrice] = useState('')
  const [results, setResults] = useState<Product[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function search() {
    setLoading(true)
    setError('')
    try {
      const r = await searchProducts({
        q: query || undefined,
        style: style || undefined,
        min_price: minPrice ? Number(minPrice) : undefined,
        max_price: maxPrice ? Number(maxPrice) : undefined,
      })
      setResults(r)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  // Load default results on mount
  useEffect(() => { search() }, [])

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    search()
  }

  return (
    <main className="max-w-7xl mx-auto px-4 py-6">
      <h1 className="text-2xl font-bold text-rs-dark mb-5">Browse Furniture</h1>

      {/* Filters */}
      <form onSubmit={handleSubmit} className="card p-4 mb-6 flex flex-wrap gap-3 items-end">
        <div className="flex-1 min-w-48">
          <label className="text-xs text-stone-500 block mb-1">Search</label>
          <input
            className="input"
            placeholder="sofa, lamp, table…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>

        <div className="min-w-36">
          <label className="text-xs text-stone-500 block mb-1">Style</label>
          <select
            className="input"
            value={style}
            onChange={(e) => setStyle(e.target.value)}
          >
            {STYLES.map((s) => (
              <option key={s} value={s}>{s || 'Any style'}</option>
            ))}
          </select>
        </div>

        <div className="min-w-24">
          <label className="text-xs text-stone-500 block mb-1">Min S$</label>
          <input
            className="input"
            type="number"
            min="0"
            placeholder="0"
            value={minPrice}
            onChange={(e) => setMinPrice(e.target.value)}
          />
        </div>

        <div className="min-w-24">
          <label className="text-xs text-stone-500 block mb-1">Max S$</label>
          <input
            className="input"
            type="number"
            min="0"
            placeholder="∞"
            value={maxPrice}
            onChange={(e) => setMaxPrice(e.target.value)}
          />
        </div>

        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>

      {/* Results */}
      {error && <p className="text-red-500 text-sm mb-4">{error}</p>}

      {loading ? (
        <div className="flex items-center justify-center py-20 text-stone-400">
          <span className="animate-spin mr-2">⏳</span> Searching IKEA catalogue…
        </div>
      ) : results.length === 0 ? (
        <div className="text-center py-20 text-stone-400">
          <p className="text-3xl mb-2">🛋️</p>
          <p>No results found. Try a different search.</p>
        </div>
      ) : (
        <>
          <p className="text-sm text-stone-500 mb-3">{results.length} items found</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
            {results.map((p) => (
              <ProductCard key={p.product_id} product={p} />
            ))}
          </div>
        </>
      )}
    </main>
  )
}
