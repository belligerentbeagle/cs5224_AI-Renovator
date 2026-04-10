import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { indexFromUrl, searchProducts, createProject, presignUpload, uploadFileToS3, generateRoom, pollGeneration } from '../api'
import ProductCard from '../components/ProductCard'
import { Product, GenerationDone } from '../types'

const STYLES = ['Modern', 'Scandinavian', 'Cozy Warm', 'Futuristic', 'Nature', 'Industrial']
const STYLE_COLORS: Record<string, string> = {
  Modern: 'bg-stone-100',
  Scandinavian: 'bg-sky-50',
  'Cozy Warm': 'bg-amber-50',
  Futuristic: 'bg-indigo-50',
  Nature: 'bg-green-50',
  Industrial: 'bg-zinc-100',
}

type LeftTab = 'url' | 'browse'

export default function Home() {
  const { token } = useAuth()
  const navigate = useNavigate()

  // ── Furniture selection ────────────────────────────────────────────────────
  const [leftTab, setLeftTab] = useState<LeftTab>('url')
  const [urlInput, setUrlInput] = useState('')
  const [urlLoading, setUrlLoading] = useState(false)
  const [urlError, setUrlError] = useState('')
  const [selectedItems, setSelectedItems] = useState<Product[]>([])
  const [browseResults, setBrowseResults] = useState<Product[]>([])
  const [browseQuery, setBrowseQuery] = useState('')
  const [browseLoading, setBrowseLoading] = useState(false)

  // ── Room upload ────────────────────────────────────────────────────────────
  const [roomFile, setRoomFile] = useState<File | null>(null)
  const [roomPreview, setRoomPreview] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // ── Style ──────────────────────────────────────────────────────────────────
  const [selectedStyle, setSelectedStyle] = useState('Modern')

  // ── Generation ────────────────────────────────────────────────────────────
  const [generating, setGenerating] = useState(false)
  const [genStatus, setGenStatus] = useState('')
  const [genResult, setGenResult] = useState<GenerationDone | null>(null)

  // ── Furniture via URL ──────────────────────────────────────────────────────
  async function handleAddUrl(e: React.FormEvent) {
    e.preventDefault()
    if (!urlInput.trim()) return
    setUrlError('')
    setUrlLoading(true)
    try {
      const product = await indexFromUrl(urlInput.trim())
      addItem(product)
      setUrlInput('')
    } catch (err: unknown) {
      setUrlError(err instanceof Error ? err.message : 'Could not fetch that URL')
    } finally {
      setUrlLoading(false)
    }
  }

  // ── Browse products ────────────────────────────────────────────────────────
  async function handleBrowseSearch(e: React.FormEvent) {
    e.preventDefault()
    setBrowseLoading(true)
    try {
      const results = await searchProducts({ q: browseQuery || selectedStyle })
      setBrowseResults(results)
    } catch {
      setBrowseResults([])
    } finally {
      setBrowseLoading(false)
    }
  }

  function addItem(product: Product) {
    if (selectedItems.length >= 5) return
    if (selectedItems.find((p) => p.product_id === product.product_id)) return
    setSelectedItems((prev) => [...prev, product])
  }

  function removeItem(id: string) {
    setSelectedItems((prev) => prev.filter((p) => p.product_id !== id))
  }

  // ── Room photo ─────────────────────────────────────────────────────────────
  function handleFile(file: File) {
    if (!file.type.startsWith('image/')) return
    setRoomFile(file)
    setRoomPreview(URL.createObjectURL(file))
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [])

  // ── Generate ───────────────────────────────────────────────────────────────
  async function handleGenerate() {
    if (!token) {
      alert('Please sign in first to generate designs.')
      return
    }
    setGenerating(true)
    setGenResult(null)
    setGenStatus('Creating project…')

    try {
      // 1. Create project
      const project = await createProject(token, {
        title: `Room – ${selectedStyle}`,
        style_prompt: selectedStyle,
      })

      // 2. Upload room photo if provided
      let photoId: string | null = null
      if (roomFile) {
        setGenStatus('Uploading room photo…')
        const { photo_id, upload_url } = await presignUpload(
          token, project.project_id, roomFile.name, roomFile.type,
        )
        await uploadFileToS3(upload_url, roomFile)
        photoId = photo_id
      }

      // 3. Kick off generation
      setGenStatus('Searching for furniture…')
      const gen = await generateRoom(token, project.project_id, photoId, selectedStyle)

      // 4. Poll until done
      let attempts = 0
      while (attempts < 30) {
        await new Promise((r) => setTimeout(r, 2000))
        const result = await pollGeneration(token, gen.generation_id)
        if (result.status === 'done') {
          setGenResult(result as GenerationDone)
          setGenStatus('')
          break
        }
        attempts++
        setGenStatus(`Finding matching furniture (${attempts * 2}s)…`)
      }
    } catch (err: unknown) {
      setGenStatus('Generation failed: ' + (err instanceof Error ? err.message : 'unknown error'))
    } finally {
      setGenerating(false)
    }
  }

  const isSelected = (id: string) => selectedItems.some((p) => p.product_id === id)

  return (
    <main className="max-w-7xl mx-auto px-4 py-6">
      {/* Hero title */}
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold text-rs-dark mb-1">Visualize Your Dream Living Room</h1>
        <p className="text-stone-500 text-sm">Upload furniture from affiliates &amp; see how it looks in your space with AI</p>
      </div>

      {/* 3-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.4fr_1fr] gap-4">

        {/* ── LEFT: Add Furniture ───────────────────────────────────────────── */}
        <div className="card p-4 flex flex-col gap-4">
          <h2 className="font-semibold text-sm text-stone-700">
            1. Add Furniture Options (URL/Upload)
          </h2>

          {/* Tab switcher */}
          <div className="flex gap-1 bg-cream rounded-xl p-1">
            {([['url', 'Product Link'], ['browse', 'Browse Catalogue']] as const).map(([tab, label]) => (
              <button
                key={tab}
                onClick={() => setLeftTab(tab)}
                className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  leftTab === tab ? 'bg-white text-rs-dark shadow-sm' : 'text-stone-500 hover:text-stone-700'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {leftTab === 'url' ? (
            <form onSubmit={handleAddUrl} className="flex flex-col gap-2">
              <div className="flex gap-2">
                <input
                  className="input flex-1 text-sm"
                  placeholder="Paste furniture link here…"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  disabled={urlLoading}
                />
                <button
                  type="submit"
                  disabled={urlLoading || selectedItems.length >= 5}
                  className="btn-primary text-sm px-4"
                >
                  {urlLoading ? '…' : 'Add'}
                </button>
              </div>
              {urlError && <p className="text-red-500 text-xs">{urlError}</p>}
              <p className="text-xs text-stone-400">Supports IKEA, Taobao, or any product page</p>
            </form>
          ) : (
            <form onSubmit={handleBrowseSearch} className="flex gap-2">
              <input
                className="input flex-1 text-sm"
                placeholder={`Search (default: ${selectedStyle})`}
                value={browseQuery}
                onChange={(e) => setBrowseQuery(e.target.value)}
              />
              <button type="submit" disabled={browseLoading} className="btn-primary text-sm px-3">
                {browseLoading ? '…' : '🔍'}
              </button>
            </form>
          )}

          {/* Browse results */}
          {leftTab === 'browse' && browseResults.length > 0 && (
            <div className="flex flex-col gap-2 max-h-56 overflow-y-auto">
              {browseResults.slice(0, 10).map((p) => (
                <div key={p.product_id} className="flex items-center gap-2 p-2 rounded-xl border border-rs-border bg-cream/40">
                  {p.image_url
                    ? <img src={p.image_url} className="w-10 h-10 rounded-lg object-cover bg-stone-100 shrink-0" alt="" />
                    : <div className="w-10 h-10 rounded-lg bg-cream flex items-center justify-center text-lg shrink-0">🛋️</div>}
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium truncate">{p.name}</p>
                    <p className="text-xs text-rs-amber">S${p.price.toFixed(2)}</p>
                  </div>
                  <button
                    onClick={() => addItem(p)}
                    disabled={isSelected(p.product_id) || selectedItems.length >= 5}
                    className="btn-primary text-xs py-1 px-2 shrink-0"
                  >
                    {isSelected(p.product_id) ? '✓' : '+'}
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Selected items */}
          <div>
            <p className="text-xs text-stone-500 mb-2">
              Selected Items ({selectedItems.length}/5)
            </p>
            {selectedItems.length === 0 ? (
              <p className="text-xs text-stone-400 italic">No items selected yet</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {selectedItems.map((item) => (
                  <div key={item.product_id} className="relative group">
                    <div className="w-16 h-16 rounded-xl overflow-hidden border border-rs-border bg-cream">
                      {item.image_url
                        ? <img src={item.image_url} className="w-full h-full object-cover" alt={item.name} />
                        : <div className="w-full h-full flex items-center justify-center text-2xl">🛋️</div>}
                    </div>
                    <button
                      onClick={() => removeItem(item.product_id)}
                      className="absolute -top-1.5 -right-1.5 w-4 h-4 bg-stone-700 text-white rounded-full text-xs leading-none hidden group-hover:flex items-center justify-center"
                    >
                      ×
                    </button>
                    <p className="text-xs text-center mt-0.5 w-16 truncate text-stone-600">{item.name}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {selectedItems.length < 5 && (
            <button
              onClick={() => setLeftTab('browse')}
              className="btn-secondary text-xs w-full"
            >
              + Add More Items
            </button>
          )}
        </div>

        {/* ── CENTER: Upload Room + Generate ───────────────────────────────── */}
        <div className="card p-4 flex flex-col gap-4">
          <h2 className="font-semibold text-sm text-stone-700">2. Upload Your Living Room</h2>

          {/* Drop zone */}
          <div
            className={`relative rounded-2xl border-2 border-dashed transition-colors cursor-pointer flex flex-col items-center justify-center min-h-[180px] ${
              dragOver ? 'border-rs-amber bg-amber-50' : 'border-rs-border hover:border-rs-light bg-cream/40'
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            {roomPreview ? (
              <>
                <img src={roomPreview} alt="Room" className="w-full h-48 object-cover rounded-xl" />
                <div className="absolute bottom-2 right-2">
                  <button
                    onClick={(e) => { e.stopPropagation(); setRoomFile(null); setRoomPreview(null) }}
                    className="bg-white text-xs px-2 py-1 rounded-lg border border-rs-border shadow-sm hover:bg-cream"
                  >
                    Change
                  </button>
                </div>
              </>
            ) : (
              <div className="text-center p-6">
                <div className="text-4xl mb-2">📷</div>
                <p className="text-sm font-medium text-stone-600">Upload Room Photo</p>
                <p className="text-xs text-stone-400 mt-1">or drag &amp; drop an image here</p>
                <p className="text-xs text-stone-300 mt-1">JPG, PNG (Max 10MB)</p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
            />
          </div>

          {/* Style picker */}
          <div>
            <p className="text-xs text-stone-500 mb-2">Style</p>
            <div className="flex flex-wrap gap-1.5">
              {STYLES.map((s) => (
                <button
                  key={s}
                  onClick={() => setSelectedStyle(s)}
                  className={`px-3 py-1 rounded-lg text-xs font-medium border transition-all ${
                    selectedStyle === s
                      ? 'border-rs-amber bg-rs-amber text-white'
                      : `border-rs-border ${STYLE_COLORS[s]} text-stone-600 hover:border-rs-light`
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          {/* Generate button */}
          <button
            onClick={handleGenerate}
            disabled={generating}
            className="btn-primary w-full flex items-center justify-center gap-2 py-3"
          >
            <span>✨</span>
            <span>{generating ? genStatus || 'Generating…' : 'Generate AI Visualization'}</span>
            {!generating && <span>→</span>}
          </button>
          <p className="text-xs text-stone-400 text-center -mt-2">
            {token ? 'Estimated time: 30–60 seconds' : 'Sign in to generate designs'}
          </p>

          {/* Generation result */}
          {genResult && (
            <div className="rounded-2xl border border-rs-border bg-cream/40 p-4">
              <p className="text-sm font-semibold text-rs-dark mb-2">
                ✓ Design ready — {genResult.products.length} items found
              </p>
              {genResult.over_budget && (
                <p className="text-xs text-red-500 mb-2">⚠ Over budget</p>
              )}
              <p className="text-xs text-stone-500 mb-3">
                Total: S${genResult.total_cost.toFixed(2)}
              </p>
              <div className="flex flex-col gap-2 max-h-48 overflow-y-auto">
                {genResult.products.map((p) => (
                  <div key={p.product_id} className="flex items-center justify-between gap-2 text-xs">
                    <span className="truncate flex-1">{p.name}</span>
                    <span className="text-rs-amber font-medium shrink-0">S${p.price.toFixed(2)}</span>
                    <a href={p.buy_url} target="_blank" rel="noopener noreferrer"
                      className="text-rs-amber underline shrink-0">Buy</a>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── RIGHT: Inspirations + Style browse ───────────────────────────── */}
        <div className="flex flex-col gap-4">
          <div className="card p-4">
            <h2 className="font-semibold text-sm text-stone-700 mb-3">3. Inspirations</h2>
            <div className="grid grid-cols-2 gap-2">
              {STYLES.map((s) => (
                <button
                  key={s}
                  onClick={() => setSelectedStyle(s)}
                  className={`${STYLE_COLORS[s]} rounded-xl p-3 text-center border transition-all ${
                    selectedStyle === s ? 'border-rs-amber ring-1 ring-rs-amber' : 'border-rs-border hover:border-rs-light'
                  }`}
                >
                  <p className="text-xs font-medium text-stone-700">{s}</p>
                </button>
              ))}
            </div>
          </div>

          <div className="card p-4 flex-1">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-semibold text-sm text-stone-700">4. Quick Browse</h2>
              <button
                onClick={async () => {
                  setBrowseLoading(true)
                  try {
                    const r = await searchProducts({ style: selectedStyle })
                    setBrowseResults(r)
                    setLeftTab('browse')
                  } finally { setBrowseLoading(false) }
                }}
                className="text-xs text-rs-amber hover:text-rs-dark font-medium"
              >
                {browseLoading ? 'Loading…' : `Load ${selectedStyle}`}
              </button>
            </div>
            {browseResults.length > 0 ? (
              <div className="flex flex-col gap-2 max-h-64 overflow-y-auto">
                {browseResults.slice(0, 6).map((p) => (
                  <div key={p.product_id} className="flex items-center gap-2 p-1.5 rounded-xl border border-rs-border bg-cream/30 hover:bg-cream/70 transition-colors">
                    {p.image_url
                      ? <img src={p.image_url} className="w-9 h-9 rounded-lg object-cover bg-stone-100 shrink-0" alt="" />
                      : <div className="w-9 h-9 rounded-lg bg-cream flex items-center justify-center text-base shrink-0">🛋️</div>}
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium truncate">{p.name}</p>
                      <p className="text-xs text-rs-amber">S${p.price.toFixed(2)}</p>
                    </div>
                    <button
                      onClick={() => addItem(p)}
                      disabled={isSelected(p.product_id) || selectedItems.length >= 5}
                      className="btn-primary text-xs py-0.5 px-2 shrink-0"
                    >
                      {isSelected(p.product_id) ? '✓' : '+'}
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-stone-400 italic">Click "Load {selectedStyle}" to browse items</p>
            )}
          </div>
        </div>

      </div>
    </main>
  )
}
