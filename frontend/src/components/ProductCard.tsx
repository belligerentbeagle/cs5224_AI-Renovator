import { Product } from '../types'

interface Props {
  product: Product
  onAdd?: (product: Product) => void
  selected?: boolean
}

const SOURCE_BADGE: Record<string, string> = {
  ikea:    'bg-blue-50 text-blue-600',
  taobao:  'bg-red-50 text-red-600',
  scraped: 'bg-stone-100 text-stone-500',
}

export default function ProductCard({ product, onAdd, selected }: Props) {
  return (
    <div className={`card overflow-hidden transition-all ${selected ? 'ring-2 ring-rs-amber' : ''}`}>
      {product.image_url ? (
        <img
          src={product.image_url}
          alt={product.name}
          className="w-full h-36 object-cover bg-stone-100"
          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
        />
      ) : (
        <div className="w-full h-36 bg-cream flex items-center justify-center text-4xl">🛋️</div>
      )}

      <div className="p-3">
        <div className="flex items-start justify-between gap-1 mb-1">
          <p className="text-sm font-medium leading-snug line-clamp-2 flex-1">{product.name}</p>
          <span className={`text-xs px-1.5 py-0.5 rounded-md font-medium shrink-0 ${SOURCE_BADGE[product.source] ?? SOURCE_BADGE.scraped}`}>
            {product.source}
          </span>
        </div>

        <p className="text-rs-amber font-semibold text-sm mb-2">
          S${product.price.toFixed(2)}
        </p>

        <div className="flex gap-1.5">
          {onAdd && (
            <button
              onClick={() => onAdd(product)}
              disabled={selected}
              className="btn-primary text-xs py-1 px-3 flex-1"
            >
              {selected ? '✓ Added' : '+ Select'}
            </button>
          )}
          {product.buy_url && (
            <a
              href={product.buy_url}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary text-xs py-1 px-2"
            >
              View
            </a>
          )}
        </div>
      </div>
    </div>
  )
}
