export interface Product {
  product_id: string
  name: string
  price: number
  source: 'ikea' | 'taobao' | 'scraped'
  image_url?: string
  buy_url?: string
  in_stock: boolean
  style_tags: string[]
  scraped?: boolean
}

export interface AuthResponse {
  user_id: string
  token: string
}

export interface Project {
  project_id: string
  name: string
  budget_limit?: number
  created_at: string
  photo_ids: string[]
  generation_ids: string[]
}

export interface GenerationPending {
  generation_id: string
  status: 'pending'
}

export interface GeneratedProduct {
  product_id: string
  name: string
  price: number
  source: string
  buy_url: string
}

export interface GenerationDone {
  generation_id: string
  status: 'done'
  image_url?: string
  over_budget: boolean
  total_cost: number
  products: GeneratedProduct[]
}

export type GenerationResult = GenerationPending | GenerationDone
