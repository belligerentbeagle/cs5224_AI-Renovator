import { createContext, useContext, useState, ReactNode } from 'react'

interface AuthState {
  token: string | null
  userId: string | null
}

interface AuthContextValue extends AuthState {
  setAuth: (token: string, userId: string) => void
  clearAuth: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

const STORAGE_KEY = 'rs_auth'

function loadStored(): AuthState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return JSON.parse(raw)
  } catch {}
  return { token: null, userId: null }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuthState] = useState<AuthState>(loadStored)

  const setAuth = (token: string, userId: string) => {
    const next = { token, userId }
    setAuthState(next)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  }

  const clearAuth = () => {
    setAuthState({ token: null, userId: null })
    localStorage.removeItem(STORAGE_KEY)
  }

  return (
    <AuthContext.Provider value={{ ...auth, setAuth, clearAuth }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be inside AuthProvider')
  return ctx
}
