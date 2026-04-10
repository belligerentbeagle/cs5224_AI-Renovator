import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import AuthModal from './AuthModal'

export default function Header() {
  const { token, clearAuth } = useAuth()
  const [showAuth, setShowAuth] = useState(false)

  return (
    <>
      <header className="bg-white border-b border-rs-border sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 h-14 flex items-center gap-6">
          {/* Logo */}
          <NavLink to="/" className="flex items-center gap-2 font-semibold text-rs-dark shrink-0">
            <span className="text-xl">🛋️</span>
            <span>RoomStyle AI</span>
          </NavLink>

          {/* Nav */}
          <nav className="flex items-center gap-1 flex-1">
            {[
              { to: '/',        label: 'Home' },
              { to: '/browse',  label: 'Browse Furniture' },
              { to: '/projects', label: 'My Projects' },
            ].map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-cream text-rs-dark'
                      : 'text-stone-500 hover:text-stone-800 hover:bg-cream/60'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>

          {/* Auth */}
          {token ? (
            <button onClick={clearAuth} className="btn-secondary text-sm">Sign out</button>
          ) : (
            <button onClick={() => setShowAuth(true)} className="btn-secondary text-sm">Sign in</button>
          )}
        </div>
      </header>

      {showAuth && <AuthModal onClose={() => setShowAuth(false)} />}
    </>
  )
}
