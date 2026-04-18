import { useState } from 'react'
import { NavLink, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import AuthModal from './AuthModal'
import { LogIn, LogOut, Sparkles } from 'lucide-react'

export default function Header() {
  const { token, clearAuth } = useAuth()
  const [showAuth, setShowAuth] = useState(false)

  return (
    <>
      <header className="sticky top-0 z-40 w-full border-b border-rs-border/70 bg-white/88 backdrop-blur-xl supports-[backdrop-filter]:bg-white/78">
        <div className="mx-auto grid h-20 w-full max-w-[1520px] grid-cols-3 items-center px-6 lg:px-8">
          <Link to="/" className="flex items-center gap-3 group min-w-0">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-rs-dark text-white shadow-premium transition-transform duration-300 group-hover:scale-[1.03]">
              <Sparkles size={21} />
            </div>
            <div className="flex min-w-0 flex-col">
              <span className="text-[10px] font-semibold uppercase tracking-[0.34em] text-rs-amber/90">Premium Interior AI</span>
              <span className="truncate font-serif text-[1.35rem] font-semibold tracking-[0.01em] text-rs-dark">RoomStyle AI</span>
            </div>
          </Link>

          <nav className="hidden md:flex justify-center items-center gap-1 rounded-full border border-rs-border/80 bg-[#F8F5F1]/90 p-1.5 shadow-[0_10px_30px_-22px_rgba(74,63,53,0.35)]">
            <NavLink
              to="/"
              className={({ isActive }) => `nav-link ${isActive ? 'nav-link-active' : 'nav-link-inactive'}`}
            >
              Design
            </NavLink>
            <NavLink
              to="/browse"
              className={({ isActive }) => `nav-link ${isActive ? 'nav-link-active' : 'nav-link-inactive'}`}
            >
              Browse
            </NavLink>
            <NavLink
              to="/projects"
              className={({ isActive }) => `nav-link ${isActive ? 'nav-link-active' : 'nav-link-inactive'}`}
            >
              My Projects
            </NavLink>
          </nav>

          <div className="flex items-center justify-end gap-3">
            {token ? (
              <button
                onClick={clearAuth}
                className="btn-secondary !px-4 !py-2.5 text-sm"
              >
                <LogOut size={16} />
                <span>Sign Out</span>
              </button>
            ) : (
              <button
                onClick={() => setShowAuth(true)}
                className="btn-primary !px-5 !py-2.5 text-sm"
              >
                <LogIn size={16} />
                <span>Sign In</span>
              </button>
            )}
          </div>
        </div>
      </header>

      {showAuth && <AuthModal onClose={() => setShowAuth(false)} />}
    </>
  )
}
