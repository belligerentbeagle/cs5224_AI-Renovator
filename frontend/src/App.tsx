import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import Header from './components/Header'
import Home from './pages/Home'
import Browse from './pages/Browse'
import Projects from './pages/Projects'

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <div className="min-h-screen flex flex-col">
          <Header />
          <div className="flex-1">
            <Routes>
              <Route path="/"         element={<Home />} />
              <Route path="/browse"   element={<Browse />} />
              <Route path="/projects" element={<Projects />} />
            </Routes>
          </div>
          <footer className="text-center text-xs text-stone-400 py-4 border-t border-rs-border">
            RoomStyle AI — Visualise your dream space
          </footer>
        </div>
      </BrowserRouter>
    </AuthProvider>
  )
}
