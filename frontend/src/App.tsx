import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import ModelPage from './pages/ModelPage'
import StudentProfile from './pages/StudentProfile'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/students/:id" element={<StudentProfile />} />
        <Route path="/model" element={<ModelPage />} />
      </Routes>
    </BrowserRouter>
  )
}
