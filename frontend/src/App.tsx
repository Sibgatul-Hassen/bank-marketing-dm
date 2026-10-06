import { MotionConfig } from 'framer-motion'
import { BrowserRouter, Route, Routes, useParams } from 'react-router-dom'
import { ExplainPanel } from './components/ExplainPanel'
import { Layout } from './components/Layout'
import { ComparisonPage } from './pages/Comparison'
import { DataPage, EDAPage } from './pages/DataPages'
import { Home } from './pages/Home'
import { ModelPage } from './pages/ModelPage'

function ModelRoute() {
  const { id = '' } = useParams()
  return <Layout explain={<ExplainPanel modelId={id} />}><ModelPage /></Layout>
}

function NotFound() {
  return <Layout><h1>Page not found</h1><p className="secondary" style={{ marginTop: 8 }}>Use the navigation on the left.</p></Layout>
}

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout><Home /></Layout>} />
          <Route path="/data" element={<Layout><DataPage /></Layout>} />
          <Route path="/eda" element={<Layout><EDAPage /></Layout>} />
          <Route path="/models/:id" element={<ModelRoute />} />
          <Route path="/compare" element={<Layout><ComparisonPage /></Layout>} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </MotionConfig>
  )
}
