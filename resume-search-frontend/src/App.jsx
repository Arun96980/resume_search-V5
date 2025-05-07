// src/App.jsx
import { useState, useEffect } from 'react'
import axios from 'axios'
import toast, { Toaster } from 'react-hot-toast'
import { motion } from 'framer-motion'
import DOMPurify from 'dompurify'
import AnalysisCard from './components/AnalysisCard'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [topK, setTopK] = useState(5)
  const [rerank, setRerank] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [darkMode, setDarkMode] = useState(() => JSON.parse(localStorage.getItem('darkMode')) || false)
  const [searchTimeout, setSearchTimeout] = useState(null)

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode))
    document.documentElement.classList.toggle('dark', darkMode)
  }, [darkMode])

  const Spinner = () => (
    <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
    </svg>
  )

  // Parse raw AI justification text into structured props
  const parseAnalysis = (text) => {
    const lines = text.split('\n').filter(l => l.trim())
    let direct = '', conf = '', explanation = ''
    const skills = []
    lines.forEach(line => {
      const lower = line.toLowerCase()
      if (lower.includes('direct relevance:')) {
        direct = line.split(':')[1]?.trim() || ''
      } else if (lower.includes('matching skills:')) {
        // extract quoted skills
        const matches = [...line.matchAll(/"([^"]+)"/g)].map(m => m[1])
        skills.push(...matches)
      } else if (lower.includes('confidence:')) {
        conf = line.split(/confidence:/i)[1]?.trim() || ''
      } else {
        explanation += line.trim() + ' '
      }
    })
    return {
      directRelevance: direct || 'N/A',
      matchingSkills: skills,
      confidence: conf || 'Unknown',
      explanation: explanation.trim() || null
    }
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    if (searchTimeout) clearTimeout(searchTimeout)
    setSearchTimeout(setTimeout(async () => {
      setLoading(true)
      setError('')
      try {
        const response = await axios.post(
          `${API_URL}/search`,
          { query: query.trim(), top_k: topK, rerank },
          { headers: { 'X-API-Key': import.meta.env.VITE_GOOGLE_API_KEY } }
        )
        setResults(response.data)
      } catch (err) {
        setError(err.response?.data?.detail || 'Search failed. Please try again.')
      } finally {
        setLoading(false)
      }
    }, 500))
  }

  const sendFeedback = async (sentenceHash, isRelevant) => {
    try {
      await axios.post(
        `${API_URL}/feedback`,
        { query, sentence_hash: sentenceHash, is_relevant: isRelevant }
      )
      toast.success(isRelevant ? 'Marked Relevant' : 'Marked Not Relevant')
    } catch {
      toast.error('Feedback failed')
    }
  }

  const handlePdfOpen = (source) => async (e) => {
    e.preventDefault()
    if (!source) return toast.error('No PDF associated')
    try {
      const url = `${API_URL}/resumes/${encodeURIComponent(source)}`
      await axios.head(url)
      window.open(url, '_blank', 'noopener')
    } catch {
      toast.error('PDF not found')
    }
  }

  const AccessibleButton = ({ children, onClick, ariaLabel, ...props }) => (
    <button onClick={onClick} aria-label={ariaLabel} {...props}>{children}</button>
  )

  const cardVariants = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } }

  return (
    <div className={`min-h-screen transition-colors ${darkMode ? 'bg-gray-900 text-white' : 'bg-white text-gray-900'}`}>
      <Toaster position="top-right" />
      <div className="container mx-auto p-6">
        <header className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Resume Analyzer
          </h1>
          <AccessibleButton
            onClick={() => setDarkMode(!darkMode)}
            ariaLabel="Toggle dark mode"
            className="px-4 py-2 border rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition"
          >{darkMode ? '🌞 Light' : '🌙 Dark'}</AccessibleButton>
        </header>

        <form onSubmit={handleSearch} className="space-y-4 mb-6">
          <div className="flex gap-3">
            <input
              type="text" value={query} onChange={e => setQuery(e.target.value)}
              placeholder="Search for skills or job requirements..."
              className="flex-1 p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-800"
            />
            <AccessibleButton
              type="submit" disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-3 rounded-lg flex items-center gap-2"
            >{loading ? <Spinner /> : '🔍 Search'}</AccessibleButton>
          </div>
          <div className="flex gap-6 items-center">
            <label className="flex items-center gap-2">
              <span className="text-sm">Results:</span>
              <input type="number" min="1" value={topK} onChange={e => setTopK(Math.max(1, +e.target.value))}
                className="w-16 p-2 border rounded-lg dark:bg-gray-800" />
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={rerank} onChange={e => setRerank(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:bg-gray-800" />
              <span className="text-sm">AI Reranking</span>
            </label>
          </div>
        </form>

        {error && <div className="text-red-500 bg-red-100 p-3 rounded mb-4">{error}</div>}
        {loading && [...Array(3)].map((_,i) => <div key={i} className="animate-pulse h-24 bg-gray-200 dark:bg-gray-700 rounded-lg mb-4" />)}

        {!loading && results.length > 0 && (
          <motion.div className="space-y-6" initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.1 } } }}>
            {results.map(r => {
              const analysis = r.justification ? parseAnalysis(r.justification) : {}
              return (
                <motion.div key={r.sentence_hash} variants={cardVariants} transition={{ duration: 0.3 }}
                  className="group bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition border border-gray-100 dark:border-gray-700">

                  {/* PDF link & score & feedback row */}
                  <div className="flex justify-between items-center mb-4">
                    <div className="flex items-center gap-2">
                      <AccessibleButton onClick={handlePdfOpen(r.source)} ariaLabel="Open PDF"
                        className="text-blue-600 dark:text-blue-400 hover:underline">
                        {r.source?.replace(/\.[^/.]+$/, '') || 'Document'}
                      </AccessibleButton>
                      {r.score != null && (
                        <span className="text-sm font-medium text-blue-600 dark:text-blue-400">
                          Score: {r.score.toFixed(2)}
                        </span>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <AccessibleButton onClick={() => sendFeedback(r.sentence_hash, true)}
                        className="px-3 py-1 border border-green-500 text-green-600 rounded-full">
                        Relevant
                      </AccessibleButton>
                      <AccessibleButton onClick={() => sendFeedback(r.sentence_hash, false)}
                        className="px-3 py-1 border border-red-500 text-red-600 rounded-full">
                        Not Relevant
                      </AccessibleButton>
                    </div>
                  </div>

                  {/* Analysis Card */}
                  <AnalysisCard
                    directRelevance={analysis.directRelevance}
                    matchingSkills={analysis.matchingSkills}
                    confidence={analysis.confidence}
                    explanation={analysis.explanation}
                  />
                </motion.div>
              )
            })}
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default App
