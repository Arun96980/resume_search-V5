import { useState, useEffect } from 'react'
import axios from 'axios'
import toast, { Toaster } from 'react-hot-toast'
import { motion } from 'framer-motion'
import DOMPurify from 'dompurify'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [topK, setTopK] = useState(5)
  const [rerank, setRerank] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [darkMode, setDarkMode] = useState(() => {
    return JSON.parse(localStorage.getItem('darkMode')) || false
  })
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

  const formatAnalysisContent = (text) => {
    const lines = text
      .split('\n')
      .filter(line => line.trim().length > 0)
      .map(line => line
        .replace(/^- \[ \]\s*/, '')
        .replace(/^\d+\.\s*/, '')
        .trim()
      )

    let confidence = ''
    const skills = lines.map(line => {
      const [skillText, conf] = line.split(' Confidence:')
      if (conf) confidence = conf.trim()
      return skillText.replace(/\*\*/g, '')
    })

    let formattedText = skills.join(', ')
    if (confidence) {
      formattedText += ` <span class="text-purple-600 dark:text-purple-400 text-sm ml-2">(Confidence: ${confidence})</span>`
    }

    return formattedText
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    if (searchTimeout) clearTimeout(searchTimeout)
    
    setSearchTimeout(setTimeout(async () => {
      setLoading(true)
      setError('')
      try {
        const response = await axios.post(`${API_URL}/search`, {
          query: query.trim(),
          top_k: topK,
          rerank
        }, {
          headers: {
            'X-API-Key': import.meta.env.VITE_GOOGLE_API_KEY
          }
        })
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
      await axios.post(`${API_URL}/feedback`, {
        query,
        sentence_hash: sentenceHash,
        is_relevant: isRelevant
      })
      toast.success(isRelevant ? 'Marked Relevant' : 'Marked Not Relevant')
    } catch (err) {
      toast.error('Feedback failed')
    }
  }

  const handlePdfOpen = (source) => async (e) => {
    e.preventDefault()
    if (!source) {
      toast.error('No PDF associated with this result')
      return
    }

    try {
      const encodedSource = encodeURIComponent(source)
      const pdfUrl = `${API_URL}/resumes/${encodedSource}`
      
      const response = await axios.head(pdfUrl)
      if (response.status === 200) {
        window.open(pdfUrl, '_blank', 'noopener,noreferrer')
      }
    } catch (error) {
      toast.error('PDF file not found or inaccessible')
    }
  }

  const AccessibleButton = ({ children, onClick, ariaLabel, ...props }) => (
    <button
      onClick={onClick}
      aria-label={ariaLabel}
      {...props}
    >
      {children}
    </button>
  )

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  }

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
            className="px-4 py-2 border rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          >
            {darkMode ? '🌞 Light' : '🌙 Dark'}
          </AccessibleButton>
        </header>

        <form onSubmit={handleSearch} className="space-y-4 mb-6">
          <div className="flex gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for skills or job requirements..."
              className="flex-1 p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-800"
            />
            <AccessibleButton
              type="submit"
              disabled={loading}
              ariaLabel="Perform search"
              className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-3 rounded-lg flex items-center gap-2 transition-colors"
            >
              {loading ? <Spinner /> : '🔍 Search'}
            </AccessibleButton>
          </div>
          <div className="flex gap-6 items-center">
            <label className="flex items-center gap-2">
              <span className="text-sm">Results:</span>
              <input
                type="number"
                min="1"
                value={topK}
                onChange={(e) => setTopK(Math.max(1, Number(e.target.value)))}
                className="w-16 p-2 border rounded-lg dark:bg-gray-800"
              />
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={rerank}
                onChange={(e) => setRerank(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:bg-gray-800"
              />
              <span className="text-sm">AI Reranking</span>
            </label>
          </div>
        </form>

        {error && <div className="text-red-500 bg-red-100 p-3 rounded-lg mb-4">{error}</div>}

        {loading && (
          <div className="space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="animate-pulse h-24 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
            ))}
          </div>
        )}

        {!loading && results.length > 0 && (
          <motion.div 
            className="space-y-6"
            initial="hidden"
            animate="visible"
            variants={{
              visible: { transition: { staggerChildren: 0.1 } }
            }}
          >
            {results.map((r, index) => (
              <motion.div
                key={r.sentence_hash}
                variants={cardVariants}
                transition={{ duration: 0.3 }}
                className="relative group bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow border border-gray-100 dark:border-gray-700"
              >
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
                  <div className="space-y-2">
                    <AccessibleButton
                      onClick={handlePdfOpen(r.source)}
                      ariaLabel="Open PDF document"
                      className="flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <span className="font-medium truncate max-w-[200px]">
                        {r.source?.replace(/\.[^/.]+$/, "") || "Document"}
                      </span>
                    </AccessibleButton>
                    {r.score && (
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-blue-400 to-purple-400" 
                            style={{ width: `${Math.min(r.score * 100, 100)}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium text-blue-600 dark:text-blue-400">
                          {r.score.toFixed(2)}
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="flex gap-2">
                    <AccessibleButton
                      onClick={() => sendFeedback(r.sentence_hash, true)}
                      ariaLabel="Mark as relevant"
                      className="flex items-center gap-1 px-3 py-1 rounded-full border border-green-500/30 hover:border-green-500/60 text-green-600 dark:text-green-400 hover:bg-green-50/50 dark:hover:bg-green-900/20 transition-all"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      <span className="text-sm">Relevant</span>
                    </AccessibleButton>
                    <AccessibleButton
                      onClick={() => sendFeedback(r.sentence_hash, false)}
                      ariaLabel="Mark as not relevant"
                      className="flex items-center gap-1 px-3 py-1 rounded-full border border-red-500/30 hover:border-red-500/60 text-red-600 dark:text-red-400 hover:bg-red-50/50 dark:hover:bg-red-900/20 transition-all"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                      <span className="text-sm">Not Relevant</span>
                    </AccessibleButton>
                  </div>
                </div>

                <div className="mt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <svg className="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    <h3 className="font-semibold text-gray-800 dark:text-gray-200">AI Analysis</h3>
                  </div>
                  
                  {r.justification ? (
                    <motion.div
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="p-4 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100/50 dark:hover:bg-gray-700/50 transition-colors"
                    >
                      <div 
                        className="text-gray-700 dark:text-gray-300 leading-relaxed prose dark:prose-invert"
                        dangerouslySetInnerHTML={{
                          __html: DOMPurify.sanitize(
                            formatAnalysisContent(r.justification)
                              .replace(/\b(\w+)\b/g, '<span class="font-medium text-blue-600 dark:text-blue-400">$1</span>')
                          )
                        }}
                      />
                    </motion.div>
                  ) : (
                    <div className="text-gray-500 dark:text-gray-400 italic p-4 rounded-lg bg-gray-50/50 dark:bg-gray-800/50">
                      No analysis available for this result
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default App