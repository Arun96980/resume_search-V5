// src/components/AnalysisCard.jsx
import React from 'react'
import { Sparkles } from 'lucide-react'

export default function AnalysisCard({
  directRelevance,
  matchingSkills = [],
  confidence,
  explanation,
}) {
  return (
    <div className="bg-slate-900 text-white p-4 rounded-xl shadow-lg border border-slate-700 space-y-4">
      <div className="flex items-center gap-2 text-indigo-400 font-semibold text-lg">
        <Sparkles className="h-5 w-5" />
        AI Analysis
      </div>

      {/* Direct Relevance */}
      <div className="bg-slate-800 p-3 rounded-lg">
        <h4 className="text-green-400 font-bold mb-1">Direct Relevance</h4>
        <p className="text-gray-300">{directRelevance}</p>
      </div>

      {/* Matching Skills */}
      <div className="bg-slate-800 p-3 rounded-lg">
        <h4 className="text-cyan-400 font-bold mb-1">Matching Skills</h4>
        {matchingSkills.length > 0 ? (
          <ul className="list-disc list-inside text-gray-300 space-y-1">
            {matchingSkills.map((skill, i) => <li key={i}>{skill}</li>)}
          </ul>
        ) : (
          <p className="text-gray-500 italic">No matching skills found</p>
        )}
      </div>

      {/* Confidence */}
      <div className="bg-slate-800 p-3 rounded-lg">
        <h4 className="text-yellow-400 font-bold mb-1">Confidence</h4>
        <p className="text-gray-300">{confidence}</p>
      </div>

      {/* Explanation */}
      {explanation && (
        <div className="text-sm text-gray-400">
          {explanation}
        </div>
      )}
    </div>
  )
}
