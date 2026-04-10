import React, { useState } from 'react';
import mockData from './mock_results.json';

const API_BASE = 'http://localhost:8000';
const USE_MOCK = true;

/* ─── helpers ─────────────────────────────────────────────── */
const pct = (n) => `${Math.round((n ?? 0) * 100)}%`;

const noveltyScore = (r) => {
  const base = 1 - (r.combo_score ?? 0);
  const boost = r.is_synergistic ? 0.12 : 0;
  const tripleBoost = r.n_drugs === 3 ? 0.08 : 0;
  return Math.min(1, base + boost + tripleBoost);
};

const successRate = (r) => r.orr_estimate ?? r.combo_score ?? 0;

/* ─── styles ──────────────────────────────────────────────── */
const styles = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --ink: #0f0e0d;
    --ink-soft: #4a4743;
    --ink-faint: #9c9894;
    --paper: #f7f5f2;
    --paper-warm: #ede9e2;
    --paper-card: #ffffff;
    --accent: #c8502a;
    --accent-light: #f0e8e4;
    --success: #2a7a4b;
    --success-light: #e4f0e8;
    --mid: #7a6e2a;
    --mid-light: #f0ece4;
    --rule: #e0dbd4;
    --shadow: 0 1px 3px rgba(15,14,13,0.06), 0 4px 16px rgba(15,14,13,0.05);
    --shadow-hover: 0 2px 8px rgba(15,14,13,0.1), 0 8px 32px rgba(15,14,13,0.09);
    --radius: 3px;
    --font-display: 'DM Serif Display', Georgia, serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
  }

  body {
    background: var(--paper);
    color: var(--ink);
    font-family: var(--font-body);
    font-weight: 300;
    min-height: 100vh;
  }

  .app {
    max-width: 780px;
    margin: 0 auto;
    padding: 64px 24px 120px;
  }

  .header { margin-bottom: 56px; }

  .header-eyebrow {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 16px;
  }

  .header-title {
    font-family: var(--font-display);
    font-size: clamp(40px, 7vw, 62px);
    line-height: 1.05;
    color: var(--ink);
    margin-bottom: 14px;
  }

  .header-title em { color: var(--accent); font-style: italic; }

  .header-sub {
    font-size: 15px;
    color: var(--ink-soft);
    font-weight: 300;
    line-height: 1.6;
    max-width: 480px;
  }

  .search-block { margin-bottom: 56px; }

  .search-label {
    display: block;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 10px;
  }

  .search-row { display: flex; gap: 12px; align-items: stretch; }

  .search-input {
    flex: 1;
    background: var(--paper-card);
    border: 1.5px solid var(--rule);
    border-radius: var(--radius);
    padding: 14px 18px;
    font-family: var(--font-body);
    font-size: 16px;
    font-weight: 300;
    color: var(--ink);
    outline: none;
    transition: border-color 0.2s;
  }

  .search-input::placeholder { color: var(--ink-faint); }
  .search-input:focus { border-color: var(--accent); }

  .search-btn {
    background: var(--ink);
    color: var(--paper);
    border: none;
    border-radius: var(--radius);
    padding: 14px 28px;
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.2s, transform 0.1s;
  }

  .search-btn:hover:not(:disabled) { background: var(--accent); }
  .search-btn:active:not(:disabled) { transform: scale(0.98); }
  .search-btn:disabled { opacity: 0.45; cursor: not-allowed; }

  .error-bar {
    margin-bottom: 32px;
    padding: 12px 16px;
    background: #fef2f2;
    border-left: 3px solid #dc2626;
    font-family: var(--font-mono);
    font-size: 13px;
    color: #dc2626;
    border-radius: 0 var(--radius) var(--radius) 0;
  }

  .results-header {
    border-top: 1.5px solid var(--ink);
    padding-top: 28px;
    margin-bottom: 36px;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }

  .results-title { font-family: var(--font-display); font-size: 28px; color: var(--ink); }
  .results-meta { font-family: var(--font-mono); font-size: 11px; color: var(--ink-faint); letter-spacing: 0.1em; }

  .regimen-list { display: flex; flex-direction: column; gap: 2px; }

  .regimen-row {
    background: var(--paper-card);
    border: 1.5px solid var(--rule);
    border-radius: var(--radius);
    overflow: hidden;
    transition: box-shadow 0.2s, border-color 0.2s;
    cursor: pointer;
  }

  .regimen-row:hover { box-shadow: var(--shadow-hover); border-color: #ccc8c2; }

  .regimen-main {
    display: flex;
    align-items: center;
    padding: 18px 22px;
    gap: 16px;
  }

  .regimen-rank {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--ink-faint);
    min-width: 28px;
    flex-shrink: 0;
    align-self: flex-start;
    padding-top: 3px;
  }

  .regimen-name-block { flex: 1; min-width: 0; }

  .regimen-name {
    font-family: var(--font-display);
    font-size: 18px;
    color: var(--ink);
    margin-bottom: 2px;
    /* No truncation — allow wrap */
    white-space: normal;
    word-break: break-word;
  }

  /* Drug list for multi-drug combos */
  .regimen-drug-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 6px;
  }

  .drug-pill {
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 2px 8px;
    border: 1px solid var(--rule);
    border-radius: 20px;
    color: var(--ink-soft);
    background: var(--paper-warm);
    white-space: nowrap;
  }

  .regimen-mechanism {
    font-size: 12px;
    color: var(--ink-faint);
    font-weight: 300;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 2px;
  }

  .regimen-tags { display: flex; gap: 6px; flex-shrink: 0; align-self: flex-start; padding-top: 2px; }

  .tag {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    padding: 3px 8px;
    border-radius: 2px;
    font-weight: 500;
  }

  .tag-synergy { background: var(--success-light); color: var(--success); }
  .tag-triple  { background: #e8eaf0; color: #3a4270; }
  .tag-high    { background: var(--success-light); color: var(--success); }
  .tag-medium  { background: var(--mid-light); color: var(--mid); }
  .tag-low     { background: #fce8e4; color: var(--accent); }

  .regimen-meters { display: flex; gap: 20px; flex-shrink: 0; align-self: flex-start; padding-top: 2px; }

  .meter { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; min-width: 80px; }

  .meter-labels { display: flex; justify-content: space-between; width: 100%; }

  .meter-title {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-faint);
  }

  .meter-value { font-family: var(--font-mono); font-size: 13px; font-weight: 500; color: var(--ink); }

  .bar-track { width: 80px; height: 3px; background: var(--rule); border-radius: 2px; overflow: hidden; }

  .bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s cubic-bezier(0.16,1,0.3,1);
  }

  .bar-success { background: var(--success); }
  .bar-novelty { background: var(--accent); }

  .chevron { color: var(--ink-faint); font-size: 11px; flex-shrink: 0; transition: transform 0.2s; margin-left: 4px; align-self: flex-start; padding-top: 5px; }
  .chevron.open { transform: rotate(180deg); }

  .regimen-detail {
    border-top: 1px solid var(--rule);
    padding: 20px 22px 20px 66px;
    background: #faf9f7;
    display: flex;
    flex-direction: column;
    gap: 14px;
    animation: slideDown 0.18s ease;
  }

  @keyframes slideDown {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .detail-row { display: flex; gap: 32px; flex-wrap: wrap; }

  .detail-stat { display: flex; flex-direction: column; gap: 2px; }

  .detail-stat-label {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-faint);
  }

  .detail-stat-value { font-family: var(--font-mono); font-size: 15px; font-weight: 500; color: var(--ink); }

  .detail-one-liner {
    font-size: 13px;
    color: var(--ink-soft);
    font-weight: 300;
    line-height: 1.6;
    border-left: 2px solid var(--accent);
    padding-left: 12px;
    border-radius: 0;
  }

  /* ── AI rationale ── */
  .rationale-block {
    border-left: 2px solid #b8d4b8;
    padding-left: 12px;
    border-radius: 0;
  }

  .rationale-label {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--success);
    margin-bottom: 5px;
  }

  .rationale-text {
    font-size: 13px;
    color: var(--ink-soft);
    font-weight: 300;
    line-height: 1.7;
  }

  .rationale-loading {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .rationale-dots { display: flex; gap: 4px; }

  .r-dot {
    width: 4px;
    height: 4px;
    background: var(--success);
    border-radius: 50%;
    animation: pulse 1.2s ease-in-out infinite;
    opacity: 0.5;
  }
  .r-dot:nth-child(2) { animation-delay: 0.2s; }
  .r-dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes pulse {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.85); }
    40% { opacity: 0.8; transform: scale(1); }
  }

  .rationale-loading-text {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--ink-faint);
    letter-spacing: 0.1em;
  }

  .detail-genes { display: flex; gap: 6px; flex-wrap: wrap; }

  .gene-chip {
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 3px 8px;
    background: var(--paper-warm);
    border-radius: 2px;
    color: var(--ink-soft);
    border: 1px solid var(--rule);
  }

  .loading-state { padding: 48px 0; display: flex; align-items: center; gap: 16px; }

  .loading-dots { display: flex; gap: 6px; }

  .dot {
    width: 5px;
    height: 5px;
    background: var(--ink);
    border-radius: 50%;
    animation: pulse 1.2s ease-in-out infinite;
  }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }

  .loading-text {
    font-family: var(--font-mono);
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-faint);
  }

  .footer {
    margin-top: 80px;
    padding-top: 24px;
    border-top: 1px solid var(--rule);
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--ink-faint);
    letter-spacing: 0.1em;
    line-height: 1.8;
  }

  @media (max-width: 600px) {
    .regimen-meters { display: none; }
    .regimen-main { padding: 14px 16px; gap: 10px; }
    .regimen-detail { padding-left: 16px; }
    .search-row { flex-direction: column; }
  }
`;

/* ─── AI Rationale fetcher ────────────────────────────────── */
/* ─── Update this function in your App.jsx ─── */

async function fetchRationale(regimen, disease) {
  if (USE_MOCK) {
    return "Mock rationale: This combination targets complementary pathways, providing synergistic efficacy with reduced resistance potential.";
  }

  const mechanisms = [regimen.mechanism_a, regimen.mechanism_b, regimen.mechanism_c]
    .filter(Boolean).join(', ');

  const response = await fetch(`${API_BASE}/generate_rationale`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ regimen: regimen.regimen, disease, mechanisms }),
  });

  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  return data.rationale ?? 'Rationale unavailable.';
}

/* ─── parse drug names from regimen string ─────────────────── */
function parseDrugNames(regimenStr) {
  if (!regimenStr) return [];
  // Common delimiters: +, /, " and ", " with "
  return regimenStr
    .split(/\s*[\+\/]\s*|\s+and\s+|\s+with\s+/i)
    .map(d => d.trim())
    .filter(Boolean);
}

/* ─── RegimenRow ──────────────────────────────────────────── */
function RegimenRow({ regimen, rank, disease }) {
  const [open, setOpen] = useState(false);
  const [rationale, setRationale] = useState(null);
  const [rationaleLoading, setRationaleLoading] = useState(false);

  const sr = successRate(regimen);
  const novelty = noveltyScore(regimen);
  const td = regimen.trial_detail ?? {};
  const mechanisms = [regimen.mechanism_a, regimen.mechanism_b, regimen.mechanism_c]
    .filter(Boolean).join(' · ');

  const drugNames = parseDrugNames(regimen.regimen);
  const isMultiDrug = drugNames.length > 2;

  const priorityTag = (p) => {
    const cls = { HIGH: 'tag-high', MEDIUM: 'tag-medium', LOW: 'tag-low' }[p] ?? 'tag-low';
    return <span className={`tag ${cls}`}>{p}</span>;
  };

  const handleToggle = async () => {
    const nextOpen = !open;
    setOpen(nextOpen);
    if (nextOpen && !rationale && !rationaleLoading) {
      setRationaleLoading(true);
      try {
        const text = await fetchRationale(regimen, disease);
        setRationale(text);
      } catch {
        setRationale('Could not generate rationale at this time.');
      } finally {
        setRationaleLoading(false);
      }
    }
  };

  return (
    <div className="regimen-row">
      <div className="regimen-main" onClick={handleToggle}>
        {/* rank */}
        <span className="regimen-rank">{String(rank).padStart(2, '0')}</span>

        {/* name + drugs + mechanism */}
        <div className="regimen-name-block">
          <div className="regimen-name">{regimen.regimen}</div>
          {isMultiDrug && (
            <div className="regimen-drug-list">
              {drugNames.map((d, i) => (
                <span key={i} className="drug-pill">{d}</span>
              ))}
            </div>
          )}
          {mechanisms && <div className="regimen-mechanism">{mechanisms}</div>}
        </div>

        {/* tags */}
        <div className="regimen-tags">
          {regimen.is_synergistic && <span className="tag tag-synergy">Synergy</span>}
          {regimen.n_drugs === 3 && <span className="tag tag-triple">Triple</span>}
          {regimen.priority && priorityTag(regimen.priority)}
        </div>

        {/* meters */}
        <div className="regimen-meters">
          <div className="meter">
            <div className="meter-labels">
              <span className="meter-title">Success</span>
              <span className="meter-value">{pct(sr)}</span>
            </div>
            <div className="bar-track">
              <div className="bar-fill bar-success" style={{ width: pct(sr) }} />
            </div>
          </div>
          <div className="meter">
            <div className="meter-labels">
              <span className="meter-title">Novelty</span>
              <span className="meter-value">{pct(novelty)}</span>
            </div>
            <div className="bar-track">
              <div className="bar-fill bar-novelty" style={{ width: pct(novelty) }} />
            </div>
          </div>
        </div>

        <span className={`chevron ${open ? 'open' : ''}`}>▾</span>
      </div>

      {open && (
        <div className="regimen-detail">
          {/* stats row */}
          <div className="detail-row">
            {regimen.combo_score != null && (
              <div className="detail-stat">
                <span className="detail-stat-label">Combo Score</span>
                <span className="detail-stat-value">{pct(regimen.combo_score)}</span>
              </div>
            )}
            {regimen.pfs6_estimate != null && (
              <div className="detail-stat">
                <span className="detail-stat-label">PFS-6</span>
                <span className="detail-stat-value">{pct(regimen.pfs6_estimate)}</span>
              </div>
            )}
            {regimen.p2_probability != null && (
              <div className="detail-stat">
                <span className="detail-stat-label">Ph.2 Prob.</span>
                <span className="detail-stat-value">{regimen.p2_probability?.toFixed(2)}</span>
              </div>
            )}
            {regimen.combined_gene_coverage != null && (
              <div className="detail-stat">
                <span className="detail-stat-label">Gene Coverage</span>
                <span className="detail-stat-value">{regimen.combined_gene_coverage}</span>
              </div>
            )}
            {td.dcr != null && (
              <div className="detail-stat">
                <span className="detail-stat-label">DCR</span>
                <span className="detail-stat-value">{pct(td.dcr)}</span>
              </div>
            )}
            {td.median_pfs_weeks != null && (
              <div className="detail-stat">
                <span className="detail-stat-label">Median PFS</span>
                <span className="detail-stat-value">{td.median_pfs_weeks?.toFixed(1)} wk</span>
              </div>
            )}
          </div>

          {/* existing one liner */}
          {regimen.pag_one_liner && (
            <p className="detail-one-liner">{regimen.pag_one_liner}</p>
          )}

          {/* AI rationale */}
          <div className="rationale-block">
            <div className="rationale-label">AI Rationale</div>
            {rationaleLoading ? (
              <div className="rationale-loading">
                <div className="rationale-dots">
                  <div className="r-dot" /><div className="r-dot" /><div className="r-dot" />
                </div>
                <span className="rationale-loading-text">Generating rationale...</span>
              </div>
            ) : (
              <p className="rationale-text">{rationale}</p>
            )}
          </div>

          {/* genes */}
          {regimen.shared_genes?.length > 0 && (
            <div className="detail-genes">
              {regimen.shared_genes.slice(0, 12).map(g => (
                <span key={g} className="gene-chip">{g}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── App ─────────────────────────────────────────────────── */
export default function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResults(null);

    if (USE_MOCK) {
      setTimeout(() => { setResults(mockData); setLoading(false); }, 900);
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/treatment_plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          disease_name: query,
          max_regimens: 15,
          include_triples: true,
          fetch_ppi: true,
          fetch_similarity: true,
          use_tissue: true,
        }),
      });
      const data = await res.json();
      if (!data.success) setError(data.error || 'Unknown error.');
      else setResults(data);
    } catch {
      setError(`Cannot reach backend at ${API_BASE}.`);
    } finally {
      setLoading(false);
    }
  };

  // Sort by success rate descending
  const regimens = [...(results?.ranked_regimens ?? [])].sort(
    (a, b) => successRate(b) - successRate(a)
  );
  const diseaseName = results?.disease ?? query;

  return (
    <>
      <style>{styles}</style>
      <div className="app">

        <header className="header">
          <p className="header-eyebrow">POPPY v1</p>
          <h1 className="header-title">
            Drug repurposing,<br /><em>reimagined.</em>
          </h1>
          <p className="header-sub">
            Enter a disease to surface ranked combination regimens, scored by success rate and therapeutic novelty.
          </p>
        </header>

        <div className="search-block">
          <label className="search-label" htmlFor="disease-input">Target disease</label>
          <form className="search-row" onSubmit={handleSubmit}>
            <input
              id="disease-input"
              type="text"
              className="search-input"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="e.g. pulmonary arterial hypertension"
              autoComplete="off"
              spellCheck="false"
            />
            <button type="submit" className="search-btn" disabled={loading || !query.trim()}>
              {loading ? 'Analysing...' : 'Analyse →'}
            </button>
          </form>
        </div>

        {error && <div className="error-bar">Error: {error}</div>}

        {loading && (
          <div className="loading-state">
            <div className="loading-dots">
              <div className="dot" /><div className="dot" /><div className="dot" />
            </div>
            <span className="loading-text">Screening compounds · Building regimens · Scoring</span>
          </div>
        )}

        {results && !loading && (
          <>
            <div className="results-header">
              <h2 className="results-title">{diseaseName}</h2>
              <span className="results-meta">{regimens.length} regimens · ranked by success rate</span>
            </div>

            <div className="regimen-list">
              {regimens.map((r, i) => (
                <RegimenRow key={r.regimen} regimen={r} rank={i + 1} disease={diseaseName} />
              ))}
            </div>
          </>
        )}

        <footer className="footer">
          Data sources: OpenTargets · ChEMBL · DGIdb · STRING · Reactome · KEGG · ClinicalTrials.gov
        </footer>
      </div>
    </>
  );
}
