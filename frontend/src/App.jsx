import React, { useState } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

/* ─── helpers ─────────────────────────────────────────────── */
const priorityColor = (p) =>
  ({ HIGH: '#16a34a', MEDIUM: '#d97706', LOW: '#dc2626' }[p] ?? '#6b7280');

const scoreColor = (s) =>
  s >= 0.65 ? '#16a34a' : s >= 0.40 ? '#d97706' : '#dc2626';

const confidenceCls = (c) =>
  ({ high: 'badge-green', medium: 'badge-yellow', low: 'badge-red' }[
    c?.toLowerCase()
  ] ?? 'badge-red');

const riskColor = (r) =>
  ({ LOW: '#16a34a', MEDIUM: '#d97706', HIGH: '#dc2626' }[r] ?? '#6b7280');

const pct = (n) => `${Math.round((n ?? 0) * 100)}%`;
const fmt = (n, d = 3) => (n ?? 0).toFixed(d);

/* ─── sub-components ──────────────────────────────────────── */

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

function MetricBox({ label, value }) {
  return (
    <div className="metric-box">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

function Badge({ children, cls }) {
  return <span className={`badge ${cls}`}>{children}</span>;
}

function GeneBadge({ gene }) {
  return <span className="gene-badge">{gene}</span>;
}

function PathwayBadge({ pathway }) {
  return <span className="pathway-badge">{pathway}</span>;
}

/* ─── Regimen Card ────────────────────────────────────────── */
function RegimenCard({ regimen, rank }) {
  const [open, setOpen] = useState(false);
  const td = regimen.trial_detail ?? {};

  return (
    <div className="drug-card">
      <div className="card-top">
        <div className="card-left">
          <div className="card-title-row">
            <span className="rank-num">#{rank}</span>
            <h3 className="drug-name">{regimen.regimen?.toUpperCase()}</h3>
            {regimen.is_synergistic && (
              <Badge cls="badge-green">SYNERGISTIC</Badge>
            )}
            {regimen.n_drugs === 3 && (
              <Badge cls="badge-blue">TRIPLE</Badge>
            )}
          </div>
          <p className="drug-sub">
            {'>'} MECHANISMS: {[regimen.mechanism_a, regimen.mechanism_b, regimen.mechanism_c]
              .filter(Boolean)
              .join(' + ') || 'N/A'}
          </p>
        </div>

        <div className="score-display">
          <div className="score-label">COMBO SCORE</div>
          <div
            className="score-value"
            style={{ color: scoreColor(regimen.combo_score) }}
          >
            {pct(regimen.combo_score)}
          </div>
        </div>
      </div>

      {/* key trial metrics */}
      <div className="metrics-row">
        <MetricBox label="EST. ORR" value={pct(regimen.orr_estimate)} />
        <MetricBox label="PFS-6" value={pct(regimen.pfs6_estimate)} />
        <MetricBox label="P2 PROB" value={fmt(regimen.p2_probability, 2)} />
        <MetricBox label="GENE COV" value={regimen.combined_gene_coverage ?? 0} />
        <MetricBox
          label="PRIORITY"
          value={
            <span style={{ color: priorityColor(regimen.priority), fontWeight: 900 }}>
              {regimen.priority}
            </span>
          }
        />
      </div>

      {regimen.pag_one_liner && (
        <p className="one-liner">{'>'} {regimen.pag_one_liner}</p>
      )}

      {regimen.shared_genes?.length > 0 && (
        <div className="tags-row">
          <span className="tags-label">TARGET GENES:</span>
          {regimen.shared_genes.slice(0, 8).map((g) => (
            <GeneBadge key={g} gene={g} />
          ))}
        </div>
      )}

      {/* expandable trial detail */}
      {Object.keys(td).length > 0 && (
        <>
          <button className="expand-btn" onClick={() => setOpen((o) => !o)}>
            {open ? '▲ HIDE TRIAL DETAIL' : '▼ SHOW TRIAL DETAIL'}
          </button>
          {open && (
            <div className="trial-detail">
              <div className="metrics-row">
                <MetricBox
                  label="DCR"
                  value={pct(td.dcr)}
                />
                <MetricBox
                  label="MEDIAN PFS (wk)"
                  value={fmt(td.median_pfs_weeks, 1)}
                />
                <MetricBox
                  label="PATIENTS"
                  value={td.n_patients ?? 200}
                />
                <MetricBox
                  label="NET EFFECT"
                  value={fmt(td.network_effect, 3)}
                />
              </div>
              {td.orr_ci_90?.length === 2 && (
                <p className="ci-text">
                  {'>'} ORR 90% CI: {pct(td.orr_ci_90[0])} – {pct(td.orr_ci_90[1])}
                </p>
              )}
              {td.recommendation && (
                <p className="recommendation">{'>'} {td.recommendation}</p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ─── Single Drug Card ────────────────────────────────────── */
function SingleDrugCard({ candidate, rank, diseaseName, onValidate, validation, validating }) {
  const [open, setOpen] = useState(false);
  const score = candidate.composite_score ?? candidate.score ?? 0;

  return (
    <div className="drug-card">
      <div className="card-top">
        <div className="card-left">
          <div className="card-title-row">
            <span className="rank-num">#{rank}</span>
            <h3 className="drug-name">{candidate.drug_name?.toUpperCase()}</h3>
            <Badge cls={confidenceCls(candidate.confidence)}>
              {(candidate.confidence ?? 'LOW').toUpperCase()}
            </Badge>
          </div>
          <p className="drug-sub">{'>'} {candidate.indication || 'Unknown indication'}</p>
        </div>
        <div className="score-display">
          <div className="score-label">MATCH SCORE</div>
          <div className="score-value" style={{ color: scoreColor(score) }}>
            {pct(score)}
          </div>
        </div>
      </div>

      {candidate.mechanism && (
        <div className="info-block">
          <span className="info-label">{'>'} MECHANISM:</span>
          <span className="info-text">{candidate.mechanism}</span>
        </div>
      )}

      <div className="metrics-row">
        <MetricBox label="GENE SCORE" value={pct(candidate.gene_score)} />
        <MetricBox label="PATHWAY" value={pct(candidate.pathway_score)} />
        <MetricBox label="PPI" value={pct(candidate.ppi_score)} />
        <MetricBox label="SIMILARITY" value={pct(candidate.similarity_score)} />
        <MetricBox label="SHARED GENES" value={candidate.shared_genes?.length ?? 0} />
      </div>

      {candidate.shared_genes?.length > 0 && (
        <div className="tags-row">
          <span className="tags-label">SHARED GENES:</span>
          {candidate.shared_genes.slice(0, 10).map((g) => (
            <GeneBadge key={g} gene={g} />
          ))}
        </div>
      )}

      {candidate.shared_pathways?.length > 0 && (
        <div className="tags-row">
          <span className="tags-label">PATHWAYS:</span>
          {candidate.shared_pathways.slice(0, 5).map((p) => (
            <PathwayBadge key={p} pathway={p} />
          ))}
        </div>
      )}

      {/* clinical validation */}
      <div className="validate-section">
        {!validation && (
          <button
            className="terminal-button"
            onClick={() => onValidate(candidate)}
            disabled={validating}
          >
            {validating ? (
              <span className="flex-center gap-2">
                <span className="loader" /> VALIDATING...
              </span>
            ) : (
              '🔬 VALIDATE CLINICALLY'
            )}
          </button>
        )}

        {validation && !validation.error && (
          <div className="validation-result">
            <div
              className="risk-bar"
              style={{ borderColor: riskColor(validation.risk_level) }}
            >
              <span className="risk-label">RISK LEVEL:</span>
              <span style={{ color: riskColor(validation.risk_level), fontWeight: 900 }}>
                {validation.risk_level}
              </span>
            </div>
            <p className="recommendation">{validation.recommendation}</p>
            <div className="evidence-grid">
              {validation.evidence_summary?.map((item, i) => (
                <p key={i} className="evidence-item">• {item}</p>
              ))}
            </div>
            {/* trial detail */}
            {validation.clinical_trials && (
              <div className="mini-block">
                <span className="info-label">📋 TRIALS:</span>
                <span className="info-text">{validation.clinical_trials.summary}</span>
              </div>
            )}
            {validation.literature_evidence && (
              <div className="mini-block">
                <span className="info-label">📚 LITERATURE:</span>
                <span className="info-text">{validation.literature_evidence.summary}</span>
              </div>
            )}
          </div>
        )}

        {validation?.error && (
          <p className="error-text">❌ {validation.error}</p>
        )}
      </div>

      {/* score breakdown toggle */}
      <button className="expand-btn" onClick={() => setOpen((o) => !o)}>
        {open ? '▲ HIDE SCORE BREAKDOWN' : '▼ SCORE BREAKDOWN'}
      </button>
      {open && (
        <div className="trial-detail">
          {candidate.explanation && Array.isArray(candidate.explanation)
            ? candidate.explanation.map((line, i) => (
                <p key={i} className="evidence-item">{'>'} {line}</p>
              ))
            : candidate.explanation && (
                <p className="evidence-item">{'>'} {candidate.explanation}</p>
              )}
        </div>
      )}
    </div>
  );
}

/* ─── Main App ────────────────────────────────────────────── */
export default function App() {
  const [diseaseName, setDiseaseName] = useState('');
  const [maxRegimens, setMaxRegimens] = useState(10);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState('combos'); // 'combos' | 'singles' | 'wetlab'
  const [validations, setValidations] = useState({});
  const [validatingKey, setValidatingKey] = useState(null);

  /* ── submit ── */
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);
    setValidations({});
    setTab('combos');

    try {
      const res = await fetch(`${API_BASE}/treatment_plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          disease_name: diseaseName,
          max_regimens: maxRegimens,
          include_triples: true,
          fetch_ppi: true,
          fetch_similarity: true,
          use_tissue: true,
        }),
      });
      const data = await res.json();

      if (!data.success) {
        setError(data.error || 'Unknown error — check the backend logs.');
      } else {
        setResults(data);
      }
    } catch (err) {
      setError(`Cannot reach backend at ${API_BASE}. Is it running?`);
    } finally {
      setLoading(false);
    }
  };

  /* ── clinical validate ── */
  const handleValidate = async (candidate) => {
    const key = candidate.drug_name;
    setValidatingKey(key);
    try {
      const res = await fetch(`${API_BASE}/validate_clinical`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          drug_name: candidate.drug_name,
          // results.disease is a plain string in the treatment_plan response
          disease_name: results.disease,
          drug_data: { mechanism: candidate.mechanism, indication: candidate.indication },
          disease_data: { name: results.disease },
        }),
      });
      const data = await res.json();
      setValidations((v) => ({
        ...v,
        [key]: data.success ? data.validation : { error: data.error ?? 'Validation failed' },
      }));
    } catch (err) {
      setValidations((v) => ({ ...v, [key]: { error: 'Connection failed' } }));
    } finally {
      setValidatingKey(null);
    }
  };

  /* ── derived ── */
  const header = results?.header ?? {};
  const regimens = results?.ranked_regimens ?? [];
  const singles = results?.candidates ?? [];
  const wetLab = results?.wet_lab_brief ?? {};
  const stats = results?.pipeline_stats ?? {};

  return (
    <div className="min-h-screen relative">
      <div className="graph-paper-bg" />

      <div className="container mx-auto px-4 py-8 max-w-7xl relative z-10">

        {/* ── Header ── */}
        <div className="text-center mb-12">
          <h1 className="glitch-text text-6xl font-black mb-3">🧬 NAVARA AI</h1>
          <p className="font-mono text-lg tracking-widest">
            {'>'} AI-POWERED THERAPEUTIC DISCOVERY SYSTEM {'<'}
          </p>
          <div className="flex justify-center gap-4 flex-wrap mt-4">
            {['DATABASES: ONLINE', 'AI: ACTIVE', 'PIPELINE: READY'].map((s) => (
              <div key={s} className="status-indicator">
                <span className="status-dot" />
                <span className="font-mono text-sm font-bold">{s}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Query Form ── */}
        <div className="terminal-window mb-8">
          <div className="terminal-header">
            <div className="flex gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <div className="w-3 h-3 rounded-full bg-green-500" />
            </div>
            <span className="font-mono text-sm">QUERY_INTERFACE.EXE</span>
          </div>
          <div className="terminal-body">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block font-mono text-sm font-bold mb-2 tracking-widest">
                  {'>'} TARGET_DISEASE:
                </label>
                <input
                  type="text"
                  value={diseaseName}
                  onChange={(e) => setDiseaseName(e.target.value)}
                  placeholder="e.g. pulmonary arterial hypertension"
                  className="terminal-input"
                  required
                />
              </div>
              <div>
                <label className="block font-mono text-sm font-bold mb-2 tracking-widest">
                  {'>'} MAX_REGIMENS (1–30):
                </label>
                <input
                  type="number"
                  value={maxRegimens}
                  onChange={(e) => setMaxRegimens(Number(e.target.value))}
                  min="1"
                  max="30"
                  className="terminal-input"
                />
              </div>
              <button type="submit" disabled={loading} className="terminal-button">
                {loading ? (
                  <span className="flex-center gap-2">
                    <span className="loader" /> ANALYSING — THIS CAN TAKE 1–2 MIN...
                  </span>
                ) : (
                  '⚡ INITIATE REPURPOSING ANALYSIS'
                )}
              </button>
            </form>

            {error && (
              <div className="mt-6 p-4 border-2 border-red-500 bg-red-50">
                <p className="font-mono text-sm font-bold text-red-600">❌ ERROR: {error}</p>
              </div>
            )}
          </div>
        </div>

        {/* ── Results ── */}
        {results && (
          <div className="space-y-8">

            {/* Disease summary */}
            <div className="terminal-window">
              <div className="terminal-header">
                <span className="font-mono text-sm">DISEASE_ANALYSIS.DAT</span>
              </div>
              <div className="terminal-body">
                {/* results.disease is a plain string */}
                <h2 className="glitch-text text-4xl font-black mb-6 font-mono">
                  {(results.disease ?? diseaseName).toUpperCase()}
                </h2>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <StatCard
                    label="DISEASE GENES"
                    value={header.disease_genes_count ?? '—'}
                  />
                  <StatCard
                    label="PATHWAYS"
                    value={header.disease_pathways_count ?? '—'}
                  />
                  <StatCard
                    label="DRUGS SCREENED"
                    value={stats.total_drugs_evaluated ?? header.drug_pool?.final_pool ?? '—'}
                  />
                  <StatCard
                    label="REGIMENS RANKED"
                    value={regimens.length}
                  />
                </div>

                {/* pipeline stats */}
                {Object.keys(stats).length > 0 && (
                  <div className="info-block">
                    <span className="info-label">{'>'} PIPELINE:</span>
                    <span className="info-text">
                      {stats.after_generic_filter ?? '—'} generics →{' '}
                      {stats.after_safety_filter ?? '—'} safe →{' '}
                      {stats.combos_generated ?? '—'} combos →{' '}
                      {stats.trials_run ?? '—'} trials run
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 flex-wrap">
              {[
                { id: 'combos', label: `COMBINATION REGIMENS (${regimens.length})` },
                { id: 'singles', label: `TOP SINGLE DRUGS (${singles.length})` },
                { id: 'wetlab', label: 'WET LAB BRIEF' },
                { id: 'briefs', label: 'PAG + BIOTECH BRIEFS' },
              ].map((t) => (
                <button
                  key={t.id}
                  className={`tab-btn ${tab === t.id ? 'tab-active' : ''}`}
                  onClick={() => setTab(t.id)}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {/* ── Tab: Combo Regimens ── */}
            {tab === 'combos' && (
              <div className="terminal-window">
                <div className="terminal-header">
                  <span className="font-mono text-sm">RANKED_REGIMENS.DAT</span>
                </div>
                <div className="terminal-body space-y-6">
                  {regimens.length === 0 ? (
                    <p className="font-mono text-yellow-600 font-bold">
                      ⚠️ No regimens generated. Try a different disease name.
                    </p>
                  ) : (
                    regimens.map((r, i) => (
                      <RegimenCard key={r.regimen} regimen={r} rank={i + 1} />
                    ))
                  )}
                </div>
              </div>
            )}

            {/* ── Tab: Single Drugs ── */}
            {tab === 'singles' && (
              <div className="terminal-window">
                <div className="terminal-header">
                  <span className="font-mono text-sm">SINGLE_DRUG_CANDIDATES.DAT</span>
                </div>
                <div className="terminal-body space-y-6">
                  {singles.length === 0 ? (
                    <p className="font-mono text-yellow-600 font-bold">
                      ⚠️ No single-drug candidates available.
                    </p>
                  ) : (
                    singles.map((c, i) => (
                      <SingleDrugCard
                        key={c.drug_name}
                        candidate={c}
                        rank={i + 1}
                        diseaseName={results.disease}
                        onValidate={handleValidate}
                        validation={validations[c.drug_name]}
                        validating={validatingKey === c.drug_name}
                      />
                    ))
                  )}
                </div>
              </div>
            )}

            {/* ── Tab: Wet Lab Brief ── */}
            {tab === 'wetlab' && (
              <div className="terminal-window">
                <div className="terminal-header">
                  <span className="font-mono text-sm">WET_LAB_BRIEF.DAT</span>
                </div>
                <div className="terminal-body space-y-6">
                  {wetLab.priority_targets?.length > 0 && (
                    <div>
                      <p className="font-mono text-sm font-bold mb-3 tracking-widest">
                        {'>'} PRIORITY TARGET GENES:
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {wetLab.priority_targets.map((g) => (
                          <GeneBadge key={g} gene={g} />
                        ))}
                      </div>
                    </div>
                  )}
                  {wetLab.rationale && (
                    <div className="info-block">
                      <span className="info-label">{'>'} RATIONALE:</span>
                      <p className="info-text mt-1">{wetLab.rationale}</p>
                    </div>
                  )}
                  {wetLab.suggested_assays?.length > 0 && (
                    <div>
                      <p className="font-mono text-sm font-bold mb-2 tracking-widest">
                        {'>'} SUGGESTED ASSAYS:
                      </p>
                      {wetLab.suggested_assays.map((a, i) => (
                        <p key={i} className="evidence-item">• {a}</p>
                      ))}
                    </div>
                  )}
                  {wetLab.university_partner_note && (
                    <div className="info-block">
                      <span className="info-label">{'>'} PARTNER NOTE:</span>
                      <p className="info-text mt-1">{wetLab.university_partner_note}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* ── Tab: Briefs ── */}
            {tab === 'briefs' && (
              <div className="space-y-6">
                {results.pag_brief && (
                  <div className="terminal-window">
                    <div className="terminal-header">
                      <span className="font-mono text-sm">PAG_BRIEF.TXT</span>
                    </div>
                    <div className="terminal-body">
                      <pre className="font-mono text-sm whitespace-pre-wrap leading-relaxed">
                        {results.pag_brief}
                      </pre>
                    </div>
                  </div>
                )}
                {results.biotech_brief && (
                  <div className="terminal-window">
                    <div className="terminal-header">
                      <span className="font-mono text-sm">BIOTECH_BRIEF.TXT</span>
                    </div>
                    <div className="terminal-body">
                      <pre className="font-mono text-sm whitespace-pre-wrap leading-relaxed">
                        {results.biotech_brief}
                      </pre>
                    </div>
                  </div>
                )}
                {results.limitations?.length > 0 && (
                  <div className="terminal-window">
                    <div className="terminal-header">
                      <span className="font-mono text-sm">LIMITATIONS.TXT</span>
                    </div>
                    <div className="terminal-body">
                      {results.limitations.map((l, i) => (
                        <p key={i} className="evidence-item mb-2">⚠️ {l}</p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="text-center py-8 relative z-10">
        <p className="font-mono text-sm font-bold tracking-widest">
          POWERED BY: OpenTargets · ChEMBL · DGIdb · STRING · Reactome · KEGG · ClinicalTrials.gov
        </p>
      </div>
    </div>
  );
}