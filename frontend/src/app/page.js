"use client";

import React, { useState, useEffect } from "react";
import EvidenceMap from "./components/EvidenceMap";
import CertaintyCard from "./components/CertaintyCard";
import CaseMode from "./components/CaseMode";
import ModelTrace from "./components/ModelTrace";
import ClaimVerification from "./components/ClaimVerification";

// Cookie helper to read CSRF token
function getCookie(name) {
  if (typeof document === "undefined") return "";
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return "";
}

export default function Home() {
  const [activeTab, setActiveTab] = useState("search");
  
  // Auth state
  const [user, setUser] = useState(null);
  const [email, setEmail] = useState("clinician@indence.org");
  const [password, setPassword] = useState("DoctorPassword1");
  const [loginError, setLoginError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);

  // Search state
  const [query, setQuery] = useState("Is trastuzumab deruxtecan effective in patients with HER2-positive breast cancer?");
  const [studyTypes, setStudyTypes] = useState({
    rct: true,
    meta_analysis: true,
    systematic_review: true,
    guideline: true,
    observational: false
  });
  const [yearFrom, setYearFrom] = useState("2020");
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [results, setResults] = useState(null);
  
  // Hover/Active states
  const [activeClaim, setActiveClaim] = useState(null);
  const [feedbackClaimId, setFeedbackClaimId] = useState(null);
  const [feedbackText, setFeedbackText] = useState("");
  const [feedbackSent, setFeedbackSent] = useState(false);
  
  // Trace audit state
  const [traceDetails, setTraceDetails] = useState(null);
  const [traceLoading, setTraceLoading] = useState(false);

  const [apiBase, setApiBase] = useState(() =>
    typeof window !== "undefined"
      ? `http://${window.location.hostname || "localhost"}:8000/api/v1`
      : "http://localhost:8000/api/v1"
  );
  const [csrfToken, setCsrfToken] = useState("");
  const [expandedContradictions, setExpandedContradictions] = useState({});
  const [highlightSource, setHighlightSource] = useState(null);

  // Fetch CSRF token on mount using dynamic API base host
  useEffect(() => {
    if (typeof window !== "undefined") {
      const base = apiBase;
      fetch(`${base}/csrf-token`, { credentials: "include" })
        .then((res) => res.json())
        .then((data) => {
          if (data && data.csrf_token) setCsrfToken(data.csrf_token);
        })
        .catch((err) => console.error("Could not fetch CSRF token", err));
      // Restore an existing session after a page refresh
      fetch(`${base}/auth/me`, { credentials: "include" })
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => { if (data && data.email) setUser(data); })
        .catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoginError("");
    setAuthLoading(true);

    try {
      const res = await fetch(`${apiBase}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
        credentials: "include",
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Authentication failed");
      }

      const data = await res.json();
      setUser(data);
    } catch (err) {
      setLoginError(err.message);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await fetch(`${apiBase}/auth/logout`, { method: "POST", credentials: "include" });
    } catch (e) {
      console.error(e);
    }
    setUser(null);
    setResults(null);
    setTraceDetails(null);
    setActiveClaim(null);
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setSearchLoading(true);
    setSearchError("");
    setResults(null);
    setActiveClaim(null);
    setFeedbackSent(false);

    const selectedTypes = Object.keys(studyTypes).filter(k => studyTypes[k]);

    // Helper to perform search request with automatic login fallback
    const executeSearchRequest = async (baseUrl) => {
      let currentCsrf = csrfToken || getCookie("csrf_token");

      // 1. Fetch CSRF token if missing
      if (!currentCsrf) {
        try {
          const csrfRes = await fetch(`${baseUrl}/csrf-token`, { credentials: "include" });
          const csrfData = await csrfRes.json();
          if (csrfData && csrfData.csrf_token) {
            currentCsrf = csrfData.csrf_token;
            setCsrfToken(currentCsrf);
          }
        } catch (e) {
          console.warn("CSRF pre-fetch warning:", e);
        }
      }

      // 2. Execute Search
      let res = await fetch(`${baseUrl}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRF-Token": currentCsrf || ""
        },
        body: JSON.stringify({
          query,
          filters: {
            study_types: selectedTypes,
            year_from: parseInt(yearFrom) || 2000
          }
        }),
        credentials: "include"
      });

      // 3. Expired session: return to the login screen (no silent re-login).
      if (res.status === 401) {
        setUser(null);
        setLoginError("Your session has expired. Please sign in again.");
        throw new Error("Session expired");
      }

      if (!res.ok) {
        let errorMsg = "Search request failed";
        try {
          const errData = await res.json();
          errorMsg = errData.detail || errorMsg;
        } catch (e) {}
        throw new Error(errorMsg);
      }

      return await res.json();
    };

    try {
      let data = null;
      let usedBase = apiBase;
      try {
        // First try primary apiBase
        data = await executeSearchRequest(apiBase);
      } catch (firstErr) {
        if (firstErr.message && firstErr.message.includes("Failed to fetch")) {
          // Fall back to alternative localhost / 127.0.0.1 target
          const altHost = apiBase.includes("127.0.0.1") 
            ? "http://localhost:8000/api/v1" 
            : "http://127.0.0.1:8000/api/v1";
          console.warn(`Primary fetch failed, retrying on fallback API target: ${altHost}`);
          data = await executeSearchRequest(altHost);
          setApiBase(altHost);
          usedBase = altHost;
        } else {
          throw firstErr;
        }
      }

      setResults(data);
      if (data.direct_answer_claims && data.direct_answer_claims.length > 0) {
        setActiveClaim(data.direct_answer_claims[0]);
      }

      if (data.trace_id) {
        fetchTrace(data.trace_id, usedBase);
      }
    } catch (err) {
      if (err.message && err.message.includes("Failed to fetch")) {
        setSearchError("Backend server is starting up or disconnected. Please ensure the backend is running on http://localhost:8000 and click 'Synthesize Evidence' again.");
      } else {
        setSearchError(err.message);
      }
    } finally {
      setSearchLoading(false);
    }
  };

  const apiPost = async (path, body) => {
    let token = csrfToken || getCookie("csrf_token");
    if (!token) {
      const r = await fetch(`${apiBase}/csrf-token`, { credentials: "include" });
      const d = await r.json();
      token = d.csrf_token;
      setCsrfToken(token);
    }
    const res = await fetch(`${apiBase}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": token || "" },
      body: JSON.stringify(body),
      credentials: "include",
    });
    if (res.status === 401) {
      setUser(null);
      setLoginError("Your session has expired. Please sign in again.");
      throw new Error("Session expired");
    }
    if (!res.ok) {
      let msg = `Request failed (${res.status})`;
      try { const e = await res.json(); msg = e.detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    return res.json();
  };

  const fetchTrace = async (traceId, base = apiBase) => {
    setTraceLoading(true);
    try {
      const res = await fetch(`${base}/traces/${traceId}`, { credentials: "include" });
      if (res.ok) {
        const data = await res.json();
        setTraceDetails(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setTraceLoading(false);
    }
  };

  const submitFeedback = (claimText) => {
    // Mimic feedback persistence
    setFeedbackSent(true);
    setFeedbackText("");
    setTimeout(() => {
      setFeedbackClaimId(null);
      setFeedbackSent(false);
    }, 2000);
  };

  // Helper to render source text with highlight coordinates
  const renderHighlightedText = (span) => {
    if (!span || !span.text) return "Select a claim card to view supporting source highlights.";
    
    // We display a snippet containing the context of the sentence
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        <div style={{ fontSize: "0.85rem", color: "var(--accent-cyan)", display: "flex", gap: "1rem" }}>
          <span>PMCID: {span.pmcid}</span>
          <span>Offsets: [{span.section_start_char}, {span.section_end_char}]</span>
        </div>
        <p style={{ fontStyle: "italic", borderLeft: "3px solid var(--accent-cyan)", paddingLeft: "10px", margin: "0.5rem 0" }}>
          &ldquo;{span.text}&rdquo;
        </p>
      </div>
    );
  };

  // Map evidence IDs (E01...) to resolvable citation metadata from the API.
  const sourceById = {};
  ((results && results.evidence_sources) || []).forEach((src) => { sourceById[src.evidence_id] = src; });
  const renderCite = (rawId, key) => {
    const evId = String(rawId).split(".S")[0];
    const src = sourceById[evId];
    return (
      <button
        type="button"
        key={key}
        className={`citation-reference cite-btn ${highlightSource === evId ? "active" : ""}`}
        title={src ? `${src.title} (${src.year || "n.d."}) · ${src.hierarchy_label || src.study_type}` : String(rawId)}
        onClick={(e) => {
          e.stopPropagation();
          setHighlightSource(evId);
          const el = typeof document !== "undefined" ? document.getElementById(`src-${evId}`) : null;
          if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
        }}
      >
        {rawId}
      </button>
    );
  };

  if (!user) {
    return (
      <div className="login-container">
        <form className="login-card glass" onSubmit={handleLogin}>
          <div className="login-title-group">
            <div className="logo-icon" style={{ margin: "0 auto 1rem auto" }}></div>
            <h1 className="login-title glow-text">Indence Login</h1>
            <p className="login-subtitle">Oncology Evidence & Verification Portal</p>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <label className="filter-label">Email Address</label>
            <input 
              type="email" 
              className="input-text" 
              value={email} 
              onChange={(e) => setEmail(e.target.value)} 
              required
            />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <label className="filter-label">Password</label>
            <input 
              type="password" 
              className="input-text" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
              required
            />
          </div>

          {loginError && (
            <div style={{ color: "var(--accent-rose)", fontSize: "0.9rem", textAlign: "center" }}>
              {loginError}
            </div>
          )}

          <button type="submit" className="search-btn" style={{ width: "100%" }} disabled={authLoading}>
            {authLoading ? "Authenticating..." : "Sign In to Portal"}
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="app-logo">
          <div className="logo-icon"></div>
          <div>
            <h1 style={{ fontSize: "1.25rem", fontWeight: "700" }}>Indence</h1>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Clinical Evidence Reasoning Engine</p>
          </div>
        </div>

        <nav className="nav-tabs">
          <button 
            className={`tab-btn ${activeTab === "search" ? "active" : ""}`}
            onClick={() => setActiveTab("search")}
          >
            Clinical Search
          </button>
          <button
            className={`tab-btn ${activeTab === "case" ? "active" : ""}`}
            onClick={() => setActiveTab("case")}
          >
            Patient Case
          </button>
          <button 
            className={`tab-btn ${activeTab === "audit" ? "active" : ""}`}
            onClick={() => setActiveTab("audit")}
          >
            Execution Audit
          </button>
        </nav>

        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <span style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>{user.email}</span>
          <button 
            onClick={handleLogout}
            className="logout-btn"
          >
            Logout
          </button>
        </div>
      </header>

      {/* Case Mode stays mounted so its state survives tab switches */}
      <div className="case-container" style={{ display: activeTab === "case" ? "block" : "none" }}>
        <CaseMode apiPost={apiPost} />
      </div>

      {activeTab === "case" ? null : activeTab === "search" ? (
        <div className="dashboard-grid">
          {/* Filters Panel */}
          <aside className="sidebar-panel glass">
            <h2 style={{ fontSize: "1.1rem", borderBottom: "1px solid var(--border-color)", paddingBottom: "0.5rem" }}>Search Scope</h2>
            
            <div className="filter-group">
              <label className="filter-label">Study Types</label>
              <label className="checkbox-label">
                <input 
                  type="checkbox" 
                  checked={studyTypes.rct}
                  onChange={(e) => setStudyTypes({ ...studyTypes, rct: e.target.checked })}
                />
                Randomized Controlled Trials (RCT)
              </label>
              <label className="checkbox-label">
                <input 
                  type="checkbox" 
                  checked={studyTypes.meta_analysis}
                  onChange={(e) => setStudyTypes({ ...studyTypes, meta_analysis: e.target.checked })}
                />
                Meta-Analyses
              </label>
              <label className="checkbox-label">
                <input 
                  type="checkbox" 
                  checked={studyTypes.systematic_review}
                  onChange={(e) => setStudyTypes({ ...studyTypes, systematic_review: e.target.checked })}
                />
                Systematic Reviews
              </label>
              <label className="checkbox-label">
                <input 
                  type="checkbox" 
                  checked={studyTypes.guideline}
                  onChange={(e) => setStudyTypes({ ...studyTypes, guideline: e.target.checked })}
                />
                Clinical Practice Guidelines
              </label>
              <label className="checkbox-label">
                <input 
                  type="checkbox" 
                  checked={studyTypes.observational}
                  onChange={(e) => setStudyTypes({ ...studyTypes, observational: e.target.checked })}
                />
                Observational Studies
              </label>
            </div>

            <div className="filter-group">
              <label className="filter-label">Publication Year From</label>
              <input 
                type="number" 
                className="input-text" 
                value={yearFrom} 
                onChange={(e) => setYearFrom(e.target.value)}
              />
            </div>
          </aside>

          {/* Main Content Area */}
          <main className="content-main">
            <form onSubmit={handleSearch} className="search-container">
              <div className="search-box">
                <input 
                  type="text" 
                  className="input-text" 
                  style={{ borderRadius: "12px", padding: "0.85rem 1.25rem", fontSize: "1rem" }}
                  value={query} 
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter clinical query e.g. PFS survival statistics for breast cancer patients..."
                />
              </div>
              <button type="submit" className="search-btn" disabled={searchLoading}>
                {searchLoading ? "Analyzing..." : "Synthesize Evidence"}
              </button>
            </form>

            {searchLoading && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem", marginTop: "4rem" }}>
                <div className="loading-spinner"></div>
                <p style={{ color: "var(--text-secondary)" }}>Retrieving medical literature, weighting study types, and verifying clinical parameters...</p>
              </div>
            )}

            {searchError && (
              <div className="glass" style={{ borderColor: "var(--accent-rose)", padding: "1rem", color: "var(--accent-rose)", borderRadius: "12px" }}>
                Error: {searchError}
              </div>
            )}

            {results && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                {/* 1. Clinical Intent Extraction Card */}
                {results.clinical_intent && (
                  <div className="clinical-intent-card glass">
                    <div className="intent-header">
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <span style={{ fontSize: "0.85rem", color: "var(--accent-cyan)", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                          Extracted Clinical Intent
                        </span>
                      </div>
                      <span className={`intent-badge ${results.extraction_confidence || 'high'}`}>
                        {results.extraction_confidence || 'High'} Intent Confidence
                      </span>
                    </div>

                    <div className="intent-grid">
                      {results.clinical_intent.disease && (
                        <div className="intent-item">
                          <span className="intent-label">Target Disease</span>
                          <span className="intent-value">{results.clinical_intent.disease}</span>
                        </div>
                      )}
                      {results.clinical_intent.biomarker_status && (
                        <div className="intent-item">
                          <span className="intent-label">Primary Biomarker</span>
                          <span className="intent-value">{results.clinical_intent.biomarker_status}</span>
                        </div>
                      )}
                      {results.clinical_intent.treatment && results.clinical_intent.treatment.length > 0 && (
                        <div className="intent-item">
                          <span className="intent-label">Queried Treatment</span>
                          <span className="intent-value">
                            {Array.isArray(results.clinical_intent.treatment) 
                              ? results.clinical_intent.treatment.join(", ") 
                              : results.clinical_intent.treatment}
                          </span>
                        </div>
                      )}
                      {results.clinical_intent.line_of_therapy && (
                        <div className="intent-item">
                          <span className="intent-label">Line of Therapy</span>
                          <span className="intent-value">{results.clinical_intent.line_of_therapy}</span>
                        </div>
                      )}
                      {results.clinical_intent.stage && (
                        <div className="intent-item">
                          <span className="intent-label">Disease Stage</span>
                          <span className="intent-value">{results.clinical_intent.stage}</span>
                        </div>
                      )}
                    </div>

                    {results.clinical_intent.rejected_subgroups && results.clinical_intent.rejected_subgroups.length > 0 && (
                      <div style={{ marginTop: "0.85rem", borderTop: "1px solid var(--border-color)", paddingTop: "0.75rem" }}>
                        <span className="intent-label" style={{ display: "block", marginBottom: "0.4rem" }}>
                          Excluded Subgroups (Population Boundary Enforced)
                        </span>
                        <div>
                          {results.clinical_intent.rejected_subgroups.map((sub, sIdx) => (
                            <span key={sIdx} className="rejected-tag">✕ {sub}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* 2. Detected Contradictions Banner — Explainable */}
                {results.contradiction_analysis && results.contradiction_analysis.has_contradiction && (
                  <div className="contradiction-banner glass">
                    <div className="contradiction-header">
                      <span>⚠️ Evidence Contradictions Detected ({results.contradiction_analysis.contradiction_count})</span>
                    </div>
                    <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", marginBottom: "0.5rem" }}>
                      {results.contradiction_analysis.consensus_summary}
                    </p>
                    {results.contradiction_analysis.contradictions && results.contradiction_analysis.contradictions.map((cItem, cIdx) => {
                      const isExpanded = expandedContradictions[cIdx] || false;
                      const interpretationClass = cItem.clinical_interpretation && cItem.clinical_interpretation.includes("manual review")
                        ? "manual-review" : "contextual";
                      return (
                        <div key={cIdx} className="contradiction-card">
                          {/* Collapsed header — always visible */}
                          <div
                            className="contradiction-card-header"
                            onClick={() => setExpandedContradictions(prev => ({ ...prev, [cIdx]: !prev[cIdx] }))}
                          >
                            <div className="contradiction-header-left">
                              <span style={{ fontWeight: "600", fontSize: "0.88rem", color: "var(--text-primary)", flexShrink: 0 }}>{cItem.type}</span>
                              <span className={`contradiction-badge ${(cItem.severity || 'moderate').toLowerCase()}`}>
                                {cItem.severity}
                              </span>
                              {!isExpanded && cItem.clinical_interpretation && (
                                <span className="interpretation-preview">{cItem.clinical_interpretation}</span>
                              )}
                            </div>
                            <span className="contradiction-toggle">{isExpanded ? "▲ Collapse" : "▼ Details"}</span>
                          </div>

                          {/* Expanded body — 5 explainable sections */}
                          {isExpanded && (
                            <div className="contradiction-card-body">
                              {/* Dynamic evidence-specific description */}
                              <div className="contradiction-claim" style={{ borderLeftColor: "rgba(99, 102, 241, 0.4)", background: "rgba(99, 102, 241, 0.05)" }}>
                                {cItem.description}
                              </div>
                              {cItem.action && (
                                <p style={{ fontSize: "0.78rem", color: "var(--accent-cyan)", fontStyle: "italic", margin: 0 }}>
                                  {cItem.action}
                                </p>
                              )}

                              {/* 1. Conflicting Studies */}
                              {cItem.conflicting_evidence_ids && cItem.conflicting_evidence_ids.length > 0 && (
                                <div>
                                  <div style={{ fontSize: "0.7rem", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "0.3rem", letterSpacing: "0.04em" }}>Conflicting Studies</div>
                                  <div className="contradiction-studies">
                                    {cItem.conflicting_evidence_ids.map((study, sIdx) => (
                                      <React.Fragment key={sIdx}>
                                        {sIdx > 0 && <span className="contradiction-vs">vs</span>}
                                        <span className="contradiction-study-pill">
                                          {study.pmcid || "Unknown"}
                                          {study.study_type && <span className="study-type-tag">{study.study_type}</span>}
                                        </span>
                                      </React.Fragment>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* 2. Claim in Conflict */}
                              {cItem.claim_in_conflict && (
                                <div>
                                  <div style={{ fontSize: "0.7rem", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "0.3rem", letterSpacing: "0.04em" }}>Claim in Conflict</div>
                                  <div className="contradiction-claim">{cItem.claim_in_conflict}</div>
                                </div>
                              )}

                              {/* 3. Reason for Difference */}
                              {cItem.reason_for_difference && (
                                <div>
                                  <div style={{ fontSize: "0.7rem", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "0.3rem", letterSpacing: "0.04em" }}>Why They Differ</div>
                                  <div className="contradiction-reason">
                                    <span className="contradiction-reason-icon">🔍</span>
                                    <span>{cItem.reason_for_difference}</span>
                                  </div>
                                </div>
                              )}

                              {/* 4. Evidence Quality */}
                              {cItem.evidence_quality && cItem.evidence_quality.length > 0 && (
                                <div>
                                  <div style={{ fontSize: "0.7rem", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "0.3rem", letterSpacing: "0.04em" }}>Evidence Strength</div>
                                  <div className="contradiction-quality-grid">
                                    {cItem.evidence_quality.map((eq, eqIdx) => (
                                      <div key={eqIdx} className="contradiction-quality-card">
                                        <div className="quality-label">Study {String.fromCharCode(65 + eqIdx)}</div>
                                        <div className="quality-type">{eq.pmcid} — {eq.study_type}</div>
                                        <div className="quality-tier">{eq.strength_tier} Evidence</div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* 5. Clinical Interpretation */}
                              {cItem.clinical_interpretation && (
                                <div>
                                  <div style={{ fontSize: "0.7rem", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "0.3rem", letterSpacing: "0.04em" }}>Clinical Interpretation</div>
                                  <div className={`contradiction-interpretation ${interpretationClass}`}>
                                    {interpretationClass === "manual-review" ? "⚠️ " : "✓ "}
                                    {cItem.clinical_interpretation}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* 2b. Deterministic certainty rating + quantitative evidence map */}
                <ModelTrace trace={results.model_trace} />
                <CertaintyCard certainty={results.certainty} />
                <ClaimVerification claims={results.direct_answer_claims} />
                <EvidenceMap map={results.evidence_map} />

                {/* 3. Main Synthesis & Claims Layout */}
                <div className="claims-source-grid">
                  {/* Claims Panel */}
                  <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                    <h2 style={{ fontSize: "1.25rem", borderBottom: "1px solid var(--border-color)", paddingBottom: "0.5rem" }}>
                      Formulated Clinical Claims
                    </h2>

                    {results.status === "insufficient_evidence" ? (
                      <div className="glass" style={{ padding: "1.5rem", borderLeft: "4px solid var(--accent-rose)" }}>
                        <h3 style={{ color: "var(--accent-rose)", marginBottom: "0.5rem" }}>Abstention Triggered — no unverified answer shown</h3>
                        <p>{results.abstention_reason}</p>
                        {results.llm_error && (
                          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.5rem", fontFamily: "monospace" }}>
                            Provider error: {results.llm_error}
                          </p>
                        )}
                        {results.evidence_sources && results.evidence_sources.length > 0 && (
                          <p style={{ fontSize: "0.9rem", marginTop: "0.75rem" }}>
                            {results.evidence_sources.length} retrieved sources are listed under References for manual review.
                          </p>
                        )}
                      </div>
                    ) : results.synthesis && results.synthesis.recommendation ? (
                      <div className="synthesis-container">
                        {/* Evidence Clusters Summary */}
                        {results.evidence_clusters && results.evidence_clusters.length > 0 && (
                          <div style={{ marginBottom: "1rem" }}>
                            <span className="intent-label" style={{ display: "block", marginBottom: "0.4rem" }}>
                              Structured Trial & Population Clusters ({results.evidence_clusters.length})
                            </span>
                            <div className="clusters-container">
                              {results.evidence_clusters.map((cluster, clIdx) => (
                                <span key={clIdx} className="cluster-pill">
                                  {cluster.population_label || cluster.trial_name || `Cluster #${clIdx + 1}`} ({cluster.candidates_count || 1} studies)
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        <div 
                          className="synth-section recommendation glass glass-interactive"
                          onClick={() => setActiveClaim({ text: results.synthesis.recommendation.text, sentence_ids: results.synthesis.recommendation.sentence_ids, evidence_ids: results.synthesis.recommendation.evidence_ids, aligned_spans: results.synthesis.recommendation.aligned_spans || [] })}
                        >
                          <div className="synth-section-title">Recommendation</div>
                          <p className="synth-text">
                            {results.synthesis.recommendation.text}
                            {results.synthesis.recommendation.sentence_ids && results.synthesis.recommendation.sentence_ids.map((sid, s_idx) => (
                              renderCite(sid, s_idx)
                            ))}
                          </p>
                        </div>

                        {results.synthesis.clinical_reasoning && (
                          <div 
                            className="synth-section glass glass-interactive"
                            style={{ opacity: 0.9 }}
                            onClick={() => setActiveClaim({ text: results.synthesis.clinical_reasoning.text, sentence_ids: results.synthesis.clinical_reasoning.sentence_ids, evidence_ids: results.synthesis.clinical_reasoning.evidence_ids, aligned_spans: results.synthesis.clinical_reasoning.aligned_spans || [] })}
                          >
                            <div className="synth-section-title">Clinical Reasoning</div>
                            <p className="synth-text" style={{ fontSize: "0.95rem" }}>
                              {results.synthesis.clinical_reasoning.text}
                              {results.synthesis.clinical_reasoning.sentence_ids && results.synthesis.clinical_reasoning.sentence_ids.map((sid, s_idx) => (
                                renderCite(sid, s_idx)
                              ))}
                            </p>
                          </div>
                        )}

                        {results.synthesis.supporting_evidence && results.synthesis.supporting_evidence.length > 0 && (
                          <div className="synth-section glass">
                            <div className="synth-section-title">Supporting Evidence</div>
                            {results.synthesis.supporting_evidence.map((ev, idx) => (
                              <div 
                                key={idx} 
                                className="synth-evidence-card glass-interactive"
                                onClick={() => setActiveClaim({ text: ev.text, sentence_ids: ev.sentence_ids, evidence_ids: ev.evidence_ids, aligned_spans: ev.aligned_spans || [] })}
                              >
                                <p className="synth-text" style={{ fontSize: "0.95rem" }}>
                                  {ev.text}
                                  {ev.evidence_ids && ev.evidence_ids.map((eid, e_idx) => (
                                    renderCite(eid, e_idx)
                                  ))}
                                </p>
                              </div>
                            ))}
                          </div>
                        )}

                        {results.synthesis.alternatives && (
                          <div 
                            className="synth-section alternatives glass glass-interactive"
                            onClick={() => setActiveClaim({ text: results.synthesis.alternatives.text, sentence_ids: results.synthesis.alternatives.sentence_ids, evidence_ids: results.synthesis.alternatives.evidence_ids, aligned_spans: results.synthesis.alternatives.aligned_spans || [] })}
                          >
                            <div className="synth-section-title">Alternatives / Considerations</div>
                            <p className="synth-text" style={{ fontSize: "0.95rem" }}>
                              {results.synthesis.alternatives.text}
                              {results.synthesis.alternatives.sentence_ids && results.synthesis.alternatives.sentence_ids.map((sid, s_idx) => (
                                renderCite(sid, s_idx)
                              ))}
                            </p>
                          </div>
                        )}

                        {results.synthesis.bottom_line && (
                          <div 
                            className="synth-section bottom-line glass-interactive"
                            style={{ padding: "1.25rem", borderRadius: "12px", cursor: "pointer" }}
                            onClick={() => setActiveClaim({ text: results.synthesis.bottom_line.text, sentence_ids: results.synthesis.bottom_line.sentence_ids, evidence_ids: results.synthesis.bottom_line.evidence_ids, aligned_spans: results.synthesis.bottom_line.aligned_spans || [] })}
                          >
                            <div className="synth-section-title" style={{ color: "var(--accent-emerald)" }}>Bottom Line</div>
                            <p className="synth-text" style={{ fontWeight: "500" }}>
                              {results.synthesis.bottom_line.text}
                              {(results.synthesis.bottom_line.evidence_ids || []).map((eid, e_idx) => renderCite(eid, `bl-${e_idx}`))}
                            </p>
                          </div>
                        )}
                      </div>
                    ) : (
                      results.direct_answer_claims && results.direct_answer_claims.map((claim, idx) => {
                        const failedVal = !claim.numeric_validation_passed;
                        return (
                          <div 
                            key={idx}
                            className={`claim-card glass glass-interactive ${failedVal ? "failed-validation" : "verified"}`}
                            onMouseEnter={() => setActiveClaim(claim)}
                            onClick={() => setActiveClaim(claim)}
                          >
                            <div className="claim-header">
                              <span style={{ color: "var(--text-muted)", fontSize: "0.85rem", fontWeight: "600" }}>CLAIM #{idx + 1}</span>
                              <span className={`status-badge ${failedVal ? "failed" : "verified"}`}>
                                {failedVal ? "Hallucination Alert" : "Verified Source Grounded"}
                              </span>
                            </div>
                            <p style={{ fontSize: "1rem", lineHeight: "1.5" }}>
                              {claim.text}
                              {claim.sentence_ids && claim.sentence_ids.map((sid, s_idx) => (
                                renderCite(sid, s_idx)
                              ))}
                            </p>
                          </div>
                        );
                      })
                    )}

                    {/* Rejected Citation Safety Audit */}
                    {results.rejected_citations && results.rejected_citations.length > 0 && (
                      <div className="rejected-citations-panel glass">
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                          <span style={{ color: "var(--accent-rose)", fontWeight: "600", fontSize: "0.95rem" }}>
                            🛡️ Population Safety Filter Audit ({results.rejected_citations.length} Citations Excluded)
                          </span>
                        </div>
                        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>
                          The clinical safety validator rejected the following citations due to population or biomarker subgroup mismatches:
                        </p>
                        {results.rejected_citations.map((rejCit, rIdx) => (
                          <div key={rIdx} className="rejected-citation-item">
                            <div style={{ fontWeight: "600", color: "var(--text-primary)" }}>
                              Citation {rejCit.evidence_id} ({rejCit.pmcid})
                            </div>
                            <div style={{ color: "var(--accent-rose)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
                              Reason: {rejCit.rejection_reason}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Grounding Source Coordinates Panel */}
                  <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                    <h2 style={{ fontSize: "1.25rem", borderBottom: "1px solid var(--border-color)", paddingBottom: "0.5rem" }}>
                      Source Verification Spans
                    </h2>
                    <div className="source-highlight-panel glass" style={{ height: "400px", overflowY: "auto" }}>
                      {activeClaim && activeClaim.aligned_spans && activeClaim.aligned_spans.length > 0 ? (
                        activeClaim.aligned_spans.map((span, s_idx) => (
                          <div key={s_idx}>
                            {renderHighlightedText(span)}
                          </div>
                        ))
                      ) : activeClaim && activeClaim.evidence_ids && activeClaim.evidence_ids.length > 0 ? (
                        <div className="glass" style={{ padding: "1.25rem" }}>
                          <p style={{ color: "var(--text-secondary)", marginBottom: "0.75rem", fontSize: "0.95rem" }}>
                            Source spans for this section are available via the following evidence IDs:
                          </p>
                          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                            {activeClaim.evidence_ids.map((id, i) => (
                              renderCite(id, i)
                            ))}
                          </div>
                        </div>
                      ) : (
                        <p style={{ color: "var(--text-muted)", fontSize: "0.95rem" }}>Hover over or select a verified claim card or synthesis section to map and view exact database coordinate highlights.</p>
                      )}
                    </div>

                    {results.evidence_sources && results.evidence_sources.length > 0 && (
                      <div className="references-panel glass">
                        <h3 className="references-title">References ({results.evidence_sources.length} sources given to the generator)</h3>
                        <ol className="references-list">
                          {results.evidence_sources.map((src) => (
                            <li
                              key={src.evidence_id}
                              id={`src-${src.evidence_id}`}
                              className={`reference-item ${highlightSource === src.evidence_id ? "highlighted" : ""}`}
                            >
                              <div className="reference-head">
                                <span className="reference-id">{src.evidence_id}</span>
                                <span className="reference-title">{src.title || src.pmcid}</span>
                              </div>
                              <div className="reference-meta">
                                <span className="ref-badge">{src.hierarchy_label || src.study_type}</span>
                                {src.year && <span>{src.year}</span>}
                                {src.section && <span>§ {src.section}</span>}
                                {src.rerank_score !== null && src.rerank_score !== undefined && (
                                  <span title="Cross-encoder relevance (0–1)">relevance {src.rerank_score.toFixed(2)}</span>
                                )}
                                {src.url && (
                                  <a href={src.url} target="_blank" rel="noopener noreferrer" className="ref-link">
                                    {src.pmcid} ↗
                                  </a>
                                )}
                              </div>
                              <p className="reference-snippet">{src.snippet}</p>
                            </li>
                          ))}
                        </ol>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </main>
        </div>
      ) : (
        /* Audit Trace View */
        <div style={{ padding: "2rem" }}>
          <h2 style={{ fontSize: "1.5rem", marginBottom: "1.5rem", fontFamily: "var(--font-display)" }}>
            RAG Pipeline Execution Audit Logs
          </h2>

          {traceLoading && <p>Loading pipeline execution logs...</p>}

          {traceDetails ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
              <div className="glass" style={{ padding: "1.5rem" }}>
                <h3 style={{ marginBottom: "1rem", color: "var(--accent-cyan)" }}>Pipeline Metadata</h3>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1.5rem" }}>
                  <div>
                    <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>TRACE ID</span>
                    <p style={{ fontSize: "1.1rem", fontFamily: "monospace" }}>{traceDetails.trace_id}</p>
                  </div>
                  <div>
                    <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>QUERY TEXT</span>
                    <p style={{ fontSize: "1.1rem" }}>&ldquo;{traceDetails.query}&rdquo;</p>
                  </div>
                  <div>
                    <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>MODELS (RECORDED FOR THIS RUN)</span>
                    <p style={{ fontSize: "0.95rem", lineHeight: 1.5 }}>
                      Embedding: {traceDetails.model_versions?.embedding || "—"}<br />
                      Reranker: {traceDetails.model_versions?.reranker || "—"}<br />
                      Generator: {traceDetails.model_versions?.llm || traceDetails.generation_model || "—"}<br />
                      NLI verifier: {traceDetails.model_versions?.entailment || "—"}
                    </p>
                  </div>
                </div>
              </div>

              {traceDetails.normalization_context && (
                <div className="glass" style={{ padding: "1.5rem" }}>
                  <h3 style={{ marginBottom: "1rem", color: "var(--accent-cyan)" }}>Query Preprocessing & Medical Expansion</h3>
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    <div>
                      <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>REWRITTEN SEARCH QUERY</span>
                      <p style={{ fontSize: "1rem", fontFamily: "monospace", color: "var(--text-primary)" }}>{traceDetails.normalization_context.rewritten_query || traceDetails.normalization_context.query_text || traceDetails.query}</p>
                    </div>
                  </div>
                </div>
              )}

              {traceDetails.timings && (
                <div className="glass" style={{ padding: "1.5rem" }}>
                  <h3 style={{ marginBottom: "1rem", color: "var(--accent-cyan)" }}>
                    Stage Latency ({(traceDetails.timings.request_wall_s ?? traceDetails.timings.total_e2e_s ?? 0).toFixed(2)} s end-to-end)
                  </h3>
                  {(() => {
                    const stages = Object.entries(traceDetails.timings).filter(([k]) => !["total_e2e_s", "request_wall_s"].includes(k));
                    const maxV = Math.max(0.001, ...stages.map(([, v]) => v));
                    return (
                      <div className="timing-bars">
                        {stages.map(([k, v]) => (
                          <div key={k} className="timing-row">
                            <span className="timing-label">{k.replace(/_s$/, "").replace(/_/g, " ")}</span>
                            <div className="timing-track"><div className="timing-fill" style={{ width: `${(v / maxV) * 100}%` }} /></div>
                            <span className="timing-value">{v < 0.01 ? `${(v * 1000).toFixed(1)} ms` : `${v.toFixed(2)} s`}</span>
                          </div>
                        ))}
                      </div>
                    );
                  })()}
                </div>
              )}

              <div className="glass" style={{ padding: "1.5rem" }}>
                <h3 style={{ marginBottom: "1rem", color: "var(--accent-cyan)" }}>Retrieved Candidates — per-channel ranks and scores</h3>
                <table className="trace-table">
                  <thead>
                    <tr>
                      <th>Fused rank</th>
                      <th>Chunk ID</th>
                      <th>Dense rank</th>
                      <th>Sparse rank</th>
                      <th>RRF score</th>
                      <th>Reranker</th>
                      <th>Population</th>
                      <th>Stage</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(traceDetails.candidates || []).map((c) => {
                      const d = c.score_details || {};
                      return (
                        <tr key={`${c.chunk_id}-${c.rank}`}>
                          <td style={{ fontFamily: "monospace" }}>#{c.rank}</td>
                          <td style={{ fontFamily: "monospace", fontSize: "0.75rem" }}>{String(c.chunk_id).slice(0, 12)}…</td>
                          <td>{d.dense_rank ?? "—"}</td>
                          <td>{d.sparse_rank ?? "—"}</td>
                          <td>{typeof c.score === "number" ? c.score.toFixed(4) : "—"}</td>
                          <td>{typeof d.rerank_score === "number" ? d.rerank_score.toFixed(3) : "—"}</td>
                          <td>{d.population_match_score ?? "filtered"}</td>
                          <td>
                            <span className={`stage-pill ${c.stage === "selected" ? "selected" : ""}`}>
                              {c.stage === "selected" ? "In evidence bundle" : "Candidate"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="glass" style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>
              No recent searches recorded. Execute a clinical search query first to populate trace audit tables.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
