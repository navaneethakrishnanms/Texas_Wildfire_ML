import React, { useEffect, useRef, useState, useCallback } from "react";
import WildfireMap from "./WildfireMap.jsx";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api/v1";
const HISTORICAL_CUTOVER = "2026-07-31";

const TABS = [
  { key: "live", label: "Live Weather", icon: "☁️", enabled: true },
  { key: "severity", label: "Hazard Severity Outlook", icon: "\u{1F30A}", enabled: false },
  { key: "impact", label: "Impact Forecast", icon: "\u{1F32B}️", enabled: false },
  { key: "infra", label: "Infrastructure Status", icon: "⚡", enabled: false },
];

export default function App() {
  const mapHandle = useRef(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("live");
  const [liveStatus, setLiveStatus] = useState(null);
  const [selectedDate, setSelectedDate] = useState(HISTORICAL_CUTOVER);
  const [autoFollow, setAutoFollow] = useState(true);
  const [showWildfire, setShowWildfire] = useState(true);
  const [cellInfo, setCellInfo] = useState({ count: 0, status: "ok" });

  const pollLiveStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/wildfire/live-status`);
      const data = await res.json();
      setLiveStatus(data);
      if (autoFollow) {
        setSelectedDate(data.published_through || data.historical_through);
      }
    } catch (err) {
      console.error("live-status poll failed", err);
    }
  }, [autoFollow]);

  useEffect(() => {
    pollLiveStatus();
    const id = setInterval(pollLiveStatus, 15000);
    return () => clearInterval(id);
  }, [pollLiveStatus]);

  const isLiveDate = selectedDate > HISTORICAL_CUTOVER;

  return (
    <div className="app-shell">
      <button className="back-btn" onClick={() => setSidebarOpen((s) => !s)}>
        {sidebarOpen ? "←" : "≡"}
      </button>

      <div className={`sidebar ${sidebarOpen ? "" : "hidden"}`}>
        <div className="sidebar-logo">WILDFIRE RISK PORTAL</div>
        <h1>Wildfire Risk for a Safer Texas</h1>
        <p>
          A replica research dashboard built on our own XGBoost ignition-risk
          model (32 features, Test AUROC 0.83), demonstrating what a live
          hazard portal could look like on top of it. This is an independent
          demo project, not an official TDIS product.
        </p>
        <div className="sidebar-nav">
          <a href="#">Model Report</a>
          <a href="#">Feature Importance</a>
          <a href="#">About This Demo</a>
          <a href="#">Data &amp; Methodology</a>
        </div>
        <div className="sidebar-badge">
          Model: model_32feat_tuned.json &middot; Test AUROC 0.8309
        </div>
      </div>

      <div className="top-tabbar">
        {TABS.map((t) => (
          <button
            key={t.key}
            className={`tab-btn ${activeTab === t.key ? "active" : ""} ${
              t.enabled ? "" : "disabled"
            }`}
            onClick={() => t.enabled && setActiveTab(t.key)}
            title={t.enabled ? "" : "No data source connected in this replica"}
          >
            <span>{t.icon}</span> {t.label}
          </button>
        ))}
      </div>

      <div className="status-strip">
        <span className={`live-dot ${isLiveDate ? "" : "waiting"}`} />
        {isLiveDate ? (
          <b>SIMULATED LIVE &mdash; {selectedDate}</b>
        ) : (
          <b>HISTORICAL &mdash; {selectedDate}</b>
        )}
        <span>|</span>
        <label>
          <input
            type="checkbox"
            checked={autoFollow}
            onChange={(e) => setAutoFollow(e.target.checked)}
          />{" "}
          Auto-follow live
        </label>
        <input
          type="date"
          value={selectedDate}
          min="2014-01-01"
          max="2026-08-31"
          disabled={autoFollow}
          onChange={(e) => setSelectedDate(e.target.value)}
        />
        <span>
          Published through: <b>{liveStatus?.published_through || "not started"}</b>
        </span>
        <span>Cells shown: {cellInfo.count.toLocaleString()}</span>
      </div>

      <WildfireMap
        ref={mapHandle}
        selectedDate={selectedDate}
        showWildfire={showWildfire}
        onCellCount={(count, status) => setCellInfo({ count, status })}
      />

      <div className="map-controls">
        <div className="group">
          <button onClick={() => mapHandle.current?.zoomIn()}>+</button>
          <button onClick={() => mapHandle.current?.zoomOut()}>&minus;</button>
          <button onClick={() => mapHandle.current?.resetView()}>&#8635;</button>
        </div>
        <div className="group">
          <button onClick={() => mapHandle.current?.fitScreen()}>&#9974;</button>
        </div>
      </div>

      <div className={`risk-legend ${sidebarOpen ? "" : "no-sidebar"}`}>
        <div style={{ fontWeight: 700, marginBottom: 6 }}>Ignition Risk</div>
        <div className="swatch-row">
          <span className="swatch" style={{ background: "#4ade80" }} /> Low
        </div>
        <div className="swatch-row">
          <span className="swatch" style={{ background: "#facc15" }} /> Moderate
        </div>
        <div className="swatch-row">
          <span className="swatch" style={{ background: "#fb923c" }} /> High
        </div>
        <div className="swatch-row">
          <span className="swatch" style={{ background: "#ef4444" }} /> Extreme
        </div>
      </div>

      <div className="legend">
        <h4>Active Events</h4>
        <div className="legend-row">
          <span className="label">
            <span className="legend-dot wildfire" /> Wildfire
          </span>
          <span
            className="legend-check"
            style={{ cursor: "pointer" }}
            onClick={() => setShowWildfire((v) => !v)}
          >
            {showWildfire ? "✓" : "–"}
          </span>
        </div>
        <div className="legend-row">
          <span className="label">
            <span className="legend-dot rain" /> Rainfall
          </span>
          <span className="legend-dash" title="No data source connected in this replica">
            &ndash;
          </span>
        </div>
        <div className="legend-row">
          <span className="label">
            <span className="legend-dot wind" /> High wind
          </span>
          <span className="legend-dash" title="No data source connected in this replica">
            &ndash;
          </span>
        </div>
        <div className="legend-row">
          <span className="label">
            <span className="legend-dot quake" /> Earthquake
          </span>
          <span className="legend-dash" title="No data source connected in this replica">
            &ndash;
          </span>
        </div>
        <div className="legend-row">
          <span className="label">
            <span className="legend-dot hail" /> Hail
          </span>
          <span className="legend-dash" title="No data source connected in this replica">
            &ndash;
          </span>
        </div>
      </div>

      <div className={`scale-bar ${sidebarOpen ? "with-sidebar" : ""}`}>100 mi</div>

      <div className="disclaimer-bar">
        <span>
          Demo data presented here is model output for research purposes only
          and must not be used for real emergency decisions. See{" "}
          <a href="#">methodology</a>.
        </span>
        <span>Refresh cadence: every 15 seconds (demo) &middot; live drip: 1 day / 2 min</span>
      </div>
    </div>
  );
}
