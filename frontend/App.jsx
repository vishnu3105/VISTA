import { useState, useEffect, useRef } from "react";

const API = "http://localhost:8000";
const WS  = "ws://localhost:8000/ws/alerts";

const ALERT_COLORS = {
  ticket_abuse: { bg: "#FEF2F2", border: "#FCA5A5", text: "#991B1B", label: "Ticket Abuse" },
  stampede:     { bg: "#FFF7ED", border: "#FDBA74", text: "#9A3412", label: "Stampede Risk" },
  fall:         { bg: "#FFFBEB", border: "#FCD34D", text: "#92400E", label: "Fall Detected" },
  push:         { bg: "#F0F9FF", border: "#7DD3FC", text: "#0C4A6E", label: "Push Detected" },
};

function AlertBadge({ kind }) {
  const c = ALERT_COLORS[kind] || { bg: "#F9FAFB", border: "#D1D5DB", text: "#374151", label: kind };
  return (
    <span style={{
      background: c.bg, border: `1px solid ${c.border}`, color: c.text,
      padding: "2px 8px", borderRadius: 6, fontSize: 12, fontWeight: 500,
    }}>
      {c.label}
    </span>
  );
}

function AlertRow({ alert }) {
  const time = new Date(alert.timestamp * 1000).toLocaleTimeString();
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "8px 0", borderBottom: "1px solid #F3F4F6",
    }}>
      <span style={{ fontSize: 11, color: "#9CA3AF", minWidth: 56 }}>{time}</span>
      <AlertBadge kind={alert.kind} />
      <span style={{ fontSize: 12, color: "#374151" }}>
        Cam {alert.camera_id}
        {alert.track_ids?.length > 0 && ` · IDs: ${alert.track_ids.join(", ")}`}
      </span>
      <span style={{
        marginLeft: "auto", fontSize: 11, color: "#6B7280",
        background: "#F9FAFB", padding: "1px 6px", borderRadius: 4,
      }}>
        {Math.round(alert.confidence * 100)}%
      </span>
    </div>
  );
}

function StatCard({ label, value, color = "#111827" }) {
  return (
    <div style={{
      background: "#fff", border: "1px solid #E5E7EB", borderRadius: 10,
      padding: "14px 18px", flex: 1,
    }}>
      <p style={{ fontSize: 12, color: "#6B7280", margin: "0 0 4px" }}>{label}</p>
      <p style={{ fontSize: 24, fontWeight: 600, color, margin: 0 }}>{value}</p>
    </div>
  );
}

export default function App() {
  const [alerts, setAlerts]   = useState([]);
  const [live, setLive]       = useState(false);
  const wsRef                 = useRef(null);

  // Fetch initial alerts
  useEffect(() => {
    fetch(`${API}/alerts?limit=50`)
      .then(r => r.json())
      .then(setAlerts)
      .catch(() => {});
  }, []);

  // WebSocket for live alerts
  useEffect(() => {
    const ws = new WebSocket(WS);
    wsRef.current = ws;

    ws.onopen  = () => setLive(true);
    ws.onclose = () => setLive(false);
    ws.onmessage = (e) => {
      const alert = JSON.parse(e.data);
      if (alert.kind === "ping") return;
      setAlerts(prev => [alert, ...prev].slice(0, 100));
    };

    return () => ws.close();
  }, []);

  const counts = alerts.reduce((acc, a) => {
    acc[a.kind] = (acc[a.kind] || 0) + 1;
    return acc;
  }, {});

  return (
    <div style={{ fontFamily: "Inter, sans-serif", background: "#F9FAFB", minHeight: "100vh", padding: 24 }}>
      
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
        <div style={{
          width: 36, height: 36, background: "#1D4ED8", borderRadius: 8,
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "#fff", fontWeight: 700, fontSize: 14,
        }}>V</div>
        <div>
          <h1 style={{ margin: 0, fontSize: 18, fontWeight: 600 }}>VISTA Dashboard</h1>
          <p style={{ margin: 0, fontSize: 12, color: "#6B7280" }}>
            Visual Intelligence for Surveillance and Threat Analysis
          </p>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: live ? "#22C55E" : "#EF4444",
          }}/>
          <span style={{ fontSize: 12, color: "#6B7280" }}>{live ? "Live" : "Disconnected"}</span>
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
        <StatCard label="Total Alerts" value={alerts.length} />
        <StatCard label="Ticket Abuse" value={counts.ticket_abuse || 0} color="#DC2626" />
        <StatCard label="Stampede Risk" value={counts.stampede || 0} color="#EA580C" />
        <StatCard label="Falls" value={counts.fall || 0} color="#D97706" />
      </div>

      {/* Main grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 16 }}>

        {/* Camera feeds */}
        <div>
          <h2 style={{ fontSize: 14, fontWeight: 500, margin: "0 0 12px", color: "#374151" }}>
            Camera Feeds
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {[0, 1].map(camId => (
              <div key={camId} style={{
                background: "#111827", borderRadius: 10, overflow: "hidden",
                aspectRatio: "16/9", position: "relative",
              }}>
                <img
                  src={`${API}/stream/${camId}`}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  onError={e => { e.target.style.display = "none"; }}
                  alt={`Camera ${camId}`}
                />
                <div style={{
                  position: "absolute", top: 8, left: 8,
                  background: "rgba(0,0,0,0.6)", color: "#fff",
                  fontSize: 11, padding: "2px 8px", borderRadius: 4,
                }}>
                  CAM {camId}
                </div>
              </div>
            ))}
          </div>

          {/* Gate ROI hint */}
          <div style={{
            marginTop: 12, background: "#EFF6FF", border: "1px solid #BFDBFE",
            borderRadius: 8, padding: "10px 14px", fontSize: 12, color: "#1E40AF",
          }}>
            Yellow box on feed = Gate ROI. Adjust <code>--gate-roi X1 Y1 X2 Y2</code> when starting detector.
          </div>
        </div>

        {/* Alert feed */}
        <div style={{
          background: "#fff", border: "1px solid #E5E7EB",
          borderRadius: 10, padding: 16, maxHeight: 520, overflowY: "auto",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
            <h2 style={{ fontSize: 14, fontWeight: 500, margin: 0, color: "#374151" }}>Alert Feed</h2>
            <button
              onClick={() => setAlerts([])}
              style={{
                fontSize: 11, color: "#6B7280", background: "none",
                border: "1px solid #E5E7EB", borderRadius: 4, padding: "2px 8px", cursor: "pointer",
              }}
            >
              Clear
            </button>
          </div>

          {alerts.length === 0 ? (
            <p style={{ fontSize: 13, color: "#9CA3AF", textAlign: "center", marginTop: 40 }}>
              No alerts yet. Start the detector to see live events.
            </p>
          ) : (
            alerts.map((a, i) => <AlertRow key={i} alert={a} />)
          )}
        </div>
      </div>
    </div>
  );
}
