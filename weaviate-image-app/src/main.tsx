// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./widget.css";

function Boot() {
  React.useEffect(() => {
    console.info("[sinde-widget] React montato");
  }, []);
  return <App />;
}

function blockedOrigin(uri: string): string {
  if (!uri) return "";
  try {
    if (uri.includes("://")) return new URL(uri).origin;
  } catch {
    // URI non assoluto: niente query string.
  }
  return uri.split("?")[0].slice(0, 160);
}

const rootEl = document.getElementById("root");
if (!rootEl) {
  console.error("[sinde-widget] elemento #root assente");
} else {
  window.addEventListener("securitypolicyviolation", (event) => {
    console.error("[sinde-widget] violazione CSP", {
      directive: event.effectiveDirective,
      blocked: blockedOrigin(event.blockedURI),
    });
  });
  try {
    ReactDOM.createRoot(rootEl).render(
      <React.StrictMode>
        <Boot />
      </React.StrictMode>
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : "errore sconosciuto";
    console.error("[sinde-widget] montaggio React fallito", message);
  }
}
