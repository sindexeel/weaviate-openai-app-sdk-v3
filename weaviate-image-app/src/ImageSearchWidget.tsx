// src/ImageSearchWidget.tsx
import React, { useState, useEffect } from "react";

// Usa l'origine dell'asset JS servito dal backend corrente.
// In questo modo il widget chiama sempre lo stesso host da cui e' stato caricato.
const MCP_BASE_URL = (() => {
  try {
    return new URL(import.meta.url).origin;
  } catch {
    return "https://weaviate-openai-app-sdk-v3.onrender.com";
  }
})();

type SearchResult = {
  uuid?: string;
  properties?: {
    name?: string;
    source_pdf?: string;
    page_index?: number;
    mediaType?: string;
    image_b64?: string;
    [key: string]: any;
  };
  distance?: number;
};

export const ImageSearchWidget: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [enlargedImage, setEnlargedImage] = useState<{
    src: string;
    alt: string;
  } | null>(null);

  // Chiudi il modal con ESC
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && enlargedImage) {
        setEnlargedImage(null);
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [enlargedImage]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResults(null);
    setStatus(null);
  };

  const handleUploadAndSearch = async () => {
    if (!file) {
      setStatus("Seleziona prima un progetto.");
      return;
    }

    try {
      setIsLoading(true);
      setStatus("Caricamento del file in corso...");

      // 1️⃣ Upload immagine/PDF al tuo endpoint /upload-image (HTTP, non MCP tool)
      const form = new FormData();
      form.append("image", file);

      const uploadResp = await fetch(`${MCP_BASE_URL}/upload-image`, {
        method: "POST",
        body: form,
      });

      if (!uploadResp.ok) {
        const text = await uploadResp.text();
        throw new Error(
          `Upload fallito (${uploadResp.status}): ${text || "errore sconosciuto"}`
        );
      }

      const uploadData = await uploadResp.json();
      const imageId = uploadData.image_id as string | undefined;

      if (!imageId) {
        throw new Error("Risposta /upload-image senza image_id");
      }

      setStatus(`Progetto caricato. Avvio la ricerca tra i progetti Sinde...`);

      // 2️⃣ Chiama il backend HTTP /image-search (non più MCP)
      const searchResp = await fetch(`${MCP_BASE_URL}/image-search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection: "Sinde",
          image_id: imageId,
          limit: 20,
        }),
      });

      if (!searchResp.ok) {
        const err = await searchResp.json().catch(() => ({}));
        throw new Error(err.error || "Errore nella ricerca progetti");
      }

      const searchJson = await searchResp.json();
      if (searchJson.error) {
        throw new Error(searchJson.error || "Errore nella ricerca progetti");
      }

      // 3) Mostra i risultati nella UI
      const results = searchJson.results || [];
      setResults(Array.isArray(results) ? results : []);

      // 4) PREPARA il riassunto da mandare al modello
      const summaryParts = results.slice(0, 3).map((r: SearchResult, idx: number) => {
        const props = r.properties || {};
        const name = props.name || "(senza nome)";
        const pdf = props.source_pdf || "(sorgente sconosciuta)";
        const page = props.page_index ?? "?";
        const mediaType = props.mediaType || "";
        return `${idx + 1}. ${name} [${pdf} - pag. ${page}] ${mediaType}`;
      });

      const resultsSummary =
        results.length === 0
          ? "Nessun risultato trovato."
          : `Ho trovato ${results.length} risultati simili. I primi sono:\n` +
            summaryParts.join("\n");

      // 5️⃣ Invia i risultati al backend MCP via HTTP
      try {
        const resp = await fetch(`${MCP_BASE_URL}/widget-push-results`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            results_summary: resultsSummary,
            raw_results: searchJson,
          }),
        });

        if (!resp.ok) {
          const errJson = await resp.json().catch(() => ({}));
          console.error("Errore /widget-push-results:", errJson);
          setStatus(
            `Ricerca completata. ${results.length} progetti trovati (errore salvataggio per ChatGPT)`
          );
        } else {
          console.log("✅ Risultati salvati lato server per ChatGPT");
          setStatus(
            `Ricerca completata. ${results.length} progetti trovati.`
          );
        }
      } catch (err: any) {
        console.error("Errore chiamando /widget-push-results:", err);
        setStatus(
          `Ricerca completata. ${results.length} progetti trovati (errore integrazione: ${
            err?.message || "errore sconosciuto"
          })`
        );
      }
    } catch (err: any) {
      console.error(err);
      setStatus(`Errore: ${err?.message || String(err)}`);
      setResults(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      style={{
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        maxWidth: "900px",
        margin: "0 auto",
        padding: "20px",
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: "24px", textAlign: "center" }}>
        <h1
          style={{
            margin: "0 0 8px 0",
            fontSize: "24px",
            fontWeight: "600",
            color: "#1a1a1a",
          }}
        >
          Ricerca progetti Sinde
        </h1>
        <p
          style={{
            margin: "0",
            fontSize: "14px",
            color: "#666",
          }}
        >
          Carica un'immagine o un PDF per trovare progetti simili nella collezione Sinde
        </p>
      </div>

      {/* Upload Section */}
      <div
        style={{
          marginBottom: "24px",
          padding: "20px",
          border: "2px dashed #ddd",
          borderRadius: "12px",
          backgroundColor: "#fafafa",
          textAlign: "center",
        }}
      >
        <div style={{ marginBottom: "12px" }}>
          <input
            type="file"
            accept=".pdf,application/pdf,image/*"
            onChange={handleFileChange}
            id="file-input"
            style={{ display: "none" }}
          />
          <label
            htmlFor="file-input"
            style={{
              display: "inline-block",
              padding: "12px 24px",
              backgroundColor: "#007bff",
              color: "white",
              borderRadius: "8px",
              cursor: "pointer",
              fontSize: "14px",
              fontWeight: "500",
              transition: "background-color 0.2s",
            }}
            onMouseEnter={(e) => {
              if (!isLoading) e.currentTarget.style.backgroundColor = "#0056b3";
            }}
            onMouseLeave={(e) => {
              if (!isLoading) e.currentTarget.style.backgroundColor = "#007bff";
            }}
          >
            {file ? "Cambia progetto" : "Seleziona progetto"}
          </label>
        </div>
        {file && (
          <div style={{ marginTop: "12px", fontSize: "13px", color: "#666" }}>
            Progetto selezionato: <strong>{file.name}</strong>
          </div>
        )}
        <button
          onClick={handleUploadAndSearch}
          disabled={!file || isLoading}
          style={{
            marginTop: "12px",
            padding: "12px 32px",
            backgroundColor: file && !isLoading ? "#28a745" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: "8px",
            fontSize: "14px",
            fontWeight: "500",
            cursor: file && !isLoading ? "pointer" : "not-allowed",
            transition: "background-color 0.2s",
          }}
          onMouseEnter={(e) => {
            if (file && !isLoading) {
              e.currentTarget.style.backgroundColor = "#218838";
            }
          }}
          onMouseLeave={(e) => {
            if (file && !isLoading) {
              e.currentTarget.style.backgroundColor = "#28a745";
            }
          }}
        >
          {isLoading ? "Ricerca in corso..." : "Cerca progetti simili"}
        </button>
      </div>

      {/* Status */}
      {status && (
        <div
          style={{
            marginBottom: "24px",
            padding: "12px 16px",
            borderRadius: "8px",
            backgroundColor: status.includes("Errore")
              ? "#f8d7da"
              : status.includes("completata")
              ? "#d4edda"
              : "#d1ecf1",
            color: status.includes("Errore")
              ? "#721c24"
              : status.includes("completata")
              ? "#155724"
              : "#0c5460",
            fontSize: "14px",
          }}
        >
          {status}
        </div>
      )}

      {/* Results Grid */}
      {results && results.length > 0 && (
        <div style={{ marginTop: "24px" }}>
          <h2
            style={{
              margin: "0 0 16px 0",
              fontSize: "20px",
              fontWeight: "600",
              color: "#1a1a1a",
            }}
          >
            Progetti trovati ({results.length})
          </h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
              gap: "16px",
            }}
          >
            {results.map((r, idx) => (
              <div
                key={idx}
                style={{
                  border: "1px solid #e0e0e0",
                  borderRadius: "12px",
                  padding: "16px",
                  backgroundColor: "white",
                  boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
                  transition: "transform 0.2s, box-shadow 0.2s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "translateY(-2px)";
                  e.currentTarget.style.boxShadow = "0 4px 8px rgba(0,0,0,0.15)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "translateY(0)";
                  e.currentTarget.style.boxShadow = "0 2px 4px rgba(0,0,0,0.1)";
                }}
              >
                <div
                  style={{
                    fontSize: "12px",
                    color: "#666",
                    marginBottom: "8px",
                    fontFamily: "monospace",
                  }}
                >
                  #{idx + 1}
                </div>
                
                {/* Anteprima immagine da image_b64 */}
                {r.properties?.image_b64 && (
                  <div
                    style={{
                      marginBottom: "12px",
                      borderRadius: "8px",
                      overflow: "hidden",
                      backgroundColor: "#f5f5f5",
                      border: "1px solid #e0e0e0",
                      minHeight: "150px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      position: "relative",
                      cursor: "pointer",
                      transition: "transform 0.2s, box-shadow 0.2s",
                    }}
                    onClick={() => {
                      if (r.properties?.image_b64) {
                        setEnlargedImage({
                          src: `data:image/png;base64,${r.properties.image_b64}`,
                          alt: r.properties?.name || `Anteprima pagina ${r.properties?.page_index || ""}`,
                        });
                      }
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = "scale(1.02)";
                      e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = "scale(1)";
                      e.currentTarget.style.boxShadow = "none";
                    }}
                  >
                    <img
                      src={`data:image/png;base64,${r.properties.image_b64}`}
                      alt={r.properties?.name || `Anteprima pagina ${r.properties?.page_index || ""}`}
                      style={{
                        width: "100%",
                        height: "auto",
                        display: "block",
                        maxHeight: "200px",
                        objectFit: "contain",
                        pointerEvents: "none",
                      }}
                      onError={(e) => {
                        // Se l'immagine fallisce, nascondi il container
                        const parent = e.currentTarget.parentElement;
                        if (parent) {
                          parent.style.display = "none";
                        }
                      }}
                    />
                    {/* Icona zoom sovrapposta */}
                    <div
                      style={{
                        position: "absolute",
                        top: "8px",
                        right: "8px",
                        backgroundColor: "rgba(0, 0, 0, 0.6)",
                        borderRadius: "50%",
                        width: "32px",
                        height: "32px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "white",
                        fontSize: "16px",
                        pointerEvents: "none",
                      }}
                    >
                      🔍
                    </div>
                  </div>
                )}
                
                {r.properties?.name && (
                  <h3
                    style={{
                      margin: "0 0 12px 0",
                      fontSize: "16px",
                      fontWeight: "600",
                      color: "#1a1a1a",
                    }}
                  >
                    {r.properties.name}
                  </h3>
                )}
                <div style={{ fontSize: "13px", color: "#555", lineHeight: "1.6" }}>
                  {r.properties?.source_pdf && (
                    <div style={{ marginBottom: "6px" }}>
                      <strong>PDF:</strong> {r.properties.source_pdf}
                    </div>
                  )}
                  {typeof r.properties?.page_index === "number" && (
                    <div style={{ marginBottom: "6px" }}>
                      <strong>Pagina:</strong> {r.properties.page_index}
                    </div>
                  )}
                  {r.properties?.mediaType && (
                    <div style={{ marginBottom: "6px" }}>
                      <strong>Tipo:</strong> {r.properties.mediaType}
                    </div>
                  )}
                  {typeof r.distance === "number" && (
                    <div
                      style={{
                        marginTop: "12px",
                        padding: "6px 10px",
                        backgroundColor: "#f0f0f0",
                        borderRadius: "6px",
                        fontSize: "12px",
                      }}
                    >
                      <strong>Similarità:</strong> {(1 - r.distance).toFixed(3)}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {results && results.length === 0 && (
        <div
          style={{
            marginTop: "24px",
            padding: "24px",
            textAlign: "center",
            backgroundColor: "#f8f9fa",
            borderRadius: "12px",
            color: "#666",
          }}
        >
          Nessun progetto trovato.
        </div>
      )}

      {/* Modal per immagine ingrandita */}
      {enlargedImage && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.9)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10000,
            padding: "20px",
            cursor: "pointer",
          }}
          onClick={() => setEnlargedImage(null)}
        >
          {/* Pulsante chiudi */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              setEnlargedImage(null);
            }}
            style={{
              position: "absolute",
              top: "20px",
              right: "20px",
              backgroundColor: "rgba(255, 255, 255, 0.2)",
              border: "none",
              borderRadius: "50%",
              width: "40px",
              height: "40px",
              color: "white",
              fontSize: "24px",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "background-color 0.2s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.3)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.2)";
            }}
            aria-label="Chiudi"
          >
            ×
          </button>

          {/* Immagine ingrandita */}
          <img
            src={enlargedImage.src}
            alt={enlargedImage.alt}
            style={{
              maxWidth: "90%",
              maxHeight: "90%",
              objectFit: "contain",
              borderRadius: "8px",
              boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)",
            }}
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
};

export {};
