// src/ImageSearchWidget.tsx
import React, { useState, useEffect, useRef } from "react";
import * as pdfjsLib from "pdfjs-dist";

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

// Usa l'origine dell'asset JS servito dal backend corrente.
// In questo modo il widget chiama sempre lo stesso host da cui e' stato caricato.
const MCP_BASE_URL = (() => {
  try {
    return new URL(import.meta.url).origin;
  } catch {
    return "https://weaviate-openai-app-sdk-v3.onrender.com";
  }
})();

// Imposta a `true` per mostrare uuid, distance raw e bm25_score su ogni card
// e le emoji ✅/❌ basate sui test case in public/test_cases.json.
// Cambia questo valore nel codice, rebuilda e rideploya per attivare/disattivare.
const DEBUG_MODE = false;

type TestCase = {
  input: string;
  expected: string[];
  unwanted: string[];
};

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
  bm25_score?: number;
  dim_similarity?: number;
  combined_score?: number;
};

export const ImageSearchWidget: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);
  const pdfCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [testCases, setTestCases] = useState<TestCase[]>([]);
  const [enlargedImage, setEnlargedImage] = useState<{
    src: string;
    alt: string;
  } | null>(null);

  // Carica i test case dal JSON esterno (solo in debug mode)
  useEffect(() => {
    if (!DEBUG_MODE) return;
    fetch(`${MCP_BASE_URL}/test_cases.json`)
      .then((r) => r.json())
      .then((data: TestCase[]) => setTestCases(data))
      .catch(() => {});
  }, []);

  // Trova il test case corrispondente al file caricato (match parziale sul nome)
  const activeTestCase = debugMode && file
    ? testCases.find((tc) => file.name.includes(tc.input))
    : null;

  // Restituisce l'emoji di validazione per un risultato (null = non classificato)
  const getTestEmoji = (name: string): "✅" | "❌" | null => {
    if (!activeTestCase) return null;
    if (name.includes(activeTestCase.input)) return "✅";
    if (activeTestCase.expected.some((e) => name.includes(e))) return "✅";
    if (activeTestCase.unwanted.some((u) => name.includes(u))) return "❌";
    return null;
  };

  // Restituisce la label testuale per il pannello debug
  const getTestLabel = (name: string): "expected" | "unwanted" | "neutral" | null => {
    if (!activeTestCase) return null;
    if (name.includes(activeTestCase.input)) return "expected";
    if (activeTestCase.expected.some((e) => name.includes(e))) return "expected";
    if (activeTestCase.unwanted.some((u) => name.includes(u))) return "unwanted";
    return "neutral";
  };

  // Crea/revoca l'object URL per l'anteprima del file di input
  useEffect(() => {
    if (!file) {
      setFilePreviewUrl(null);
      return;
    }
    if (file.type === "application/pdf") {
      // Renderizza la prima pagina del PDF su canvas con PDF.js
      const url = URL.createObjectURL(file);
      const render = async () => {
        try {
          const pdf = await pdfjsLib.getDocument({ url }).promise;
          const page = await pdf.getPage(1);
          const viewport = page.getViewport({ scale: 1.5 });
          const canvas = pdfCanvasRef.current;
          if (!canvas) return;
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          await page.render({ canvasContext: ctx, viewport, canvas }).promise;
        } catch {
          // se il rendering fallisce, lascia il canvas vuoto
        } finally {
          URL.revokeObjectURL(url);
        }
      };
      render();
      return;
    }
    const url = URL.createObjectURL(file);
    setFilePreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Chiudi il modal con ESC; scroll to top e blocca body scroll quando aperto
  useEffect(() => {
    if (enlargedImage) {
      window.scrollTo({ top: 0, behavior: "instant" });
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
  }, [enlargedImage]);

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
          collection: "Sinde2",
          image_id: imageId,
          limit: 10,
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
        const name = props.name || props.source_pdf || "(senza nome)";
        const pdf = props.source_pdf || "(sorgente sconosciuta)";
        return `${idx + 1}. ${name} [${pdf}]`;
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

  const getStatusClass = (): string => {
    if (!status) return "status";
    if (status.includes("Errore")) return "status status--error";
    if (status.includes("completata")) return "status status--success";
    return "status status--info";
  };

  const getTestLabelClass = (name: string): string => {
    const label = getTestLabel(name);
    if (label === "expected") return "test-label--expected";
    if (label === "unwanted") return "test-label--unwanted";
    return "test-label--neutral";
  };

  return (
    <div className="widget-root">
      {/* Header */}
      <div className="widget-header">
        <h1 className="widget-title">Ricerca progetti Sinde</h1>
        <p className="widget-subtitle">
          Carica un'immagine o un PDF per trovare progetti simili nella collezione Sinde
        </p>
        {DEBUG_MODE && (
          <button
            onClick={() => setDebugMode((d) => !d)}
            title={debugMode ? "Disattiva modalità debug" : "Attiva modalità debug"}
            className={`debug-btn${debugMode ? " debug-btn--active" : ""}`}
          >
            {debugMode ? "DEBUG ON" : "DEBUG"}
          </button>
        )}
      </div>

      {/* Upload Section */}
      <div className="upload-section">
        <div className="upload-actions">
          <input
            type="file"
            accept=".pdf,application/pdf,image/*"
            onChange={handleFileChange}
            id="file-input"
            className="file-input-hidden"
          />
          <label htmlFor="file-input" className="btn-select-file">
            {file ? "Cambia progetto" : "Seleziona progetto"}
          </label>
        </div>
        {file && (
          <div className="file-selected">
            Progetto selezionato: <strong>{file.name}</strong>
          </div>
        )}
        {file && (
          <div className="input-preview">
            {file.type === "application/pdf" ? (
              <canvas ref={pdfCanvasRef} className="input-preview-img" />
            ) : filePreviewUrl ? (
              <img
                src={filePreviewUrl}
                alt="Anteprima progetto selezionato"
                className="input-preview-img"
              />
            ) : null}
          </div>
        )}
        <button
          onClick={handleUploadAndSearch}
          disabled={!file || isLoading}
          className="btn-search"
        >
          {isLoading ? "Ricerca in corso..." : "Cerca progetti simili"}
        </button>
      </div>

      {/* Status */}
      {status && <div className={getStatusClass()}>{status}</div>}

      {/* Results Grid */}
      {results && results.length > 0 && (
        <div className="results-section">
          <h2 className="results-title">Progetti trovati ({results.length})</h2>
          <div className="results-grid">
            {results.map((r, idx) => (
              <div key={idx} className="result-card">
                {/* Anteprima immagine da image_b64 */}
                {r.properties?.image_b64 && (
                  <div
                    className="result-preview"
                    onClick={() => {
                      if (r.properties?.image_b64) {
                        setEnlargedImage({
                          src: `data:image/png;base64,${r.properties.image_b64}`,
                          alt: r.properties?.name || r.properties?.source_pdf || "Anteprima",
                        });
                      }
                    }}
                  >
                    <img
                      src={`data:image/png;base64,${r.properties.image_b64}`}
                      alt={r.properties?.name || r.properties?.source_pdf || "Anteprima"}
                      onError={(e) => {
                        const parent = e.currentTarget.parentElement;
                        if (parent) {
                          parent.style.display = "none";
                        }
                      }}
                    />
                    <div className="preview-zoom-icon">🔍</div>
                  </div>
                )}

                <div className="result-info">
                <div className="result-index">#{idx + 1}</div>
                {(() => {
                  const displayName = r.properties?.name || r.properties?.source_pdf;
                  return displayName ? (
                    <h3 className="result-name">
                      {debugMode && getTestEmoji(displayName) !== null && (
                        <span style={{ marginRight: "6px" }}>
                          {getTestEmoji(displayName)}
                        </span>
                      )}
                      {displayName}
                    </h3>
                  ) : null;
                })()}
                <div className="result-details">
                  {debugMode ? (
                    <div className="debug-panel">
                      <div className="debug-panel-title">DEBUG</div>
                      <div><strong>uuid:</strong> {r.uuid ?? "—"}</div>
                      {typeof r.distance === "number" && (
                        <>
                          <div><strong>distance:</strong> {r.distance.toFixed(6)}</div>
                          <div><strong>similarity (1−d):</strong> {(1 - r.distance).toFixed(6)}</div>
                        </>
                      )}
                      {typeof r.bm25_score === "number" && (
                        <div><strong>bm25_score:</strong> {r.bm25_score.toFixed(6)}</div>
                      )}
                      {typeof r.dim_similarity === "number" && (
                        <div><strong>dim_similarity:</strong> {r.dim_similarity.toFixed(4)}</div>
                      )}
                      {typeof r.combined_score === "number" && (
                        <div><strong>combined_score:</strong> {r.combined_score.toFixed(6)}</div>
                      )}
                      {(() => {
                        const dn = r.properties?.name || r.properties?.source_pdf;
                        return dn && getTestLabel(dn) !== null ? (
                          <div>
                            <strong>output:</strong>{" "}
                            <span className={getTestLabelClass(dn)}>
                              {getTestLabel(dn)}
                            </span>
                          </div>
                        ) : null;
                      })()}
                    </div>
                  ) : (
                    typeof r.distance === "number" && (
                      <div className="similarity-badge">
                        <strong>Similarità:</strong> {(1 - r.distance).toFixed(3)}
                      </div>
                    )
                  )}
                </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {results && results.length === 0 && (
        <div className="empty-results">Nessun progetto trovato.</div>
      )}

      {/* Modal per immagine ingrandita */}
      {enlargedImage && (
        <div className="modal-overlay" onClick={() => setEnlargedImage(null)}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setEnlargedImage(null);
            }}
            className="modal-close"
            aria-label="Chiudi"
          >
            ×
          </button>
          <img
            src={enlargedImage.src}
            alt={enlargedImage.alt}
            className="modal-image"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
};

export {};
