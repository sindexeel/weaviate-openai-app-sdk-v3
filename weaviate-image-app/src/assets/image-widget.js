// image-widget.js

// Usa la stessa origin dello script corrente (fallback all'URL storico).
const MCP_BASE_URL = (() => {
  try {
    if (document.currentScript && document.currentScript.src) {
      return new URL(document.currentScript.src).origin;
    }
    return window.location.origin;
  } catch (_err) {
    return "https://weaviate-openai-app-sdk-v3.onrender.com";
  }
})();

function createEl(tag, attrs = {}, children = []) {
  const el = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === "style" && typeof v === "object") {
      Object.assign(el.style, v);
    } else if (k === "className") {
      el.className = v;
    } else {
      el.setAttribute(k, v);
    }
  });
  children.forEach((c) => {
    if (typeof c === "string") el.appendChild(document.createTextNode(c));
    else if (c) el.appendChild(c);
  });
  return el;
}

async function uploadImage(file) {
  const form = new FormData();
  form.append("image", file); // deve chiamarsi "image" per /upload-image

  const resp = await fetch(`${MCP_BASE_URL}/upload-image`, {
    method: "POST",
    body: form,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(
      `Upload fallito (${resp.status}): ${text || "errore sconosciuto"}`
    );
  }

  const data = await resp.json();
  if (!data.image_id) {
    throw new Error("Risposta di /upload-image senza image_id");
  }
  return data.image_id;
}

async function callImageSearchVertex(imageId, limit = 5) {
  if (!window.openai || typeof window.openai.callTool !== "function") {
    throw new Error(
      "window.openai.callTool non è disponibile. Questo funziona solo dentro ChatGPT come widget."
    );
  }

  const resp = await window.openai.callTool("image_search_vertex", {
    collection: "Sinde3",
    image_id: imageId,
    limit,
  });

  let payload = resp;
  if (payload && payload.structuredContent) {
    payload = payload.structuredContent;
  } else if (payload && payload.content) {
    payload = payload.content;
  }

  const results =
    (payload && payload.results) ||
    (Array.isArray(payload) ? payload : payload?.data?.results) ||
    [];

  return Array.isArray(results) ? results : [];
}

function renderWidget(root) {
  root.innerHTML = "";

  const container = createEl("div", {
    style: {
      fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
      fontSize: "14px",
      border: "1px solid #ddd",
      borderRadius: "8px",
      padding: "12px",
      maxWidth: "650px",
      margin: "0 auto",
    },
  });

  const title = createEl("h3", { style: { marginTop: "0", marginBottom: "4px" } }, [
    "Weaviate Image Search",
  ]);

  const subtitle = createEl(
    "p",
    { style: { marginTop: "0", marginBottom: "12px" } },
    [
      "Carica un'immagine o un PDF, lo inviamo al tuo server MCP e usiamo ",
      createEl("code", {}, ["image_search_vertex"]),
      " sulla collection ",
      createEl("strong", {}, ["Sinde3"]),
      ".",
    ]
  );

  const fileInput = createEl("input", {
    type: "file",
    accept: ".pdf,application/pdf,image/*",
  });
  const button = createEl(
    "button",
    {
      type: "button",
      style: {
        marginLeft: "8px",
        padding: "6px 12px",
        borderRadius: "6px",
        border: "1px solid #ccc",
        cursor: "pointer",
      },
    },
    ["Carica e cerca"]
  );
  const status = createEl("p", { style: { marginTop: "8px", minHeight: "18px" } });
  const resultsBox = createEl("div", { style: { marginTop: "12px" } });

  let currentFile = null;
  let loading = false;

  fileInput.addEventListener("change", (e) => {
    const target = e.target;
    currentFile = target.files && target.files[0] ? target.files[0] : null;
    status.textContent = "";
    resultsBox.innerHTML = "";
  });

  async function handleClick() {
    if (loading) return;
    if (!currentFile) {
      status.textContent = "Seleziona prima un'immagine o un PDF.";
      return;
    }

    try {
      loading = true;
      button.textContent = "Attendere...";
      button.style.cursor = "default";
      button.disabled = true;
      status.textContent = "Caricamento file...";

      const imageId = await uploadImage(currentFile);
      status.textContent = `File caricato (image_id = ${imageId}). Ricerca in corso...`;

      const results = await callImageSearchVertex(imageId, 5);

      status.textContent = "Ricerca completata.";
      resultsBox.innerHTML = "";

      const heading = createEl("h4", { style: { marginTop: "0" } }, ["Risultati"]);
      resultsBox.appendChild(heading);

      if (!results.length) {
        resultsBox.appendChild(
          createEl("p", {}, ["Nessun risultato trovato."])
        );
        return;
      }

      const list = createEl("ul", { style: { paddingLeft: "18px" } });
      results.forEach((r) => {
        const li = createEl("li", { style: { marginBottom: "6px" } });
        const uuid = r.uuid || "";

        li.appendChild(
          createEl("div", {}, [createEl("code", {}, [uuid])])
        );

        if (r.properties && r.properties.name) {
          li.appendChild(
            createEl("div", {}, ["Nome: ", r.properties.name])
          );
        }
        if (typeof r.distance === "number") {
          li.appendChild(
            createEl("div", {}, [
              "Distanza: ",
              r.distance.toFixed(3),
            ])
          );
        }
        if (r.properties && r.properties.source_pdf) {
          li.appendChild(
            createEl("div", {}, [
              "PDF: ",
              r.properties.source_pdf,
            ])
          );
        }
        if (
          r.properties &&
          typeof r.properties.page_index === "number"
        ) {
          li.appendChild(
            createEl("div", {}, [
              "Pagina: ",
              String(r.properties.page_index),
            ])
          );
        }

        list.appendChild(li);
      });

      resultsBox.appendChild(list);
    } catch (err) {
      console.error(err);
      resultsBox.innerHTML = "";
      status.textContent =
        "Errore: " + (err && err.message ? err.message : String(err));
    } finally {
      loading = false;
      button.textContent = "Carica e cerca";
      button.style.cursor = "pointer";
      button.disabled = false;
    }
  }

  button.addEventListener("click", handleClick);

  container.appendChild(title);
  container.appendChild(subtitle);
  container.appendChild(
    createEl("div", {}, [fileInput, button])
  );
  container.appendChild(status);
  container.appendChild(resultsBox);

  root.appendChild(container);
}

document.addEventListener("DOMContentLoaded", () => {
  // ChatGPT metterà il tuo HTML dentro un iframe;
  // noi montiamo il widget in un div con id "root".
  const root = document.getElementById("root") || document.body;
  renderWidget(root);
});
