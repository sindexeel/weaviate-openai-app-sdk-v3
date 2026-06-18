# Resoconto tecnico — Progetto Sinde
**Data:** 16 giugno 2025  
**Preparato da:** Giada Del Testa  
---

## Obiettivo

Sviluppo di un sistema di ricerca per similarità su disegni tecnici meccanici (PDF).  
L'utente carica un disegno e il sistema restituisce i disegni più simili presenti nel database.

---

## Architettura attuale (Sinde4)

Il sistema utilizza:

- **Database vettoriale:** Weaviate Cloud (Europa — `europe-west3`)
- **Modello di embedding:** OpenAI `text-embedding-3-large`
- **Descrizione immagini:** GPT-4.1-mini (genera una caption testuale del disegno)
- **Ricerca:** Hybrid search — combinazione di ricerca vettoriale (semantica) e BM25 (lessicale)
- **Dataset indicizzato:** 200 disegni tecnici in formato PDF

**Flusso di funzionamento:**

1. L'utente carica un disegno (PDF o PNG)
2. GPT-4.1-mini descrive la geometria del pezzo in testo strutturato
3. Il testo viene convertito in vettore numerico (embedding)
4. Weaviate cerca i vettori più vicini nel database
5. I risultati vengono restituiti ordinati per similarità

---

## Risultati della valutazione (Recall@10)

La valutazione è stata condotta sui **6 casi di test forniti dal cliente**, con la seguente metrica:

> **Score = Recall@10 − penalità per risultati indesiderati**

| Modalità | Score medio | Recall@10 | MRR |
|---|---|---|---|
| Oracle (upper bound teorico) | 36% | — | 0.144 |
| Hybrid α=0.7 (produzione attuale) | **47%** | 56% | 0.550 |
| Near-text puro (solo vettoriale) | 33% | 42% | 0.521 |

> **α=0.7** significa: 70% peso sulla ricerca vettoriale (semantica), 30% sul BM25 (parole chiave).  
> Il valore ottimale è stato determinato sperimentalmente su un range da 0.3 a 1.0.

**Risultati per caso di test:**

| Query | Score oracle | Score produzione α=0.7 |
|---|---|---|
| 702.0019.00.0 | 0% | 100% |
| PF00009371 | 67% | 33% |
| PF00018620 | 100% | 0% |
| 701.2425.00.0 | 0% | 0% |
| PF00016814 | 0% | 50% |
| 701.1828.00.0 | 50% | 100% |
| **MEDIA** | **36%** | **47%** |

---

## Problemi identificati e cause

### 1. PF00009371 — regressione tra oracle e produzione

**Sintomo:** L'oracle trova tutti e 3 i pezzi attesi ai rank 2, 3, 4. La produzione ne trova solo 1.

**Causa tecnica:** GPT descrive lo stesso pezzo con categorie diverse a seconda dell'immagine di input.

- Caption indicizzata (da PDF completo): *"ingranaggio cilindrico scanalato con dentatura esterna diritta"*
- Caption di query (da PNG di test): *"albero scanalato con profilo esterno a denti scanalati"*

Il componente BM25 della ricerca ibrida penalizza il mismatch lessicale: `ingranaggio` ≠ `albero`. Il modello AI interpreta correttamente la geometria ma usa vocabolari diversi in contesti diversi.

**Domanda al cliente:** Il pezzo `PF00009371` è classificato internamente come **ingranaggio** o come **albero scanalato**? Avere la denominazione ufficiale del cliente aiuterebbe a uniformare le descrizioni.

---

### 2. PF00018620 — pezzo atteso appena fuori dalla top-10

**Sintomo:** Il pezzo atteso (`704.0256.00.0`) è al **rank 8** nello spazio vettoriale (distanza coseno 0.109), ma viene scalzato fuori dalla top-10 da 6 documenti con distanze molto simili (0.090–0.111).

**Causa tecnica:** Il dataset contiene molti pezzi geometricamente simili (blocchi cubici con fori) e il margine tra rank 7 e rank 8 è meno di 0.002 di distanza coseno — di fatto rumore.

**Impatto:** Aumentando il risultati restituiti da 10 a 15, il pezzo atteso rientrerebbe sempre nella risposta.

**Domanda al cliente:** Qual è il numero massimo accettabile di risultati visualizzati nel widget? Restituire 15 invece di 10 risultati risolverebbe questo caso specifico senza modifiche al modello.

---

### 3. 701.2425.00.0 — due pezzi attesi con distanza vettoriale diversa

**Sintomo:** Dei due pezzi attesi, `701.2332` è al rank 10 (appena dentro), mentre `701.2320` è al **rank 28** — genuinamente lontano nello spazio vettoriale.

**Causa tecnica:** I due pezzi attesi (`701.2332` e `701.2320`) sembrano descrivere geometrie abbastanza diverse dal punto di vista del modello AI. Il sistema li considera non simili tra loro in misura sufficiente.

**Domanda al cliente:** I pezzi `701.2425`, `701.2332` e `701.2320` appartengono alla stessa famiglia per quale criterio? (stesso stampo, stessa funzione, stessa commessa?) Capire il criterio di similarità utilizzato dal cliente aiuterebbe a riorientare il modello verso le caratteristiche rilevanti.

---

### 4. Nota generale — qualità delle immagini di test

Le immagini di test fornite dal cliente (PNG in `/test_cases/`) sono ritagli del disegno senza il cartiglio (riquadro con codice, revisione, denominazione). I PDF indicizzati includono invece il cartiglio completo.

Questa differenza influenza le descrizioni generate da GPT e può causare discrepanze nelle caption, impattando sia la ricerca vettoriale che quella BM25.

**Domanda al cliente:** In produzione, gli utenti caricheranno il PDF completo o solo la parte grafica del disegno? La risposta determina come ottimizzare il sistema.

---

## Confronto con approccio alternativo (Sinde3)

È stato testato un approccio alternativo basato su embedding multimodale (immagine + testo simultaneamente, modello Google `multimodalembedding@001`). I risultati sono stati inferiori:

| | Oracle | Produzione |
|---|---|---|
| Sinde3 (multimodale) | 49% | 24% |
| Sinde4 (testo via GPT) | 36% | **47%** |

Il modello multimodale di Google risulta meno efficace di GPT+OpenAI per disegni tecnici 2D in bianco e nero. L'approccio Sinde4 rimane quello da sviluppare.

---

## Prossimi passi proposti

| Priorità | Azione | Impatto atteso |
|---|---|---|
| Alta | Uniformare il vocabolario del prompt GPT (categoria fissa: *ingranaggio, albero, flangia, ...*) tra indexing e query | +10–15% score su PF00009371 e casi simili |
| Media | Aumentare K da 10 a 15 risultati | Recupera PF00018620 senza modifiche al modello |
| Media | Chiarire con il cliente il criterio di similarità per la famiglia 701.24xx | Permette di riorientare il modello |
| Bassa | Re-indicizzare con il prompt allineato dopo chiarimenti cliente | Baseline più alta per tutti i casi |

---

## Domande aperte per il cliente

1. I pezzi `701.2425`, `701.2332` e `701.2320` sono simili per quale criterio? (funzione, famiglia, processo produttivo?)
2. Il pezzo `PF00009371` è classificato come *ingranaggio* o *albero scanalato* nella vostra nomenclatura interna?
3. In produzione gli utenti caricheranno **PDF completi** o **immagini ritagliate** del disegno?
4. È accettabile visualizzare **15 risultati** invece di 10 nel widget di ricerca?
5. I casi di test forniti (6 query) sono rappresentativi della varietà tipica delle ricerche, o ci sono famiglie di pezzi non ancora coperte?

---

*Documento generato internamente — non distribuire al cliente nella forma attuale.*
