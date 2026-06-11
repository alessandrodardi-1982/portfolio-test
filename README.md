# Portfolio — Operations, Automazione, AI applicata

Portfolio professionale di **Alessandro Dardi** — riposizionamento da Service Manager industriale verso ruoli ibridi Operations + Digital.

🔗 **Sito live:** https://alessandro-dardi.netlify.app

Il principio è semplice: il portfolio non dichiara competenze, le dimostra. Ogni lavoro nasce da un problema operativo reale e arriva a uno strumento funzionante.

---

## I lavori

| # | Strumento | Cosa risolve |
|---|-----------|--------------|
| A1 | **n8n** | Report settimanale automatico: legge i dati, genera una dashboard con Claude API, la invia via email |
| A2 | **Tableau** | Analisi performance fornitori su 7 fornitori e 6 mesi |
| A3 | **Excel avanzato** | Dashboard costi manutenzione con formule avanzate e KPI |
| A4 | **AI applicata** | Chat conversazionale che trasforma una richiesta libera in un ticket strutturato |
| A5 | **Python / Machine Learning** | Manutenzione predittiva: stima la probabilità di guasto e suggerisce il tecnico |

---

## A5 — Manutenzione predittiva (Machine Learning)

Modello che analizza lo storico interventi e stima, per ogni macchinario, la probabilità di guasto imminente. Suggerisce inoltre il tecnico in base alla specializzazione.

**Stack:** Python · pandas · scikit-learn (Random Forest)

**Risultati:**
- 75% di accuratezza sul test set per la previsione di guasto imminente
- Le variabili che pesano di più sono età macchinario (27%) e ore di utilizzo (27%) — coerenti con la logica della manutenzione reale
- Output: parco macchine ordinato per rischio, con motivazione in chiaro e tecnico consigliato, esportabile in CSV

🔗 **Dashboard:** https://alessandro-dardi.netlify.app/dashboard_manutenzione_predittiva.html

### Eseguire lo script

```bash
pip install pandas numpy scikit-learn joblib
python manutenzione_predittiva.py
```

Lo script genera il dataset, addestra il modello, ne stampa la valutazione (accuratezza + feature importance) ed esporta i file. Risultati riproducibili a ogni esecuzione (seed fissi).

**File:**
- `manutenzione_predittiva.py` — script completo e commentato
- `crm_ml_dataset.csv` — dataset di esempio (500 interventi)
- `dashboard_manutenzione_predittiva.html` — visualizzazione dei risultati

---

## Stack del progetto

`HTML/CSS/JS` · `Python` · `pandas` · `scikit-learn` · `n8n` · `Claude API` · `Tableau` · `GitHub Pages` · `Netlify`

---

## Nota sui dati

Tutti i dataset sono fittizi ma realistici, costruiti su casi tipici della manutenzione industriale. I progetti illustrano l'approccio metodologico, non previsioni operative su dati reali.

> L'AI accelera — l'umano decide.
