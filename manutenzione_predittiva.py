"""
================================================================================
 MANUTENZIONE PREDITTIVA — Random Forest su storico interventi CRM
================================================================================
 Autore:  Alessandro Dardi
 Stack:   Python · pandas · scikit-learn
 Scopo:   Stimare la probabilità di guasto imminente per ogni macchinario
          a partire dallo storico interventi, e suggerire il tecnico in base
          alla specializzazione.

 Progetto dimostrativo: il dataset è fittizio ma realistico (parco macchine
 industriale dell'Emilia-Romagna). Il modello illustra l'approccio, non una
 previsione operativa reale.

 Esecuzione:
     pip install pandas numpy scikit-learn joblib
     python manutenzione_predittiva.py

 Output:
     - crm_ml_dataset.csv      (dataset generato)
     - modelli.pkl             (modelli addestrati + encoder)
     - report a console        (accuratezza, feature importance)
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Semi fissi: risultati riproducibili a ogni esecuzione
random.seed(42)
np.random.seed(42)


# ==============================================================================
# 1. GENERAZIONE DATASET
#    Simula uno storico interventi con la logica del mondo reale:
#    macchinari più vecchi e più usati tendono a guastarsi di più.
# ==============================================================================

def genera_dataset(n=500):
    aziende = [
        ("Carpigiani SpA", "Bologna"), ("Fluid System Srl", "Modena"),
        ("Termomeccanica Srl", "Parma"), ("IMA Industries", "Bologna"),
        ("Marchesini Group", "Bologna"), ("Coesia SpA", "Bologna"),
        ("SCM Group", "Rimini"), ("Datalogic SpA", "Bologna"),
        ("Aeroquip Srl", "Ferrara"), ("Marposs SpA", "Bologna"),
        ("Tecnoinox Srl", "Reggio Emilia"), ("Frigomec Srl", "Modena"),
        ("Elcon Srl", "Piacenza"), ("Rollon SpA", "Milano"),
        ("Pneumax Srl", "Bergamo"), ("Bonfiglioli SpA", "Bologna"),
        ("Cefla Srl", "Imola"), ("Samp SpA", "Bologna"),
    ]

    # Ogni tipo di macchinario ha modelli reali associati
    macchinari = {
        "Compressore": ["Compressore Atlas Copco GA55", "Compressore Kaeser SK25",
                        "Compressore Boge C15", "Compressore Ingersoll R55"],
        "Refrigerazione": ["Gruppo Frigo Carrier 30XA", "Chiller Trane CGAM",
                           "Refrigeratore Clivet WSAN", "Gruppo Frigo Daikin EWAD"],
        "Quadro Elettrico": ["Quadro MCC Schneider", "Quadro BT ABB",
                             "Pannello PLC Siemens S7", "Quadro Automazione Allen Bradley"],
        "Pompa": ["Pompa Centrifuga Grundfos NB", "Pompa KSB Etanorm",
                  "Pompa Lowara FC", "Pompa Flygt Serie 3"],
        "Trasmissione": ["Riduttore Bonfiglioli VF", "Variatore SEW Eurodrive",
                         "Cinghia trapezoidale SKF", "Giunto elastico Flender"],
    }

    # Ogni tecnico copre una o più tipologie (specializzazione)
    specializzazione = {
        "Ferretti": ["Compressore"],
        "Bernardini": ["Refrigerazione"],
        "Monti": ["Quadro Elettrico"],
        "Ansaloni": ["Pompa", "Trasmissione"],
    }

    tipi = ["Manutenzione Preventiva", "Guasto Urgente",
            "Manutenzione Correttiva", "Ispezione Periodica"]
    stati = ["Completato", "Completato", "Completato", "Parziale", "Recidiva"]
    prio = ["Alta", "Media", "Bassa"]

    rows, start = [], datetime(2022, 1, 1)

    for i in range(1, n + 1):
        azienda, citta = random.choice(aziende)
        tipo_m = random.choice(list(macchinari.keys()))
        macch = random.choice(macchinari[tipo_m])

        # Nell'80% dei casi il tecnico è coerente con la specializzazione
        adatti = [t for t, s in specializzazione.items() if tipo_m in s]
        tecnico = random.choice(adatti) if (adatti and random.random() < 0.80) \
            else random.choice(list(specializzazione.keys()))

        eta = round(random.uniform(0.5, 12), 1)
        ore = int(eta * random.uniform(1200, 2200))   # ore proporzionali all'età
        mesi_senza = random.randint(0, 18)
        interventi_12m = random.randint(0, 8)

        # CUORE DELLA LOGICA: età + frequenza interventi guidano gravità ed esito
        if eta > 8 and interventi_12m >= 4:
            t = random.choices(tipi, weights=[10, 50, 30, 10])[0]
            p = random.choices(prio, weights=[60, 30, 10])[0]
            s = random.choices(stati, weights=[40, 10, 30, 10, 10])[0]
        elif eta > 5 or interventi_12m >= 2:
            t = random.choices(tipi, weights=[30, 25, 30, 15])[0]
            p = random.choices(prio, weights=[35, 45, 20])[0]
            s = random.choices(stati, weights=[60, 10, 15, 10, 5])[0]
        else:
            t = random.choices(tipi, weights=[50, 10, 20, 20])[0]
            p = random.choices(prio, weights=[15, 40, 45])[0]
            s = random.choices(stati, weights=[75, 5, 10, 7, 3])[0]

        rows.append({
            "ID_Intervento": f"INT-{i:04d}", "Azienda": azienda, "Città": citta,
            "Macchinario": macch, "Tipo_Macchinario": tipo_m,
            "Tipo_Intervento": t, "Priorità": p,
            "Data_Intervento": (start + timedelta(days=random.randint(0, 900))).strftime("%Y-%m-%d"),
            "Tecnico": tecnico, "Durata_Ore": round(random.uniform(1, 12), 1),
            "Stato": s, "Costo_Euro": round(random.uniform(80, 2800), 2),
            "Età_Macchinario_Anni": eta, "Ore_Utilizzo_Totali": ore,
            "Mesi_Senza_Intervento": mesi_senza,
            "Interventi_Ultimi_12_Mesi": interventi_12m, "Esito_Intervento": s,
        })

    return pd.DataFrame(rows)


# ==============================================================================
# 2. PREPARAZIONE
#    Definisce il target da prevedere e codifica le variabili testuali in numeri
#    (i modelli ML lavorano solo con numeri).
# ==============================================================================

def prepara(df):
    # Target: "a rischio guasto" = intervento urgente OPPURE recidiva
    df["Guasto_Imminente"] = (
        (df["Tipo_Intervento"] == "Guasto Urgente") |
        (df["Esito_Intervento"] == "Recidiva")
    ).astype(int)

    # Encoding delle variabili categoriche
    encoders = {}
    for col in ["Tipo_Macchinario", "Priorità"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    features = ["Età_Macchinario_Anni", "Ore_Utilizzo_Totali",
                "Mesi_Senza_Intervento", "Interventi_Ultimi_12_Mesi",
                "Tipo_Macchinario_enc", "Priorità_enc"]
    return df, encoders, features


# ==============================================================================
# 3. ADDESTRAMENTO E VALUTAZIONE
#    Random Forest: robusto, interpretabile, ideale per dati tabellari.
#    class_weight="balanced" compensa il fatto che i guasti sono rari.
# ==============================================================================

def addestra(df, features):
    X = df[features]
    y = df["Guasto_Imminente"]

    # 75% training, 25% test — stratify mantiene le proporzioni del target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42)
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)

    print(f"\n{'='*60}")
    print(f" MODELLO — Guasto Imminente")
    print(f"{'='*60}")
    print(f" Accuratezza sul test set: {acc*100:.1f}%\n")
    print(classification_report(yte, pred,
          target_names=["No rischio", "A rischio"], digits=2))

    print(" Feature importance (cosa pesa di più):")
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for f, v in imp.items():
        print(f"   {f:30s} {v*100:5.1f}%")

    return model


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("1. Genero il dataset...")
    df = genera_dataset(500)
    df.to_csv("crm_ml_dataset.csv", index=False)
    print(f"   {len(df)} interventi salvati in crm_ml_dataset.csv")

    print("\n2. Preparo i dati...")
    df, encoders, features = prepara(df)
    print(f"   Target: {df['Guasto_Imminente'].mean()*100:.1f}% a rischio")

    print("\n3. Addestro il modello...")
    model = addestra(df, features)

    joblib.dump({"model": model, "encoders": encoders, "features": features},
                "modelli.pkl")
    print(f"\n   Modello salvato in modelli.pkl")
    print("\nFatto. Per applicarlo a nuovi dati: joblib.load('modelli.pkl')")
