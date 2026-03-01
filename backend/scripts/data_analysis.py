"""
Análise exploratória do novo dataset PEDE 2022-2024.
Usado para embasar decisões de redesign do pipeline ML.
"""

import pandas as pd

PATH = "/app/data/raw/BASE DE DADOS PEDE 2022-2024 - DATATHON.xlsx"


def load_year(sheet: str, year: int) -> pd.DataFrame:
    df = pd.ExcelFile(PATH).parse(sheet)
    # Expected INDE column for this year (format varies: "INDE 22" vs "INDE 2023")
    inde_candidates = [f"INDE {year}", f"INDE {str(year)[-2:]}"]
    pedra_candidates = [f"Pedra {year}", f"Pedra {str(year)[-2:]}"]
    renames = {}
    inde_renamed = False
    pedra_renamed = False
    for c in df.columns:
        cl = c.strip()
        if cl in ("Defas", "Defasagem"):
            renames[c] = "defasagem"
        if cl in ("Nome", "Nome Anonimizado"):
            renames[c] = "nome"
        if cl in ("Fase Ideal", "Fase ideal"):
            renames[c] = "fase_ideal"
        if not inde_renamed and cl in inde_candidates:
            renames[c] = "INDE"
            inde_renamed = True
        if not pedra_renamed and cl in pedra_candidates:
            renames[c] = "Pedra"
            pedra_renamed = True
    df = df.rename(columns=renames)
    df["year"] = year
    return df


d22 = load_year("PEDE2022", 2022)
d23 = load_year("PEDE2023", 2023)
d24 = load_year("PEDE2024", 2024)

years = [(2022, d22), (2023, d23), (2024, d24)]

ra22, ra23, ra24 = set(d22["RA"]), set(d23["RA"]), set(d24["RA"])

print("=== OVERLAP DE ALUNOS ===")
print(f"2022={len(ra22)}  2023={len(ra23)}  2024={len(ra24)}")
print(
    f"22∩23={len(ra22 & ra23)}  22∩24={len(ra22 & ra24)}  23∩24={len(ra23 & ra24)}  22∩23∩24={len(ra22 & ra23 & ra24)}"
)
print(f"Novo em 23 (não estava em 22): {len(ra23 - ra22)}")
print(f"Novo em 24 (não estava em 22 nem 23): {len(ra24 - ra22 - ra23)}")
print(f"Saiu em 23 (estava em 22, sumiu): {len(ra22 - ra23)}")

print()
print("=== DEFASAGEM DISTRIBUIÇÃO POR ANO ===")
for yr, df in years:
    dist = dict(sorted(df["defasagem"].value_counts().items()))
    pos_rate = (df["defasagem"] < 0).mean()
    print(f"  {yr}: {dist}  |  defasado(raw<0)={pos_rate:.1%}")

print()
print("=== IAN × DEFASAGEM (leakage check) ===")
for yr, df in years:
    corr = df["IAN"].corr(df["defasagem"])
    uniq = sorted(df["IAN"].dropna().unique())
    print(f"  {yr}: corr={corr:.4f}  IAN únicos={uniq}")

print()
print("=== IPP NULOS POR ANO ===")
for yr, df in years:
    if "IPP" in df.columns:
        n = df["IPP"].isna().sum()
        pct = df["IPP"].isna().mean()
        print(f"  {yr}: {n} nulos ({pct:.1%})  mean={df['IPP'].mean():.2f}")
    else:
        print(f"  {yr}: coluna ausente")

print()
print("=== INDE (índice composto) ===")
for yr, df in years:
    if "INDE" in df.columns:
        inde = pd.to_numeric(df["INDE"], errors="coerce")
        corr = inde.corr(df["defasagem"])
        n_null = inde.isna().sum()
        print(f"  {yr}: mean={inde.mean():.3f}  nulls={n_null}  corr_defasagem={corr:.4f}")

print()
print("=== PEDRA (classificação de performance) ===")
for yr, df in years:
    if "Pedra" in df.columns:
        print(f"  {yr}: {dict(df['Pedra'].value_counts())}")

print()
print("=== CORRELAÇÕES COM DEFASAGEM (2023) ===")
for c in ["IAA", "IEG", "IPS", "IDA", "IAN", "IPV", "IPP", "INDE"]:
    if c in d23.columns:
        col = pd.to_numeric(d23[c], errors="coerce")
        corr = col.corr(d23["defasagem"])
        nulls = col.isna().sum()
        print(f"  {c}: corr={corr:.4f}  nulls={nulls}")

print()
print("=== TRAJETÓRIA: 468 alunos em 22∩23∩24 ===")
trio = ra22 & ra23 & ra24
p22 = d22[d22["RA"].isin(trio)][["RA", "defasagem"]].rename(columns={"defasagem": "d22"})
p23 = d23[d23["RA"].isin(trio)][["RA", "defasagem"]].rename(columns={"defasagem": "d23"})
p24 = d24[d24["RA"].isin(trio)][["RA", "defasagem"]].rename(columns={"defasagem": "d24"})
p = p22.merge(p23, on="RA").merge(p24, on="RA")
p["delta_22_23"] = p["d23"] - p["d22"]
p["delta_23_24"] = p["d24"] - p["d23"]
print(f"N={len(p)}")
print("delta 22→23:", dict(sorted(p["delta_22_23"].value_counts().items())))
print("delta 23→24:", dict(sorted(p["delta_23_24"].value_counts().items())))
print()
print(
    "22→23  Piorou:",
    (p["delta_22_23"] < 0).sum(),
    " Estável:",
    (p["delta_22_23"] == 0).sum(),
    " Melhorou:",
    (p["delta_22_23"] > 0).sum(),
)
print(
    "23→24  Piorou:",
    (p["delta_23_24"] < 0).sum(),
    " Estável:",
    (p["delta_23_24"] == 0).sum(),
    " Melhorou:",
    (p["delta_23_24"] > 0).sum(),
)

print()
print("=== ALUNOS COM 2 ANOS (22∩23 ou 23∩24) ===")
pair_23 = ra22 & ra23
p2 = d22[d22["RA"].isin(pair_23)][["RA", "defasagem"]].rename(columns={"defasagem": "d22"})
p2 = p2.merge(d23[d23["RA"].isin(pair_23)][["RA", "defasagem"]].rename(columns={"defasagem": "d23"}), on="RA")
p2["delta"] = p2["d23"] - p2["d22"]
print(
    f"22→23 ({len(p2)} alunos): piorou={(p2['delta'] < 0).sum()} estável={(p2['delta'] == 0).sum()} melhorou={(p2['delta'] > 0).sum()}"
)

pair_24 = ra23 & ra24
p3 = d23[d23["RA"].isin(pair_24)][["RA", "defasagem"]].rename(columns={"defasagem": "d23"})
p3 = p3.merge(d24[d24["RA"].isin(pair_24)][["RA", "defasagem"]].rename(columns={"defasagem": "d24"}), on="RA")
p3["delta"] = p3["d24"] - p3["d23"]
print(
    f"23→24 ({len(p3)} alunos): piorou={(p3['delta'] < 0).sum()} estável={(p3['delta'] == 0).sum()} melhorou={(p3['delta'] > 0).sum()}"
)

print()
print("=== FASE DISTRIBUIÇÃO ===")
for yr, df in years:
    print(f"  {yr}: {dict(sorted(df['Fase'].astype(str).value_counts().items()))}")

print()
print("=== AMOSTRA: aluno completo nos 3 anos ===")
sample_ra = list(trio)[:5]
for ra in sample_ra:
    rows = []
    for yr, df in years:
        row = df[df["RA"] == ra][["year", "Fase", "defasagem", "IAA", "IEG", "IPS", "IDA", "IPV", "IAN", "INDE"]].copy()
        rows.append(row)
    print(pd.concat(rows).to_string(index=False))
    print()
