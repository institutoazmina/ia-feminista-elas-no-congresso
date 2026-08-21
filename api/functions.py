import pandas as pd
import sys 

sys.path.insert(0, "../")

from pre_processamento.pre_process import format_markdown

def to_proba(item):
    d = {x["label"]: x["score"] for x in item}
    return d.get("LABEL_1", d.get("1", 0.0))


def format_text(row):
    text_template = """Partido: {partido}; Genero: {genero}; UF: {uf}; Conteúdo: {texto}"""
    text = text_template.format(partido=row["partido"], genero=row["genero"], uf=row["uf"], texto=row["textInteiroTeorFormatFill"])
    return text


def df_transform(df):
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["textInteiroTeorFormat"] = df["textInteiroTeor"].apply(format_markdown)
    df["textInteiroTeorFormatFill"] = df["textInteiroTeorFormat"].replace("", pd.NA).fillna(df["ementa"])
    df["genero"] = df["genero"].apply(lambda x: x.upper() if pd.notna(x) else "")
    df["partido"] = df["partido"].fillna("")
    df["text"] = df.apply(format_text, axis=1)
    return df