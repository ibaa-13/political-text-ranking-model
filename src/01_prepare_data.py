"""
Step 1: Load Congressional Record speeches (Stanford hein-daily), keep speeches by
Democrats and Republicans, split them into paragraph-sized chunks, and attach
party, chamber, date and DW-NOMINATE scores.

Input : data/raw/hein-daily/{speeches,descr}_<congress>.txt, <congress>_SpeakerMap.txt
        data/raw/HSall_members.csv  (voteview.com)
Output: data/processed/paragraphs.parquet
        output/tables/summary_stats.csv
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hein-daily"
OUT = ROOT / "data" / "processed"
TAB = ROOT / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

CONGRESSES = ["111", "112", "113", "114"]
MIN_SPEECH_WORDS = 60      # drop short procedural remarks
CHUNK_MIN, CHUNK_MAX = 60, 160  # words per "paragraph"


def read_pipe(path, **kw):
    return pd.read_csv(path, sep="|", encoding="latin-1", quoting=3,
                       on_bad_lines="skip", dtype=str, **kw)


def load_congress(c):
    sp = read_pipe(RAW / f"speeches_{c}.txt")
    ds = read_pipe(RAW / f"descr_{c}.txt", usecols=["speech_id", "date", "word_count"])
    sm = read_pipe(RAW / f"{c}_SpeakerMap.txt")
    df = sp.merge(sm, on="speech_id").merge(ds, on="speech_id")
    df = df[df.party.isin(["D", "R"])]
    df["word_count"] = pd.to_numeric(df.word_count, errors="coerce")
    df = df[df.word_count >= MIN_SPEECH_WORDS]
    df["congress"] = int(c)
    return df


SENT_SPLIT = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")


def chunk_speech(text):
    """Group sentences into chunks of CHUNK_MIN..CHUNK_MAX words."""
    sents = SENT_SPLIT.split(text)
    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if n + w > CHUNK_MAX and n >= CHUNK_MIN:
            chunks.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += w
    if n >= CHUNK_MIN:
        chunks.append(" ".join(cur))
    elif chunks and n > 0:
        chunks[-1] += " " + " ".join(cur)
    return chunks


def attach_nominate(df):
    vv = pd.read_csv(ROOT / "data" / "raw" / "HSall_members.csv",
                     usecols=["congress", "chamber", "state_abbrev", "district_code",
                              "bioname", "nominate_dim1", "nominate_dim2"])
    vv = vv[vv.congress.isin([int(c) for c in CONGRESSES]) & vv.chamber.isin(["House", "Senate"])]
    vv["lastname"] = vv.bioname.str.split(",").str[0].str.upper().str.strip()
    vv["chamber"] = vv.chamber.str[0]
    vv = vv.rename(columns={"state_abbrev": "state"})
    # Senate: congress+state+lastname; House: also district
    key_s = ["congress", "chamber", "state", "lastname"]
    vv_s = vv.drop_duplicates(key_s, keep=False)[key_s + ["nominate_dim1", "nominate_dim2"]]
    out = df.merge(vv_s, on=key_s, how="left")
    # second pass for House members with duplicate names, using district
    miss = out.nominate_dim1.isna() & (out.chamber == "H")
    if miss.any():
        vh = vv[vv.chamber == "H"].copy()
        vh["district"] = vh.district_code.astype(int).astype(str)
        key_h = key_s + ["district"]
        vh = vh.drop_duplicates(key_h, keep=False)[key_h + ["nominate_dim1", "nominate_dim2"]]
        fix = out.loc[miss, key_h].reset_index().merge(vh, on=key_h, how="left").set_index("index")
        out.loc[miss, ["nominate_dim1", "nominate_dim2"]] = fix[["nominate_dim1", "nominate_dim2"]]
    return out


def make_paragraphs(speeches):
    rows = []
    for r in speeches.itertuples(index=False):
        for k, ch in enumerate(chunk_speech(r.speech)):
            rows.append((f"{r.speech_id}_{k}", r.speech_id, r.congress, r.date, r.chamber,
                         r.party, r.speakerid, f"{r.firstname} {r.lastname}", r.state,
                         r.gender, r.nominate_dim1, ch))
    paras = pd.DataFrame(rows, columns=["para_id", "speech_id", "congress", "date", "chamber",
                                        "party", "speakerid", "speaker", "state", "gender",
                                        "nominate_dim1", "text"])
    paras["date"] = pd.to_datetime(paras.date, format="%Y%m%d", errors="coerce")
    paras["n_words"] = paras.text.str.split().str.len().astype("int32")
    for c in ["party", "chamber", "state", "gender"]:
        paras[c] = paras[c].astype("category")
    return paras


def main():
    parts = []
    for c in CONGRESSES:  # one congress at a time to keep memory low
        sp = load_congress(c)
        sp["lastname"] = sp.lastname.str.upper().str.strip()
        sp = attach_nominate(sp)
        print(f"congress {c}: speeches {len(sp):,}, matched to DW-NOMINATE {sp.nominate_dim1.notna().mean():.1%}", flush=True)
        part = make_paragraphs(sp)
        del sp
        f = OUT / f"paragraphs_{c}.parquet"
        part.to_parquet(f, index=False)
        parts.append(f)
        del part
    paras = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    paras.to_parquet(OUT / "paragraphs.parquet", index=False)
    for f in parts:
        f.unlink()
    print(f"paragraphs: {len(paras):,}")

    # Summary statistics table (Data section)
    g = paras.groupby(["congress", "party"])
    stats = pd.DataFrame({
        "speakers": g.speakerid.nunique(),
        "speeches": g.speech_id.nunique(),
        "paragraphs": g.size(),
        "mean_words_per_para": g.n_words.mean().round(1),
    }).reset_index()
    stats.to_csv(TAB / "summary_stats.csv", index=False)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
