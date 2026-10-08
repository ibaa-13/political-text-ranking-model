"""
Step 0: Download the raw data (about 700 MB).

  - Congressional Record speeches, congresses 111-114 (Gentzkow, Shapiro & Taddy,
    Stanford "hein-daily"). Only the needed files are pulled out of the 4 GB zip.
  - DW-NOMINATE scores for all members (voteview.com).

Output: data/raw/hein-daily/*, data/raw/HSall_members.csv
"""
from pathlib import Path

import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
ZIP_URL = "https://stacks.stanford.edu/file/druid:md374tz9962/hein-daily.zip"
VOTEVIEW = "https://voteview.com/static/data/out/members/HSall_members.csv"
CONGRESSES = ["111", "112", "113", "114"]

want = ([f"hein-daily/{p}_{c}.txt" for c in CONGRESSES for p in ["speeches", "descr"]]
        + [f"hein-daily/{c}_SpeakerMap.txt" for c in CONGRESSES])
with RemoteZip(ZIP_URL) as z:
    for f in want:
        if (RAW / f).exists():
            continue
        print("downloading", f, flush=True)
        z.extract(f, RAW)

out = RAW / "HSall_members.csv"
if not out.exists():
    out.write_bytes(requests.get(VOTEVIEW, timeout=120).content)
print("done")
