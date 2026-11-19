"""e-Stat から令和2年国勢調査の3次メッシュ（1km）人口を全国分取得し、
cover/data/mesh1km_pop2020.csv（mesh_code, lat, lon, pop）にまとめる。

生データ（zip）は cover/data/raw/ に保存する（git管理外）。
"""
import csv
import io
import time
import urllib.request
import zipfile
from pathlib import Path

STATS_ID = "T001140"  # 国勢調査2020 3次メッシュ 人口及び世帯
URL = "https://www.e-stat.go.jp/gis/statmap-search/data?statsId={sid}&code={code}&downloadType=2"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "mesh1km_pop2020.csv"


def fetch(code: str) -> bytes | None:
    path = RAW / f"{code}.zip"
    if path.exists():
        return path.read_bytes()
    try:
        with urllib.request.urlopen(URL.format(sid=STATS_ID, code=code), timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    path.write_bytes(data)
    time.sleep(0.3)
    return data


def mesh_center(code: str) -> tuple[float, float]:
    """3次メッシュコード（8桁）の中心緯度経度"""
    p, u = int(code[0:2]), int(code[2:4])
    q, v = int(code[4]), int(code[5])
    r, w = int(code[6]), int(code[7])
    lat = p / 1.5 + q * (5 / 60) + r * (30 / 3600) + 15 / 3600
    lon = u + 100 + v * (7.5 / 60) + w * (45 / 3600) + 22.5 / 3600
    return lat, lon


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    found = []
    for p in range(36, 69):
        for u in range(22, 54):
            code = f"{p}{u:02d}"
            data = fetch(code)
            if data is None:
                continue
            found.append(code)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                name = [n for n in z.namelist() if n.endswith(".txt")][0]
                text = z.read(name).decode("cp932")
            for line in csv.reader(io.StringIO(text)):
                if not line or not line[0].isdigit() or len(line[0]) != 8:
                    continue
                pop = line[4]
                if not pop.isdigit() or int(pop) == 0:
                    # 秘匿（*）は合算先に計上されているので0扱いでよい
                    continue
                lat, lon = mesh_center(line[0])
                rows.append((line[0], f"{lat:.5f}", f"{lon:.5f}", pop))
            print(code, len(rows), flush=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mesh_code", "lat", "lon", "pop"])
        w.writerows(sorted(rows))
    print(f"1次メッシュ {len(found)} 区画, 有人メッシュ {len(rows)} 件 -> {OUT}")


if __name__ == "__main__":
    main()
