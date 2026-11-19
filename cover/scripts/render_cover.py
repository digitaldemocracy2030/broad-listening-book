# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib", "pyshp", "pillow"]
# ///
"""国勢調査1kmメッシュ人口と地球地図日本の境界データから、
「光るネットワークで描いた日本列島」の表紙イラストを描く。

    uv run cover/scripts/render_cover.py               # 300dpi
    uv run cover/scripts/render_cover.py --dpi 150 -o cover/drafts/preview.png

出力:
  - PNG（塗り足し込み A5: 154×216mm）
  - 同じ投影・配置の海岸線＋県境 SVG（デザイナーの重ね合わせ用）
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapefile
from matplotlib.collections import LineCollection
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parent.parent
MESH_CSV = ROOT / "data" / "mesh1km_pop2020.csv"
GM_DIR = ROOT / "data" / "raw" / "gm" / "gm-jpn-bnd_u_2_1"

PAGE_W_MM, PAGE_H_MM = 154.0, 216.0  # A5 148×210 + 塗り足し3mm
LAT0 = 36.0  # 正距円筒の基準緯度


# ---------------------------------------------------------------- 投影・配置
class Projection:
    """経緯度 → ページ座標(mm, 左上原点)。回転・拡大・平行移動つき。"""

    def __init__(self, scale_mm_per_deg, center_lonlat, page_xy, rotate_deg):
        self.s = scale_mm_per_deg
        self.c = np.array(center_lonlat)
        self.p = np.array(page_xy)
        t = np.deg2rad(rotate_deg)
        self.R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        self.k = np.cos(np.deg2rad(LAT0))

    def __call__(self, lon, lat):
        x = (np.asarray(lon) - self.c[0]) * self.k
        y = -(np.asarray(lat) - self.c[1])
        xy = np.stack([x, y], -1) @ self.R.T * self.s
        return xy + self.p


def keep(lon, lat):
    """描く範囲。小笠原・南鳥島・沖ノ鳥島は除外（参考画像に合わせる）。"""
    lon, lat = np.asarray(lon), np.asarray(lat)
    return (lat > 24.0) & ~((lon > 141.5) & (lat < 33.0))


# ---------------------------------------------------------------- データ読み込み
def load_mesh():
    code, lon, lat, pop = [], [], [], []
    with MESH_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code.append(r["mesh_code"])
            lon.append(float(r["lon"]))
            lat.append(float(r["lat"]))
            pop.append(int(r["pop"]))
    lon, lat, pop, code = map(np.array, (lon, lat, pop, code))
    m = keep(lon, lat)
    return lon[m], lat[m], pop[m], code[m]


def mesh_grid(code):
    """3次メッシュコード → 全国共通の格子座標（行=緯度方向, 列=経度方向）"""
    d = lambda a, b: np.array([int(x[a:b]) for x in code])
    row = d(0, 2) * 80 + d(4, 5) * 10 + d(6, 7)
    col = d(2, 4) * 80 + d(5, 6) * 10 + d(7, 8)
    return row, col


def load_lines(name, use=None):
    r = shapefile.Reader(str(GM_DIR / name), encoding="utf-8")
    use_idx = [f[0] for f in r.fields[1:]].index("use") if use else None
    out = []
    for sr in r.iterShapeRecords():
        if use is not None and sr.record[use_idx] != use:
            continue
        pts = sr.shape.points
        parts = list(sr.shape.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            seg = np.array(pts[a:b])
            if len(seg) >= 2 and keep(seg[:, 0], seg[:, 1]).any():
                out.append(seg)
    return out


# ---------------------------------------------------------------- 陸地マスク
class LandMask:
    """行政区域ポリゴンを経緯度格子に塗り、つながった陸地ごとに島番号を振る。"""

    RES = 0.005  # 度（約500m）
    LON0, LON1, LAT0_, LAT1 = 122.5, 149.5, 23.5, 46.0

    def __init__(self):
        from PIL import ImageDraw
        from scipy.ndimage import distance_transform_edt, label

        w = round((self.LON1 - self.LON0) / self.RES)
        h = round((self.LAT1 - self.LAT0_) / self.RES)
        img = Image.new("1", (w, h), 0)
        d = ImageDraw.Draw(img)
        r = shapefile.Reader(str(GM_DIR / "polbnda_jpn"), encoding="utf-8")
        for shp in r.iterShapes():
            pts = np.array(shp.points)
            parts = list(shp.parts) + [len(pts)]
            for a, b in zip(parts[:-1], parts[1:]):
                d.polygon([tuple(q) for q in self._px(pts[a:b])], fill=1)
        self.land = np.asarray(img, bool)
        self.island, _ = label(self.land, structure=np.ones((3, 3)))
        # 海上に落ちた点も最寄りの陸の島番号を引けるようにする
        _, self.near = distance_transform_edt(~self.land, return_indices=True)

    def _px(self, lonlat):
        lonlat = np.asarray(lonlat)
        return np.stack([(lonlat[..., 0] - self.LON0) / self.RES,
                         (self.LAT1 - lonlat[..., 1]) / self.RES], -1)

    def _ij(self, lon, lat):
        xy = self._px(np.stack([lon, lat], -1))
        j = np.clip(xy[..., 0].astype(int), 0, self.land.shape[1] - 1)
        i = np.clip(xy[..., 1].astype(int), 0, self.land.shape[0] - 1)
        return i, j

    def island_of(self, lon, lat):
        i, j = self._ij(lon, lat)
        return self.island[self.near[0][i, j], self.near[1][i, j]]

    def land_ratio(self, lon0, lat0, lon1, lat1, n=24):
        t = np.linspace(0, 1, n)[None, :]
        i, j = self._ij(lon0[:, None] + (lon1 - lon0)[:, None] * t,
                        lat0[:, None] + (lat1 - lat0)[:, None] * t)
        return self.land[i, j].mean(1)


# ---------------------------------------------------------------- 階層ノード
T1, T2, T3 = 1, 2, 3

# 都道府県庁所在地（地球地図の市区町村名。東京は特別区部をまとめて扱う）
CAPITALS = {
    "01": "Sapporo Shi", "02": "Aomori Shi", "03": "Morioka Shi", "04": "Sendai Shi",
    "05": "Akita Shi", "06": "Yamagata Shi", "07": "Fukushima Shi", "08": "Mito Shi",
    "09": "Utsunomiya Shi", "10": "Maebashi Shi", "11": "Saitama Shi", "12": "Chiba Shi",
    "13": "Tokubetsu Ku", "14": "Yokohama Shi", "15": "Niigata Shi", "16": "Toyama Shi",
    "17": "Kanazawa Shi", "18": "Fukui Shi", "19": "Kofu Shi", "20": "Nagano Shi",
    "21": "Gifu Shi", "22": "Shizuoka Shi", "23": "Nagoya Shi", "24": "Tsu Shi",
    "25": "Otsu Shi", "26": "Kyoto Shi", "27": "Osaka Shi", "28": "Kobe Shi",
    "29": "Nara Shi", "30": "Wakayama Shi", "31": "Tottori Shi", "32": "Matsue Shi",
    "33": "Okayama Shi", "34": "Hiroshima Shi", "35": "Yamaguchi Shi", "36": "Tokushima Shi",
    "37": "Takamatsu Shi", "38": "Matsuyama Shi", "39": "Kochi Shi", "40": "Fukuoka Shi",
    "41": "Saga Shi", "42": "Nagasaki Shi", "43": "Kumamoto Shi", "44": "Oita Shi",
    "45": "Miyazaki Shi", "46": "Kagoshima Shi", "47": "Naha Shi",
}


def municipalities(lon, lat, pop, mask):
    """1kmメッシュを地球地図の市区町村ポリゴンに割り当て、2020年人口と人口重心を集計する。
    戻り値: {市区町村コード: (市区町村名, 人口, 経度, 緯度)}。東京23区は 13100 にまとめる。"""
    from PIL import ImageDraw

    r = shapefile.Reader(str(GM_DIR / "polbnda_jpn"), encoding="utf-8")
    fields = [f[0] for f in r.fields[1:]]
    i_code, i_name = fields.index("adm_code"), fields.index("laa")
    codes, names = [], {}
    img = Image.new("I", mask.land.shape[::-1], 0)
    d = ImageDraw.Draw(img)
    for sr in r.iterShapeRecords():
        code, name = sr.record[i_code], sr.record[i_name]
        if code.startswith("131") and code[2:] != "100":
            code, name = "13100", "Tokubetsu Ku"
        if code not in names:
            names[code] = name
            codes.append(code)
        pts = np.array(sr.shape.points)
        parts = list(sr.shape.parts) + [len(pts)]
        for a_, b in zip(parts[:-1], parts[1:]):
            d.polygon([tuple(q) for q in mask._px(pts[a_:b])], fill=codes.index(code) + 1)
    grid = np.asarray(img)
    i, j = mask._ij(lon, lat)
    mid = grid[mask.near[0][i, j], mask.near[1][i, j]] - 1  # 海上のメッシュは最寄りの陸へ
    w = np.bincount(mid, weights=pop, minlength=len(codes))
    mlon = np.bincount(mid, weights=pop * lon, minlength=len(codes)) / np.maximum(w, 1)
    mlat = np.bincount(mid, weights=pop * lat, minlength=len(codes)) / np.maximum(w, 1)
    return {c: (names[c], w[k], mlon[k], mlat[k]) for k, c in enumerate(codes) if w[k] > 0}


def tier1_cities(muni, top_n, fill_pop=np.inf, fill_km=np.inf):
    """各都道府県の県庁所在地 + 人口上位 top_n の市区町村。
    さらに人口 fill_pop 以上の市を人口順に見て、既存の Tier1 から fill_km 以上離れていれば加える
    （北海道東部のように、拠点のない広い地域を埋めるため）。"""
    out = []
    for pref in sorted(CAPITALS):
        cs = sorted(((v[1], c) for c, v in muni.items() if c[:2] == pref), reverse=True)
        pick = {c for _, c in cs[:top_n]}
        cap = [c for c in muni if c[:2] == pref and muni[c][0] == CAPITALS[pref]]
        assert cap, f"県庁所在地が見つからない: {pref} {CAPITALS[pref]}"
        pick |= set(cap)
        out += sorted(pick)

    xy = {c: km_xy(v[2], v[3]) for c, v in muni.items()}
    for _, c in sorted(((v[1], c) for c, v in muni.items() if v[1] >= fill_pop), reverse=True):
        if c not in out and min(np.linalg.norm(xy[c] - xy[t]) for t in out) >= fill_km:
            out.append(c)
            print(f"  Tier1 補完: {muni[c][0]}（人口 {muni[c][1]:,.0f}）")
    return out


def jitter(rng, lon, lat):
    """メッシュ内でランダムにずらして格子感を消す（1kmメッシュ ≒ 45"×30"）"""
    return (lon + rng.uniform(-22.5, 22.5, len(lon)) / 3600,
            lat + rng.uniform(-15, 15, len(lat)) / 3600)


def km_xy(lon, lat):
    k = np.cos(np.deg2rad(LAT0))
    return np.stack([np.asarray(lon) * k * 111.0, np.asarray(lat) * 111.0], -1)


def find_cities(pop, code, density):
    """人口 density 人以上のメッシュが隣接してつながった塊を都市とみなす。
    戻り値: メッシュごとの都市番号（0=都市外）, 都市ごとの人口"""
    from scipy.ndimage import label

    row, col = mesh_grid(code)
    row -= row.min()
    col -= col.min()
    grid = np.zeros((row.max() + 1, col.max() + 1), bool)
    urban = pop >= density  # density はメッシュごとの配列でもよい
    grid[row[urban], col[urban]] = True
    lab, n = label(grid, structure=np.ones((3, 3)))
    city = np.where(urban, lab[row, col], 0)
    city_pop = np.bincount(city, weights=pop, minlength=n + 1)
    return city, city_pop


def is_hokkaido(lon, lat):
    return (lat > 41.35) & (lon > 139.3)


def tiered_nodes(lon, lat, pop, code, a, rng, mask):
    """Tier1: 県庁所在地+各県の人口上位の市 / Tier2: Tier1を含まない中小都市の重心 / Tier3: 都市外のまばらな点
    北海道は都市がまばらなので、都市とみなす密度・人口の閾値を別に下げる。"""
    hk = is_hokkaido(lon, lat)
    city, city_pop = find_cities(pop, code, np.where(hk, a.hokkaido_density, a.urban_density))
    city_hk = np.bincount(city, weights=hk, minlength=len(city_pop)) > np.bincount(city, minlength=len(city_pop)) / 2
    min_pop = np.where(city_hk, a.hokkaido_small_pop, a.small_pop)
    mxy = km_xy(lon, lat)

    muni = municipalities(lon, lat, pop, mask)
    t1 = tier1_cities(muni, a.t1_top, a.t1_fill_pop, a.t1_fill_km)
    t1lon = np.array([muni[c][2] for c in t1])
    t1lat = np.array([muni[c][3] for c in t1])
    # Tier1 が属する都市（最寄りメッシュの都市）は Tier2 から外す
    from scipy.spatial import cKDTree
    _, near = cKDTree(mxy).query(km_xy(t1lon, t1lat))
    t1_city = set(city[near]) - {0}
    t1pop = np.array([muni[c][1] for c in t1])

    wsum = np.bincount(city, weights=pop, minlength=len(city_pop))
    clon = np.bincount(city, weights=pop * lon, minlength=len(city_pop)) / np.maximum(wsum, 1)
    clat = np.bincount(city, weights=pop * lat, minlength=len(city_pop)) / np.maximum(wsum, 1)
    t2 = np.array([c for c in np.where(city_pop >= min_pop)[0] if c and c not in t1_city], int)

    m = np.where(city == 0)[0]
    w = pop[m].astype(float) ** a.rural_alpha
    idx = rng.choice(m, size=min(a.rural_nodes, len(m)), replace=False, p=w / w.sum())
    t3lon, t3lat = jitter(rng, lon[idx], lat[idx])

    nlon = np.concatenate([t1lon, clon[t2], t3lon])
    nlat = np.concatenate([t1lat, clat[t2], t3lat])
    npop = np.concatenate([t1pop, city_pop[t2], pop[idx]]).astype(int)
    tier = np.concatenate([np.full(len(t1lon), T1), np.full(len(t2), T2), np.full(len(idx), T3)])
    print(f"Tier1 {len(t1lon)}, Tier2 {len(t2)}（うち北海道 {city_hk[t2].sum()}）, Tier3 {len(idx)}")
    return nlon, nlat, npop, tier


def uniform_nodes(lon, lat, pop, code, a, rng):
    """全メッシュから人口の重みで抽選（v2までの方式）"""
    w = pop.astype(float) ** a.alpha
    idx = rng.choice(len(pop), size=min(a.nodes, len(pop)), replace=False, p=w / w.sum())
    nlon, nlat = jitter(rng, lon[idx], lat[idx])
    return nlon, nlat, pop[idx], np.full(len(idx), T2)


# ---------------------------------------------------------------- ネットワーク生成
def delaunay_edges(xy_km):
    tri = Delaunay(xy_km)
    e = np.vstack([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [2, 0]]])
    return np.unique(np.sort(e, 1), axis=0)


def mst_edges(n, e, length):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    g = coo_matrix((length + 1e-6, (e[:, 0], e[:, 1])), shape=(n, n))
    m = minimum_spanning_tree(g).tocoo()
    return np.sort(np.stack([m.row, m.col], 1), 1)


def connect_uniform(nlon, nlat, mask, a):
    """v2: Delaunay の辺のうち、陸上の辺・短い海越えの辺・島ごとの最小全域木を残す"""
    xy = km_xy(nlon, nlat)
    e = delaunay_edges(xy)
    length = np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1)
    isl = mask.island_of(nlon, nlat)
    same = isl[e[:, 0]] == isl[e[:, 1]]
    on_land = mask.land_ratio(nlon[e[:, 0]], nlat[e[:, 0]], nlon[e[:, 1]], nlat[e[:, 1]]) >= 0.9
    keep_e = (same & on_land & (length < a.max_land_km)) | (length < a.max_sea_km)
    return np.unique(np.vstack([e[keep_e], mst_edges(len(xy), e[same], length[same])]), axis=0)


def nearest(src, dst, xy, isl, k=1, max_km=np.inf, sea_km=30, first_max_km=np.inf):
    """src の各点から dst の中で近い k 点を探す。
    同じ島の中で探し、同じ島に相手がいなければ海を越えて sea_km 以内の相手だけにつなぐ。"""
    from scipy.spatial import cKDTree

    out = []
    for island in np.unique(isl[src]):
        s = src[isl[src] == island]
        d = dst[isl[dst] == island]
        across = len(d) == 0
        if across:
            d = dst
        dist, j = cKDTree(xy[d]).query(xy[s], k=min(k + 1, len(d)))
        dist, j = dist.reshape(len(s), -1), j.reshape(len(s), -1)
        for si, row_d, row_j in zip(s, dist, j):
            cand = [(dd, d[jj]) for dd, jj in zip(row_d, row_j) if d[jj] != si]
            for n_, (dd, t) in enumerate(cand[:k]):
                if across and dd >= sea_km:
                    continue
                if (n_ == 0 and dd < first_max_km) or dd < max_km:  # 1本目は first_max_km まで引く
                    out.append((si, t))
    return np.array(out, int).reshape(-1, 2)


def connect_tiered(nlon, nlat, tier, mask, a):
    """戻り値: 辺の配列と、辺の種類（1: T1-T1, 2: T2-T2/T2-T1, 3: T3-T2）"""
    xy = km_xy(nlon, nlat)
    isl = mask.island_of(nlon, nlat)
    i1, i2, i3 = (np.where(tier == t)[0] for t in (T1, T2, T3))

    # Tier1 同士: 同じ島の Delaunay の辺のうち主に陸上を通るもの + 島ごとの最小全域木
    # （海は越えない。北海道・本州・四国・九州・沖縄はそれぞれ独立した網になる）
    e = delaunay_edges(xy[i1])
    length = np.linalg.norm(xy[i1][e[:, 0]] - xy[i1][e[:, 1]], axis=1)
    land = mask.land_ratio(nlon[i1][e[:, 0]], nlat[i1][e[:, 0]], nlon[i1][e[:, 1]], nlat[i1][e[:, 1]])
    same = isl[i1][e[:, 0]] == isl[i1][e[:, 1]]
    keep_e = same & (land >= a.t1_land_ratio) & (length < a.t1_max_km)
    e1 = i1[np.unique(np.vstack([e[keep_e], mst_edges(len(i1), e[same], length[same])]), axis=0)]

    # Tier2: 近い Tier2（k本, 距離上限あり）と最寄りの Tier1（1本）
    e2a = nearest(i2, i2, xy, isl, k=a.t2_k, max_km=a.t2_max_km, sea_km=a.max_sea_km)
    e2b = nearest(i2, i1, xy, isl, k=1, sea_km=a.max_sea_km, first_max_km=a.t2_t1_max_km)
    # Tier3: 最寄りの Tier2（同じ島に Tier2 がなければ Tier1）
    e3 = nearest(i3, np.concatenate([i2, i1]), xy, isl, k=1, sea_km=a.max_sea_km)

    edges = [np.sort(x, 1) for x in (e1, e2a, e2b, e3)]
    kinds = [np.full(len(e1), 1), np.full(len(e2a) + len(e2b), 2), np.full(len(e3), 3)]
    edges, uniq = np.unique(np.vstack(edges), axis=0, return_index=True)
    kind = np.concatenate(kinds)[uniq]
    print(f"edges T1-T1 {len(e1)}, T2 {len(e2a) + len(e2b)}, T3 {len(e3)}")
    return edges, kind


# ---------------------------------------------------------------- ラスタ描画
def render_layer(px_w, px_h, draw):
    """白線/白点を黒地に描いた輝度レイヤ（0..1 float）を返す。"""
    dpi = 100
    fig = plt.figure(figsize=(px_w / dpi, px_h / dpi), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W_MM)
    ax.set_ylim(PAGE_H_MM, 0)
    ax.set_facecolor("black")
    ax.axis("off")
    draw(ax)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(np.float32) / 255
    plt.close(fig)
    return img


def tonemap(v, color):
    """輝度 → 色。暗部は単色、明部は白へ飛ばす。"""
    v = 1 - np.exp(-v)  # ソフトな飽和
    c = np.array(color, np.float32)[None, None, :]
    rgb = v[..., None] * c + np.clip(v - 0.75, 0, None)[..., None] / 0.25 * (1 - c)
    return np.clip(rgb, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("-o", "--out", default=str(ROOT / "drafts" / "cover_network.png"))
    ap.add_argument("--mode", choices=["tiered", "uniform"], default="tiered",
                    help="tiered: Tier1/2/3 の階層ネットワーク, uniform: 人口の重みで一律に抽選（v2）")
    ap.add_argument("--nodes", type=int, default=4000, help="uniform のノード数")
    ap.add_argument("--t1-top", type=int, default=3, help="Tier1 にする各都道府県の人口上位の市区町村数（県庁所在地は別に必ず入る）")
    ap.add_argument("--t1-fill-pop", type=float, default=50_000, help="Tier1 補完の対象にする市の人口下限")
    ap.add_argument("--t1-fill-km", type=float, default=60, help="既存の Tier1 からこれ以上離れた市を Tier1 に補完")
    ap.add_argument("--t2-t1-max-km", type=float, default=80, help="Tier2 から最寄りの Tier1 へ引く線の長さ上限")
    ap.add_argument("--urban-density", type=int, default=1000, help="都市とみなすメッシュ人口（人/km2）")
    ap.add_argument("--small-pop", type=float, default=10_000, help="Tier2 とみなす都市人口の下限")
    ap.add_argument("--hokkaido-density", type=int, default=500, help="北海道で都市とみなすメッシュ人口")
    ap.add_argument("--hokkaido-small-pop", type=float, default=2_000, help="北海道で Tier2 とみなす都市人口の下限")
    ap.add_argument("--rural-nodes", type=int, default=2500, help="Tier3 の点数")
    ap.add_argument("--rural-alpha", type=float, default=0.3, help="Tier3 を選ぶ人口の重み指数")
    ap.add_argument("--t1-land-ratio", type=float, default=0.6, help="Tier1 同士の線に必要な陸上率")
    ap.add_argument("--t1-max-km", type=float, default=250, help="Tier1 同士の線の長さ上限（全域木は除く）")
    ap.add_argument("--t2-k", type=int, default=2, help="Tier2 から近い Tier2 へ引く本数")
    ap.add_argument("--t2-max-km", type=float, default=40, help="Tier2 同士の線の長さ上限")
    ap.add_argument("--alpha", type=float, default=0.6, help="人口の重み指数（小さいほど過疎地にも点が出る）")
    ap.add_argument("--max-sea-km", type=float, default=30, help="海をまたぐ線の長さ上限（同じ島に相手がいない点のみ）")
    ap.add_argument("--max-land-km", type=float, default=80, help="陸上の辺の上限")
    ap.add_argument("--scale", type=float, default=6.8, help="mm/度")
    ap.add_argument("--rotate", type=float, default=0, help="地図の回転（度, 反時計回り）")
    ap.add_argument("--cx", type=float, default=80, help="中心（経度136, 緯度35.5）を置くページ上のx(mm)")
    ap.add_argument("--cy", type=float, default=98, help="同 y(mm)")
    ap.add_argument("--pref", type=float, default=0.25, help="県境の明るさ（0で非表示）")
    ap.add_argument("--color", default="70d6ff", help="基調色（hex）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gain", type=float, default=1.8, help="全体の明るさ")
    a = ap.parse_args()

    proj = Projection(a.scale, (136.0, 35.5), (a.cx, a.cy), a.rotate)
    px_w = round(PAGE_W_MM / 25.4 * a.dpi)
    px_h = round(PAGE_H_MM / 25.4 * a.dpi)
    lw = a.dpi / 300  # 線幅の基準（pt）を dpi に追従

    lon, lat, pop, code = load_mesh()
    rng = np.random.default_rng(a.seed)
    mask = LandMask()
    if a.mode == "tiered":
        nlon, nlat, npop, tier = tiered_nodes(lon, lat, pop, code, a, rng, mask)
        edges, kind = connect_tiered(nlon, nlat, tier, mask, a)
    else:
        nlon, nlat, npop, tier = uniform_nodes(lon, lat, pop, code, a, rng)
        edges = connect_uniform(nlon, nlat, mask, a)
    nxy = proj(nlon, nlat)
    coast = [proj(s[:, 0], s[:, 1]) for s in load_lines("coastl_jpn")]
    pref = [proj(s[:, 0], s[:, 1]) for s in load_lines("polbndl_jpn", use=26)]
    print(f"mesh {len(pop)}, nodes {len(nxy)}, edges {len(edges)}, coast {len(coast)}, pref {len(pref)}")

    # 1) 人口密度の光（夜景のような下地）
    mxy = proj(lon, lat)
    # 解像度で見た目が変わらないよう、300dpi 相当の格子で集計してから拡大する
    from scipy.ndimage import zoom
    bh, bw = round(PAGE_H_MM / 25.4 * 300), round(PAGE_W_MM / 25.4 * 300)
    H, _, _ = np.histogram2d(mxy[:, 1], mxy[:, 0], bins=(bh, bw),
                             range=((0, PAGE_H_MM), (0, PAGE_W_MM)), weights=pop)
    H = np.log1p(H) / np.log1p(H.max())
    if (bh, bw) != (px_h, px_w):
        H = zoom(H, (px_h / bh, px_w / bw), order=1)[:px_h, :px_w]
    city = gaussian_filter(H, a.dpi / 300 * 2.0) * 0.45
    halo = gaussian_filter(H, a.dpi / 300 * 25) * 0.8

    # 2) ネットワークの線
    seg = np.stack([nxy[edges[:, 0]], nxy[edges[:, 1]]], 1)
    if a.mode == "uniform":
        edge_a = 0.35 + 0.65 * np.log1p(np.minimum(npop[edges[:, 0]], npop[edges[:, 1]])) / np.log1p(npop.max())
        edge_lw = np.full(len(edges), 0.4)
        dot_s = 0.3 + 4 * (np.log1p(npop) / np.log1p(npop.max())) ** 2
    else:
        edge_a = np.choose(kind - 1, [1.0, 0.6, 0.25])
        edge_lw = np.choose(kind - 1, [0.7, 0.4, 0.25])
        r = np.clip(np.log(npop / a.small_pop) / np.log(1e7 / a.small_pop), 0, 1)
        dot_s = np.select([tier == T1, tier == T2], [10 + 20 * r, 1.0 + 8 * r ** 1.5], 0.3)
    lines = render_layer(px_w, px_h, lambda ax: ax.add_collection(
        LineCollection(seg, colors=[(1, 1, 1, x) for x in edge_a], linewidths=edge_lw * lw)))

    # 3) ノード
    dots = render_layer(px_w, px_h, lambda ax: ax.scatter(
        nxy[:, 0], nxy[:, 1], s=dot_s * lw ** 2, c="white", alpha=0.9, linewidths=0))

    # 4) 海岸線と県境
    coast_l = render_layer(px_w, px_h, lambda ax: ax.add_collection(
        LineCollection(coast, colors="white", linewidths=0.5 * lw)))
    pref_l = render_layer(px_w, px_h, lambda ax: ax.add_collection(
        LineCollection(pref, colors="white", linewidths=0.35 * lw, linestyles=(0, (2, 2)))))

    glow = gaussian_filter(lines + dots + coast_l, a.dpi / 300 * 6)
    v = (city + halo + lines * 1.1 + dots * 1.6 + coast_l * 1.0 + glow * 1.0
         + pref_l * a.pref)
    color = tuple(int(a.color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    rgb = tonemap(v * a.gain, color)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out, dpi=(a.dpi, a.dpi))
    print(f"-> {out} ({px_w}x{px_h})")

    # 同じ配置の海岸線・県境SVG
    svg = out.with_suffix(".svg")
    def path(segs):
        return "".join("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in s) for s in segs)
    svg.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W_MM}mm" height="{PAGE_H_MM}mm" '
        f'viewBox="0 0 {PAGE_W_MM} {PAGE_H_MM}">\n'
        f'<rect width="100%" height="100%" fill="black"/>\n'
        f'<g fill="none" stroke-linejoin="round">\n'
        f'<path id="coastline" d="{path(coast)}" stroke="#{a.color}" stroke-width="0.2"/>\n'
        f'<path id="prefecture_boundaries" d="{path(pref)}" stroke="#{a.color}" stroke-width="0.12" '
        f'stroke-dasharray="0.6 0.6" opacity="0.6"/>\n'
        f'</g>\n</svg>\n', encoding="utf-8")
    print(f"-> {svg}")


if __name__ == "__main__":
    main()
