# 表紙イラスト

『政治・自治体・企業に広がるブロードリスニング』（インプレス）の表紙用イラストの作業フォルダ。

- `reference/`: 参考資料・ラフスケッチ・イメージ素材
- `drafts/`: 試作・案出し（ファイル名例: `cover_draft_01.png`）
- `final/`: 入稿用の確定版

## データから表紙イラストを生成する

```bash
uv run python cover/scripts/fetch_mesh.py      # 人口メッシュ取得（初回のみ, 約20分）
uv run cover/scripts/render_cover.py           # 300dpi PNG + 同配置の海岸線・県境SVG
uv run cover/scripts/render_cover.py --dpi 150 -o cover/drafts/preview.png   # 確認用
```

地球地図日本の境界データは `cover/data/raw/gm/` に手動で展開している（取得元は下記）。
主な調整パラメータ: `--mode`（tiered: Tier1/2/3の階層ネットワーク, uniform: v2方式）, `--t1-top` `--t1-fill-pop` `--t1-fill-km`（Tier1の選び方）, `--rural-nodes` `--t2-k` `--t2-max-km` `--t2-t1-max-km`（階層ネットワークの密度）, `--hokkaido-density` `--hokkaido-small-pop`（北海道のTier2閾値）, `--nodes`（点の数）, `--alpha`（過疎地への点の配りやすさ）, `--max-sea-km` `--max-land-km`（海をまたぐ線・陸上の線の長さの上限）, `--scale` `--cx` `--cy` `--rotate`（配置）, `--pref`（県境の明るさ）, `--color`, `--gain`, `--seed`。

### 使用データ（出典表記が必要）

- 人口: 総務省統計局「令和2年国勢調査」3次メッシュ（1kmメッシュ）人口及び世帯（e-Stat 統計ID T001140）。全国合計 126,146,099人で確定値と一致を確認済み
  - `cover/data/mesh1km_pop2020.csv` に有人メッシュ176,964件の中心座標と人口を集約
- 海岸線・県境: 国土地理院「地球地図日本」第2.1版ベクタ（境界, 2015年1月1日現在）
  https://www1.gsi.go.jp/geowww/globalmap-gsi/download/data/gm-japan/gm-jpn-bnd_u_2_1.zip
  （国土地理院コンテンツ利用規約）
- 描画対象: 北緯24度以北。小笠原諸島・南鳥島・沖ノ鳥島は除外（`keep()` で変更可）

### 階層ネットワーク（`--mode tiered`, 既定）

- Tier1: 各都道府県の県庁所在地＋人口上位3市区町村（東京23区は1つにまとめる）。さらに人口5万人以上で既存Tier1から60km以上離れた市を補完（釧路・帯広・北見など）。同じ島の中でDelaunay三角分割し、主に陸上を通る線と島ごとの最小全域木を引く
- Tier2: 人口1,000人/km²以上のメッシュの塊で人口1万人以上の中小都市（北海道は500人/km²・2,000人）。近いTier2へ2本（40km以内）と最寄りのTier1へ1本（80km以内）
- Tier3: 都市外のメッシュから抽選したまばらな点。最寄りのTier2（なければTier1）へ薄い線
- 海はまたがない（同じ島に相手がいない点のみ30km以内で接続）。北海道・本州・四国・九州・沖縄はそれぞれ独立した網
- 市区町村の人口・位置は、2020年国勢調査メッシュを地球地図の市区町村ポリゴンに割り当てた人口重心
