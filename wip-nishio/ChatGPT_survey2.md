了解。**未チェック箇所のうち、Webの一次情報で“確定できたもの”**を中心に、チェックリストを更新します（URLはコードで提示）。

---

## 第5章（都知事選／安野貴博）—「URL確定」まで埋め

### 入口：声はどこから集めた？（URL確定）

* **参加型マニフェスト（GitHub＋閲覧用サイト）**：`manifest.takahiroanno.com` が「都知事選2024マニフェスト（政策オープンソース）」の入口。募集期間も明記。 ([manifest.takahiroanno.com][1])
* **AIあんの（YouTube Liveコメント）＋電話版**：朝日が「YouTube上で24時間」「25日から電話でも音声でやり取り可」と報道。 ([朝日新聞][2])
* **選挙掲示板マップ（ポスター進捗の可視化）**：公式スタッフnoteで Netlify の公開URLが明記。 ([note（ノート）][3])
* **SNS由来の声（X等）**：Xの法人向けAPIで収集→クラスタリング→ChatGPT APIで分析、などがITmediaで説明。 ([ITmedia][4])

**URL一覧（確定）**

```txt
参加型マニフェスト（閲覧用サイト）
https://manifest.takahiroanno.com/

参加型マニフェスト（議論・提案の実体：GitHub Issues/PR）
https://github.com/takahiroanno2024/election2024

選挙掲示板マップ（進捗可視化）
https://anno-poster-map.netlify.app/

AIあんの（YouTube Live配信一覧）
https://www.youtube.com/@%E5%AE%89%E9%87%8E%E8%B2%B4%E5%8D%9A/streams
```

（YouTube URLは公式PRにも明記）([プレスリリース・ニュースリリース配信シェアNo.1｜PR TIMES][5])

---

### 期間・規模：いつ〜いつ、投稿数は？

* **参加型マニフェスト募集期間**：2024/6/20〜7/6（サイトに明記）。 ([manifest.takahiroanno.com][1])
* **AIあんの質問数**：

  * 6/27時点で「回答は6千回超」（朝日）。 ([朝日新聞][2])
  * 「選挙期間中 6,200件以上の質問に回答」（Forbes JAPAN記事）。 ([Forbes JAPAN][6])
* **マニフェスト更新**：講演レポートで「85回のバージョンアップ」と明記。 ([富士通][7])
* **GitHubでの提案規模（集計時点違いあり）**：

  * ITmedia（2024/7/22）では「8日間で157イシュー・69PR、41件反映」と紹介。 ([ITmedia][4])
  * 別のイベントレポートでは「15日間で232イシュー・104PR、85件反映」と記載（時点差の可能性）。 ([corp.aicu.ai][8])

---

### 前処理：個人情報や荒らし投稿はどう扱った？

**参加型マニフェスト側は公開ルールが明文化されています**（これが一番引用しやすい）。

* 注意事項・禁止事項（誹謗中傷、差別、スパム、**プライバシー侵害**等）／違反投稿は非表示、アクセスブロックもあり得る、と明記。 ([manifest.takahiroanno.com][1])
* 「健全な議論が行われていないと判断した場合はスレッドのインタラクション制限」も明記。 ([manifest.takahiroanno.com][1])
* さらに「OpenAI APIを使用した自動モデレーターが巡回」と説明しているページあり。 ([manifest.takahiroanno.com][9])
  **AIあんの側**は、RAGやハルシネーション対策（ダブルチェック等）を技術解説として公開。 ([ITmedia][10])

---

### 成果物：何を作ってどこまで公開？

* **参加型マニフェスト（閲覧サイト＋GitHubで提案可能）**：公開。 ([manifest.takahiroanno.com][1])
* **選挙掲示板マップ**：一般公開（進捗可視化）。 ([note（ノート）][3])
* **AIあんの（YouTube/電話）**：公開運用。 ([朝日新聞][2])
* **ブロードリスニング（X/YouTubeコメントのクラスタリング等）**：実施はITmediaが具体に説明。公開レポートURLまでは、都知事選2024については「公式の固定URL」を見つけ切れず（少なくとも無料で読める一次情報内では）。 ([ITmedia][4])

---

### “効いた瞬間”の具体例（本文に入れやすいエピ）

候補として強いのは「ポスター戦略の転換」：

* 少人数での最短経路貼りや「主要駅前だけ」も検討したが、**高齢者が多い地域やオンラインにアクセスしづらい地域ほどポスターが重要**と判断→ボランティア募集に舵を切り、進捗を地図で可視化して運用した、という流れが記事内にあります。 ([INTERNET Watch][11])

（これ、**“優先順位が変わった瞬間”**として書きやすいです）

---

### “困った瞬間”の具体例（本文に入れやすいエピ）

* 選管から渡される紙資料（記事内で「絶望の袋」）を、スキャン→GPT-4oでOCR→CSV→ジオコーディング…という“アナログの壁”が具体に描写されています。 ([INTERNET Watch][11])

---

### 図版候補（スクショで出せるもの）

* 入口：参加型マニフェストのトップ（注意事項・禁止事項が同一ページにあるので強い） ([manifest.takahiroanno.com][1])
* 可視化：選挙掲示板マップ（ピン色、達成率、ヒートマップGIFなど） ([note（ノート）][3])
* 入口：AIあんの YouTube Live（コメント欄で質問→回答の説明） ([朝日新聞][2])

---

## 第7章（チームみらい）—クラスタ例を「本文に貼れる形」で2本

※「代表意見（生の短文）」は、レポートUI内の「濃い意見」等に載っていますが、こちら側でその欄を安定して取得できず（ページ取得エラーが頻発）。代わりに、**クラスタ名＋クラスタ要約（＝代表意見に相当する要旨）**を一次情報として抜き出します。

### 例1（X言及 7/3〜7/7）

* クラスタ名：**「子育て支援と社会保障制度の持続可能な改革に向けた提言」**（577件） ([kouchou-ai.team-mir.ai][12])
* 要旨（代表意見の要約）：年少扶養控除、障害児支援、現役世代負担、税制改革、教育・文化投資などの具体論が束になって出る。 ([kouchou-ai.team-mir.ai][12])

### 例2（AIあんの質問）

* クラスタ名：**「政治システムの革新と選挙制度の透明性向上に向けた取り組み」**（314件） ([kouchou-ai.team-mir.ai][13])
* 要旨：エンジニアリング思考の導入、選挙制度改革、国会議員の質向上に関する提案が中心。 ([kouchou-ai.team-mir.ai][13])

---

## 第10章（海外潮流：Polis／台湾／BG2050）—BG2050を一次情報で確定

### BG2050の同定（正式に書ける形）

* **地域**：米国ケンタッキー州 *Bowling Green* と *Warren County*
* **プロジェクト名**：BG 2050 Project / BG2050 Initiative（地域の将来ビジョンと戦略策定）
* **実施主体・協働**：Innovation Engine（レポート制作主体）＋Google Jigsaw（Sensemaker）＋Computational Democracy Project（Polisの開発・維持）＋地域リーダー／Warren County側、など。 ([What Could BG Be?][14])

### 「Polisが入力データ」の意味（一次情報でそのまま書ける）

BG2050の公開レポートに、はっきり書いてあります：

* **Polisで公開意見募集（2025/2/14〜3/17）**し、個人情報は取らず、投票（agree/disagree等）を集計。 ([What Could BG Be?][14])
* 出力（レポート）は、トピック分類・要約に **Jigsaw Sensemaker（Gemini等）**を使い、AI要約には注意書きも付ける。 ([What Could BG Be?][14])
* その上で、**ボランティア（listening partners等）がレポートを使って提言→Warren County leadershipへ提出（late 2025予定）**と明記。 ([What Could BG Be?][14])

**一次情報URL（BG2050）**

```txt
公開レポート（集計値・注意書き・次のアクションまで書いてある）
https://report.whatcouldbgbe.com/
```

### Polis一次情報（一般説明）

* Polisの開発・維持団体（CompDem）の公式説明：Polisは「大規模集団の意見を本人の言葉で集め、分析・理解する」OSSだと明記。 ([compdemocracy.org][15])

---

## 残り（Webだけでは埋まりにくい＝取材寄り）

* 第7章「共有：誰にどう渡した？」や「意思決定に直結しなかった理由」は、公開レポートだけだと限界があるので、取材項目として残すのが妥当です。 ([kouchou-ai.team-mir.ai][16])

必要なら次は、**第5章の“参加型マニフェストで実際に何が変わったか”**を、GitHubのPR/Issueログから「代表的な変更1本」に落として（例：before/after差分）本文に挿れる、までやるのが一番効きます。

[1]: https://manifest.takahiroanno.com/?utm_source=chatgpt.com "安野たかひろ：政策リポジトリ"
[2]: https://www.asahi.com/articles/ASS6W3HJSS6WOXIE03XM.html?utm_source=chatgpt.com "街頭で、SNSで、AIでも　中盤戦の都知事選、重鎮議員も激励に [東京都]：朝日新聞"
[3]: https://note.com/annotakahiro24/n/nb7c6d5d5f172?utm_source=chatgpt.com "〖開発からリリースまで4日間〗 選挙ポスターマップ開発の舞台裏（技術編） #安野たかひろ｜安野たかひろスタッフ＠チームみらい〖公式〗"
[4]: https://www.itmedia.co.jp/aiplus/articles/2407/22/news033.html?utm_source=chatgpt.com "生成AI、選挙活動にどう使う？　都知事選に挑んだAIエンジニア・安野たかひろ氏の“デジタル選挙戦略”とは - ITmedia AI＋"
[5]: https://prtimes.jp/main/html/rd/p/000000003.000165325.html?utm_source=chatgpt.com "「AIあんの」YouTube版が再登場！ | 安野たかひろ事務所のプレスリリース"
[6]: https://forbesjapan.com/articles/detail/73736?utm_source=chatgpt.com "都知事選15万票第5位、「令和のダ・ヴィンチ」安野貴博が向かう未開領域とは | Forbes JAPAN 公式サイト（フォーブス ジャパン）"
[7]: https://www.fujitsu.com/jp/group/flm/about/resources/topics/2025/0217-01.html?utm_source=chatgpt.com "〖開催レポート〗Fujitsu 人材育成セミナー 2024 基調講演「未来を創るためのテクノロジーとは ～人と社会のアップデートを目指して～」 : 富士通ラーニングメディア"
[8]: https://corp.aicu.ai/ja/ai-event-20241228?utm_source=chatgpt.com "近未来教育フォーラム(3)安野貴博「AI で世界は変わるのか？都知事選から得た知見を基に」"
[9]: https://manifest.takahiroanno.com/contribution/?utm_source=chatgpt.com "貢献したいあなたへ - 安野たかひろ：政策リポジトリ"
[10]: https://www.itmedia.co.jp/aiplus/articles/2406/28/news199.html?utm_source=chatgpt.com "安野たかひろ氏のAITuber「AIあんの」　技術解説記事を公開　RAG活用＆ハルシネーション対策のダブルチェックなど - ITmedia AI＋"
[11]: https://internet.watch.impress.co.jp/docs/column/chizu3/1617209.html?utm_source=chatgpt.com "こんなデータ、AIエンジニアも絶望するしかなかった!? 「チーム安野」は都内1万4000カ所の都知事選ポスター掲示板をどう攻略していったのか？〖地図と位置情報〗 - INTERNET Watch"
[12]: https://kouchou-ai.team-mir.ai/5f1cd708-0c66-4f0f-8974-a1b22ece3de0/?utm_source=chatgpt.com "チームみらいについてのXでの言及(7/3~7/7) - チームみらい"
[13]: https://kouchou-ai.team-mir.ai/fc7b6c5e-ca79-4c3b-b281-3acc3d1a0cfb/?utm_source=chatgpt.com "AIあんのに寄せられた質問 - チームみらい"
[14]: https://report.whatcouldbgbe.com/?utm_source=chatgpt.com "Report | What Could BG Be"
[15]: https://compdemocracy.org/?utm_source=chatgpt.com "The Computational Democracy Project | The Computational Democracy Project"
[16]: https://kouchou-ai.team-mir.ai/?utm_source=chatgpt.com "チームみらい - 広聴AI(デジタル民主主義2030ブロードリスニング)"
