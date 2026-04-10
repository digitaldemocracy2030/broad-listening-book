# Chapter 13 Reading the Implementation of Broad Listening AI

## 13.1 Learning Objectives for This Chapter

In the previous chapter, we explained the foundational technologies behind Broad Listening AI at a conceptual level. In this chapter, we will examine how those technologies are actually combined and made to function as a single system.

By the end of this chapter, you will be able to:

- Understand the overall processing pipeline of Broad Listening AI
- Grasp the purpose and role of each processing step
- Understand the design trade-offs (accuracy vs. ease of explanation)
- Gain the basic knowledge needed to implement a mini Broad Listening AI system yourself

This chapter is aimed at programmers, but it is structured so that even readers who cannot read code can still understand the flow of processing.

---

## 13.2 A Detailed Look at the Processing Pipeline

Broad Listening AI consists, broadly speaking, of the following steps.

![Broad Listening AI processing pipeline](images/13_pipeline_overview.png)

1. **Extraction**: Use an LLM to split and structure raw opinions into individual issues
2. **Embedding**: Convert each opinion into a contextual vector (see Chapter 12)
3. **Dimensionality Reduction**: Compress into two dimensions with UMAP (see Chapter 12)
4. **Clustering**: K-means
5. **Cluster Integration**: Hierarchically group clusters using Ward’s method (see Chapter 12)
6. **Initial Labeling**: Use an LLM to name the fine-grained clusters
7. **Integrated Labeling**: Use an LLM to name the merged clusters
8. **Summary Generation**: Produce an overall summary

Let us now look at each step in detail.

---

### 13.2.1 ① Extraction

In this step, an LLM is used to extract opinions from the input text and split them into coherent units.

Raw opinions collected through public comments and similar channels are difficult to process as-is. Below is an example of a typical “raw opinion” (generated with ChatGPT).

> How much longer must we local residents continue to be tossed around by the government’s vague stance? There are urgent calls to protect the precious green space along the river and restore an environment where birds and insects can return, yet at the same time there is also a growing pile of demands to attract commercial facilities in front of the station in the name of “creating vibrancy” and increase employment opportunities for young people. On top of that, anguished pleas arrive every day demanding stronger child-rearing support, such as eliminating daycare waiting lists and extending after-school childcare hours, and despite all this, aren’t you doing nothing but offering ad hoc excuses without presenting any concrete plan?

People write emotionally charged text in order to move other people. But writing that “moves the heart” also affects the hearts of those who read it, wears them down, and eventually breaks their spirit. Raw opinions from residents or customers are often emotional in exactly this way. One reason local government public feedback staff and customer support staff are so prone to mental exhaustion is that they must read large volumes of such writing every day.

When an LLM is used to split the opinions, the emotional expressions are removed and the content is organized as follows.

- “The precious green space along the river should be protected, and the environment restored so that birds and insects can return”
- “Commercial facilities should be attracted to the station area to increase employment opportunities for young people”
- “The daycare waiting list should be reduced to zero”
- “Child-rearing support should be strengthened, including extending after-school childcare hours”

An important effect of opinion splitting is that it becomes easier to discover “opinions shared by multiple people.” For example, suppose one person submits a text containing “opinions A, B, and C,” while another submits a text containing “opinions C, B, and D.” If you compare the original texts directly, the commonalities are hard to see. But if you split them into {A, B, C} and {C, B, D}, it becomes clear that “B and C are being asserted by multiple people.”

So what kind of input should be given to the LLM in order to split and structure opinions? Below is an example of a prompt used in Broad Listening AI.

```text
You are a professional research assistant. From the given text,
extract and organize the opinions.

# Instructions
* Return a list of strings in the format shown in the input/output example
  * If necessary, split into two separate opinions.
  * In most cases, it is preferable to keep things as a single argument.
* Output the organized opinions in Japanese

## Input/output example
/human
Citizens need to be educated about AI’s capabilities, limitations, and ethical considerations.
It is also necessary to train people who can provide that education.

/ai
{
  "extractedOpinionList": [
    "Citizens should be educated about AI’s capabilities, limitations, and ethical considerations",
    "People capable of teaching AI-related topics should be trained"
  ]
}
```

The instruction “In most cases, it is preferable to keep things as a single argument” is important. Without it, the LLM tends to split opinions too finely.

Also note that the output format is JSON. To process LLM output programmatically, structured data is needed rather than natural-language prose. Broad Listening AI uses the Structured Output approach explained in Chapter 12 to ensure JSON output, while also showing an output example in the prompt so that the intended data structure is more reliably produced.

The word “opinion” in the prompt can also be changed depending on the purpose—for example, to “complaint,” “request,” or “proposal.” This makes it possible to extract only text with a particular character.

---

### 13.2.2 ② Embedding

In this step, the split opinions are converted into vectors (arrays of numbers). The technical details of embeddings were explained in Chapter 12. Broad Listening AI uses OpenAI’s Embeddings API and the Sentence Transformers library, which grew out of open-source research on Sentence-BERT.

By splitting and structuring opinions in the previous step, embedding quality improves significantly. There are two reasons for this.

The first is embedding stabilization. Because each sentence now contains only one opinion, the vector can represent the meaning of that opinion more accurately. If a sentence containing multiple opinions is vectorized as-is, the embedding vector also becomes mixed and ambiguous, making it harder for downstream clustering to group it correctly.

The second is expression normalization. LLM-based restructuring standardizes tone and corrects typos. In the example from the previous section, aggressive wording is cleaned up and the issues are extracted in a clearer form. However, it is important to note that this normalization also removes emotional expression. In public comment analysis, it is often appropriate to remove emotion and extract only the substantive issue, but in use cases such as analyzing event feedback, positive emotion itself may be important information. What to extract and what to exclude should be adjusted in the prompt according to the purpose of the analysis.

In this way, inserting an LLM-based opinion-splitting and normalization step allows embeddings to be used under near-ideal conditions.

That said, there are also concerns. Because of LLM hallucinations, content not present in the original text may be extracted. One possible countermeasure is to build a mechanism that verifies the extracted results against the source text.

---

### 13.2.3 ③ Dimensionality Reduction

In this step, high-dimensional embedding vectors are compressed into two dimensions. Broad Listening AI uses UMAP, as explained in Chapter 12.

The embeddings generated in the previous step are high-dimensional vectors: 1,536 dimensions with OpenAI’s Embeddings API and 768 dimensions with Sentence Transformer. The main reason for dimensionality reduction is visualization. Humans can intuitively understand only up to about two or three dimensions, so to display the data as a scatter plot, the dimensionality must be reduced. In Broad Listening AI, clustering is also performed on the two-dimensional data after UMAP, so dimensionality reduction also serves as preprocessing for clustering.

Broad Listening AI specifies cosine distance as the distance function for UMAP. As explained in Chapter 12, cosine similarity is the standard way to measure similarity between embedding vectors. UMAP uses Euclidean distance by default, but by specifying `metric='cosine'`, dimensionality reduction is performed based on the closeness of vector “direction.” This makes semantically similar opinions more likely to be placed near each other in the two-dimensional space.

#### The Challenge of Separating Support and Opposition Within the Same Topic

As explained in Chapter 12, embeddings have a limitation: they are not very good at distinguishing support from opposition within the same topic. As a result, in UMAP + clustering workflows, supporters and opponents of the same topic may end up grouped into the same cluster. Effectively separating support and opposition within a single topic is an algorithmic challenge in embedding-based opinion analysis.

Various approaches are being explored to address this issue, including adding support/opposition information as an auxiliary dimension to the embedding vector, relying directly on LLM semantic understanding instead of embeddings, and using voting data such as Polis. Sensemaker, developed by Google Jigsaw, uses LLMs for topic classification, but for support/opposition judgment it does not rely on text similarity. Instead, it statistically processes Polis voting data (the number of support and oppose votes for each opinion). By using voting patterns, supporters and opponents of the same topic can be distinguished because they exhibit different voting behavior, fundamentally avoiding the limitations of embeddings.

For politicians and government officials who use broad listening, one of the most important tasks is mediating between conflicting opinions. If they can understand exactly what people opposing a policy are concerned about, they can move consensus-building forward by presenting conditions that address those concerns. For example, when looking inside a cluster labeled “opposition to wind power,” the appropriate response differs completely depending on whether many opinions are concerned about noise or about visual impact on the landscape. A view that separates support and opposition provides concrete clues for this kind of mediation. In that sense, separating support and opposition is directly tied to the practical usefulness of broad listening and is an important topic for future research and development. In Section 13.4.4 at the end of this chapter, we will introduce a prototype that tackles this issue.

---

### 13.2.4 ④ Clustering

In this step, K-means is used to automatically group similar opinions. In Broad Listening AI, the user can specify the **number of lower-level clusters**, which corresponds to the number of clusters (k) in K-means.

There are two reasons K-means was chosen. First, it tends to produce clusters of roughly similar size. If cluster sizes are extremely uneven, they become harder to handle in downstream labeling and visualization. Second, because it groups nearby points together on the scatter plot, it is less likely to produce disconnected “islands.”

Here, “islands” refers to the phenomenon where points that are visually far apart end up belonging to the same cluster. TTTC Scatter, the upstream project from which Broad Listening AI was forked, uses Spectral Clustering, and in actual projects this did produce such islands. The figure below shows an example from Nippon TV’s 2024 House of Representatives election special, where TTTC Scatter was used.

![Example of Talk to the City used in Nippon TV’s 2024 House of Representatives election special](images/13_ntv_spectral.png)

Source: [Nippon TV 2024 House of Representatives election special](https://news.ntv.co.jp/static/shugiinsenkyo2024/whole-1022/index.html)

Broad Listening AI switched from Spectral Clustering to K-means as a design decision based on the assumption that its users would be ordinary citizens and politicians. When people unfamiliar with the characteristics of clustering algorithms look at a scatter plot, they will be confused if opinions in distant locations are shown in the same color. With K-means, the result matches the intuition that “opinions close together belong to the same group.”

That said, K-means assumes spherical clusters, so it may miss complex cluster shapes that Spectral Clustering could have detected. Broad Listening AI accepts this trade-off and prioritizes visual clarity. This is a decision made on the assumption that the system will be used as infrastructure for democracy, not as an analysis tool for specialists.

Clustering is performed on the data after UMAP has reduced it to two dimensions. As noted in Chapter 12, UMAP has the limitation that “distances between far-apart points are not meaningful,” but K-means relies on the distance between each point and its centroid, so practical clustering is still possible as long as neighborhood relationships are preserved. Also, by the time dimensionality reduction has been performed, much of the information from the original high-dimensional space has already been lost, so even more sophisticated algorithms would likely yield only limited gains in accuracy. It was therefore judged that simple K-means was sufficient.

---

### 13.2.5 ⑤ Cluster Integration

In this step, the lower-level clusters generated by K-means are hierarchically merged using Ward’s method to produce higher-level clusters. In Broad Listening AI, the user can also specify the **number of higher-level clusters**, which corresponds to where the dendrogram (tree structure) produced by Ward’s method is cut.

For example, if the number of lower-level clusters is set to 20 and the number of higher-level clusters is set to 5, opinions are first divided into 20 fine-grained groups by K-means, and then those are merged into 5 larger groups by Ward’s method. This combination makes it possible to create a hierarchical grouping structure of “large categories (5) → small categories (20).”

Why use this combination? One might wonder, “Why not just use Ward’s method for hierarchical clustering from the start?” However, in standard implementations, Ward’s method has computational complexity of O(n²log n), which becomes impractical when the number of opinions reaches tens of thousands. So the system first uses K-means, whose computational complexity is only O(nk), to divide opinions into lower-level clusters and compute the centroid of each cluster. Then only those centroids are merged using Ward’s method, greatly reducing the computational cost.

---

### 13.2.6 ⑥ Initial Labeling

In this step, names are assigned to the finest-grained clusters.

More specifically, a few opinions are randomly selected from each cluster and passed to the LLM, which generates a label and a description. The LLM extracts the concept shared across multiple opinions and assigns a label that represents the cluster as a whole.

Let us look at a characteristic part of the prompt used here.

```text
You are a data analyst skilled in the KJ method. The user's input consists of
labels gathered into a group. Explain why those labels belong to one group,
and assign a nameplate (label).

For the nameplate, devise a highly specific name that reflects
the concrete issues and characteristics within the group.
```

The key point here is the use of technical terms such as “KJ method” and “nameplate.”

The KJ method is an idea-generation and organization technique devised by the cultural anthropologist Jiro Kawakita. It groups scattered pieces of information and organizes them by assigning each group a “nameplate” (label). Because LLMs have been trained on Japanese academic literature, simply using this terminology helps them generate labels at an appropriate level of abstraction.

The effect of using technical terms is significant. For example, if you tried to explain the rules of Scrabble without using the word “Scrabble,” it would take several hundred characters. But if you simply say “Scrabble,” many English speakers immediately understand the basic idea. The same applies to LLMs: technical terms allow intentions to be conveyed accurately with short instructions. This is one reason people with domain expertise are often better able to use LLMs effectively.

---

### 13.2.7 ⑦ Integrated Labeling

In this step, names are assigned to the higher-level clusters created through hierarchical merging.
The LLM is given the labels and descriptions assigned at the lower level, along with sample texts belonging to those clusters, and generates a higher-level cluster name and description.

```text
You are an expert in data analysis.
We are currently performing hierarchical clustering on text data.
You will be given the titles and descriptions of lower-level clusters (opinion groups),
as well as sample texts from the higher-level cluster to which those clusters belong.
Please create a title and description for the higher-level cluster.

# Instructions
- The merged cluster name should not simply quote the names of the pre-merged clusters; create a new name based on the content.
- The title should include concrete events or actions (e.g., rapid response by region, steady progress in reconstruction planning, effective information sharing and community cooperation)
  - Use concrete expressions whenever possible and avoid abstract ones
    - Avoid abstract expressions such as “diverse opinions”
- Output in the JSON format shown in the example
```

---

### 13.2.8 ⑧ Summary Generation

In the final step, an overall summary of the analysis results is generated.

The summary-generation prompt explicitly enforces brevity.

```text
Keep the summary extremely concise (at most one paragraph, at most four sentences),
and avoid meaningless wording.
```

The reason for imposing a “maximum of four sentences” constraint is that if the summary becomes too long, it takes too much time for someone opening the report to grasp the overall picture.

In Broad Listening AI, the processing results are consolidated into a JSON file and generated as a web page.

---

## 13.3 Hands-On: Building a Mini Broad Listening AI

From here, let us try implementing the core algorithm of Broad Listening AI ourselves. By combining the techniques learned in Chapter 12, the essential processing can be implemented in about 100 lines of code.

### 13.3.1 Environment Setup

First, install the required libraries.

```bash
pip install openai pandas numpy scikit-learn umap-learn matplotlib scipy
```

Please obtain an OpenAI API key in advance (https://platform.openai.com/api-keys). In the code below, the API key is specified directly, but in actual operation you should set it as an environment variable (`OPENAI_API_KEY`) and avoid including it in source code. If code containing an API key is published on GitHub or elsewhere, it may be abused by third parties.

### 13.3.2 Implementing a Mini Broad Listening AI

Below is the minimum code needed to cluster and visualize opinions. Of the pipeline explained in Section 13.2, this mini implementation covers **② Embedding → ③ Dimensionality Reduction → ④ Clustering (K-means only) → ⑥ Initial Labeling**. It omits **① Extraction (LLM-based opinion splitting and normalization)**, **⑤ Cluster Integration (hierarchical merging with Ward’s method)**, **⑦ Integrated Labeling**, and **⑧ Summary Generation**.

```python
"""
Mini Broad Listening AI - Opinion clustering and visualization
"""
import pandas as pd
import numpy as np
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from openai import OpenAI

# Initialize OpenAI and configure fonts
client = OpenAI(api_key="sk-paste-your-api-key-here")
plt.rcParams['font.family'] = 'MS Gothic'  # For Windows. On Mac, use 'Hiragino Sans'

# 1. Prepare data
opinions = [
    "We want more parks",
    "More effort should be put into preserving green spaces",
    "There are not enough places where children can play",
    "We want more commercial facilities near the station",
    "We want a shopping mall",
    "There are too few stores for young people",
    "We want to eliminate daycare waiting lists",
    "We want after-school childcare hours extended",
    "Child-rearing support should be strengthened",
    "There is a shortage of facilities for older adults",
    "We want more barrier-free infrastructure",
    "The quality of welfare services should be improved",
    "Support for employment at local companies should be strengthened",
    "There should be more places where young people can work",
    "We want better remote work infrastructure",
    "The minimum wage should be raised",
]
df = pd.DataFrame({"opinion": opinions})

# 2. Get embeddings, vectorized into 1536 dimensions
response = client.embeddings.create(
    input=df["opinion"].tolist(),
    model="text-embedding-3-small"
)
embeddings = np.array([item.embedding for item in response.data])

# 3. Dimensionality reduction & clustering
coords_2d = umap.UMAP(n_components=2, metric='cosine', random_state=42).fit_transform(embeddings)
df["cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(coords_2d)
df["x"], df["y"] = coords_2d[:, 0], coords_2d[:, 1]

# 4. Label with an LLM
cluster_labels = {}
for cluster_id in sorted(df["cluster"].unique()):
    opinions_in_cluster = df[df["cluster"] == cluster_id]["opinion"].tolist()
    opinions_text = "\n".join([f"- {op}" for op in opinions_in_cluster])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "You are a data analyst skilled in the KJ method. Please assign a specific and concise label (within 10 characters) to the given opinion group. Output only the label."
        }, {
            "role": "user",
            "content": f"Please assign a label to the following opinion group:\n\n{opinions_text}"
        }]
    )
    label = response.choices[0].message.content.strip()
    cluster_labels[cluster_id] = label
    print(f"Cluster {cluster_id}: {label}")

# 5. Visualization
from scipy.spatial import ConvexHull

fig, ax = plt.subplots(figsize=(14, 10))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
for cluster_id in range(4):
    mask = df["cluster"] == cluster_id
    points = df.loc[mask, ["x", "y"]].values
    ax.scatter(points[:, 0], points[:, 1],
               c=colors[cluster_id], marker=markers[cluster_id],
               label=cluster_labels[cluster_id], s=100, alpha=0.7)
    # Enclose with convex hull
    if len(points) >= 3:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], c=colors[cluster_id], alpha=0.3)
        ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                c=colors[cluster_id], alpha=0.1)
    for _, row in df[mask].iterrows():
        ax.annotate(row["opinion"], (row["x"], row["y"]), fontsize=10)

ax.legend(fontsize=12)
ax.set_title("Opinion Visualization (Mini Broad Listening AI)", fontsize=14)
plt.tight_layout()
plt.savefig("mini_kouchou_ai.png", dpi=150)
plt.show()
```

This code performs the following steps.

1. **Data preparation**: Prepare sample opinion data
2. **Embedding**: Convert each opinion into a 1,536-dimensional vector using the OpenAI API
3. **Dimensionality reduction & clustering**: Compress to two dimensions with UMAP and classify into four groups with K-means
4. **Labeling**: Use an LLM to assign a name to each cluster
5. **Visualization**: Display the result as a scatter plot

When you run it, you will get output like the following.

```text
Cluster 0: Facility Gaps by Age
Cluster 1: Local Revitalization
Cluster 2: Stronger Community Support
Cluster 3: Childcare Support
```

![Result of running the mini Broad Listening AI](images/13_mini_kouchou_ai.png)

By extending this basic process, you can build a full-scale system like Broad Listening AI.

### 13.3.3 Execution Cost

Let us also touch on the execution cost of Broad Listening AI. Costs arise in the steps that call the LLM API; local processing such as UMAP and K-means does not incur API charges.

Suppose we use gpt-4o-mini (input $0.15 per 1 million tokens, output $0.60 per 1 million tokens) to process 10,000 opinions. We assume $1 = 150 yen and that one Japanese character is approximately 1.5 tokens. In Tokyo’s “New Tokyo 2050” initiative, about 27,000 opinions were collected, so a scale of 10,000 opinions is realistic for actual use.

The most expensive step is ① Extraction. For each opinion, assume an input of about 150 tokens for the system prompt plus 500 characters of text (750 tokens), and an output of about 200 characters (300 tokens), repeated 10,000 times. The input cost is approximately (150 + 750) × 10,000 ÷ 1,000,000 × $0.15 ≒ $1.35 (about 200 yen), and the output cost is 300 × 10,000 ÷ 1,000,000 × $0.60 = $1.80 (about 270 yen), for a total of about 470 yen.

| Step | Number of API calls | Estimated cost |
|---------|---------------|-----------|
| ① Extraction | 10,000 times (proportional to number of opinions) | About 470 yen |
| ② Embedding | 20,000 items (because each opinion is split into an average of 2 issues during extraction) | About 9 yen |
| ⑥⑦⑧ Labeling and summary | About 60 calls total (proportional to number of clusters) | Only a few yen |
| **Total** | | **About 480 yen** |

More than 90% of the cost comes from ① Extraction. Extraction scales with the number of opinions, but embeddings are orders of magnitude cheaper per token, and labeling and summary generation scale only with the number of clusters, so cost increases only gradually as the number of opinions grows. This estimate is also somewhat conservative; in actual operation, it is not unusual for the total to stay within a few hundred yen. LLM API prices also tend to decline over time, and Broad Listening AI supports local LLMs (such as LM Studio and Ollama), making it possible to reduce API costs to zero.

---

## 13.4 Advanced Topics: Customization Tips

### 13.4.1 Prompt Tuning

The quality of labeling depends heavily on the prompt. Key tuning points include using technical terms such as “KJ method” and “nameplate” as seen in 13.2.5, forcing specificity (“avoid abstract expressions”), setting character limits, and using few-shot learning with good and bad examples.

### 13.4.2 Optimizing the Number of Clusters

The number of clusters has a major impact on the results.

In Broad Listening AI’s default settings, for a number of opinions n, the number of lower-level clusters (fine-grained division by K-means) is calculated as n^(2/3), and the number of higher-level clusters (merged by Ward’s method) is calculated as ∛n (the cube root). Users can use these default values as-is or change them depending on the purpose. For example, for 1,000 opinions, the number of lower-level clusters becomes 1000^(2/3) ≈ 100, and the number of higher-level clusters becomes ∛1000 ≈ 10.

These default values were tuned to generate visually appealing scatter plots that are easy to read and make the overall picture easy to grasp. On the other hand, when policy planners use the system in practice, it may be more effective to increase the number of clusters. If there are only around 10 higher-level clusters, a single cluster may contain too many diverse opinions, making it difficult to translate them into concrete measures. It may therefore be worth using different cluster counts for visualization and for analysis.

### 13.4.3 Experiments in Separating Support and Opposition

As discussed in 13.2.3, support and opposition within the same topic are difficult to separate. In developing the next generation of Broad Listening AI, the author experimented with several prototypes to address this issue. Two of them are introduced here. One extends embeddings; the other avoids embeddings entirely and relies directly on the LLM.

#### Approach 1: Adding a Sentiment Dimension

In the first prototype, at the opinion extraction stage, the LLM is asked to estimate the sentiment of each opinion—the degree of support or opposition—as a score from -1.0 (strong opposition) to +1.0 (strong support). A weighted version of that score is then added as an extra dimension to the embedding vector, and this extended vector is fed into UMAP.

The figure below shows the result of processing about 1,000 public comments on Japan’s electricity policy. Each point represents one opinion; blue points indicate opinions leaning toward support, and red points indicate opinions leaning toward opposition.

<!-- TODO: Since this will be printed in monochrome, replace the figure/table -->

Without sentiment weighting (weight = 0.00), the cluster related to wind power in the lower left contains a mixture of supportive (blue) and opposing (red) opinions.

![No sentiment weighting (weight=0.00)](images/13_sentiment_weight_none.png)

When the weight is set to 0.75, the wind power cluster is separated into supporters and opponents.

![With sentiment weighting (weight=0.75)](images/13_sentiment_weight_applied.png)

This makes it possible to encourage separation of support and opposition while preserving topic similarity, but there is a trade-off: if the weight is too large, topic similarity weakens and all opinions end up arranged only along the support/opposition axis.

#### Approach 2: Discovering Axes of Conflict with an LLM (DivCon)

Another prototype, DivCon (Divide and Conciliate), performs clustering and discovers axes of conflict using only the LLM’s semantic understanding, without using embeddings at all. Like TTTC Turbo and Google Jigsaw’s Sensemaker, it takes the approach of having the LLM read the text directly.

DivCon proceeds as follows.

1. **Topic discovery**: Pass a large volume of opinions to the LLM all at once and extract the main topics
2. **Opinion classification**: Assign individual opinions to topics using the LLM
3. **Conflict-axis discovery**: Sample opinions within a topic and pass them to the LLM to infer what kinds of axes of conflict may exist
4. **Anchor generation**: For each axis of conflict, have the LLM create an extreme pro opinion and an extreme con opinion
5. **Scoring**: Using those generated extreme opinions as reference points, have the LLM estimate how strongly each opinion supports or opposes the issue

This approach fundamentally avoids the support/opposition separation problem of embeddings while also making it possible to automatically visualize the structure of conflict within each topic. Furthermore, by ordering opinions by score, the most hardline positions appear at the top. For policy planners, this helps clarify what the groups with the most urgent and radical views are actually arguing, providing a clue for identifying which issues should be addressed first.

The figure below shows an example of DivCon’s conflict view output. Supportive and opposing opinions are arranged from left to right in order of intensity, making the structure of the conflict immediately visible.

<!-- TODO: Since this will be printed in monochrome, replace the figure/table -->

![DivCon conflict view: opinions are classified by intensity on the left and right sides of the conflict axis](images/13_divcon_opposition_view.png)

Both approaches have strengths and weaknesses, and further research and development will be needed in areas such as setting appropriate weights and improving the accuracy of LLM estimation.

## 13.5 Scatter-Plot Classification vs. Long Context: Two Architectures

The Broad Listening AI pipeline explained in this chapter (Extraction → Embedding → UMAP → Clustering → Labeling) is an architecture that vectorizes opinions and classifies them on a scatter plot. It is designed to prioritize intuitive visualization—“opinions that are close together are similar”—and ease of explanation, so that results can be understood even without specialist knowledge.

One reason this architecture emerged is that early LLMs had context windows limited to 4,096 tokens (about 2,700 Japanese characters). At the time, it was impossible to pass 1,000 opinions to an LLM at once, so similarity judgments between opinions were handled using Sentence-BERT embeddings and cosine similarity.

Since then, LLM context windows have expanded rapidly. Today’s major models include the GPT-4 family (128,000 tokens), the Claude family (200,000 tokens), and the Gemini family (up to over 1 million tokens), making it possible to process thousands to tens of thousands of opinions together. Taking advantage of this evolution, a Long Context architecture—one that simply has the LLM read everything directly, without using embeddings or UMAP—has become a practical option. TTTC Turbo performs clustering by passing many opinions directly to the LLM, while Google Jigsaw’s Sensemaker combines LLM-based topic classification with voting-data-based support/opposition judgment.

Each type has its own characteristics. The scatter-plot type automatically generates an opinion scatterplot using UMAP, allowing intuitive understanding that “close = similar,” but because clustering is performed after reducing the data to two dimensions, clustering accuracy is relatively poor.

By contrast, the Long Context type can naturally separate support and opposition within the same topic through the LLM’s contextual understanding, can more easily distinguish textual nuance, and offers higher clustering accuracy, but it is difficult to produce an attractive scatter plot.

Both types are developing rapidly, and their respective strengths and weaknesses may change in the future. If possible, the ideal approach is to try both and compare them, using each according to the analysis target and purpose.

---

## Chapter Summary

In this chapter, we explained the processing pipeline and design philosophy of Broad Listening AI. Its design philosophy prioritizes ease of explanation over accuracy, emphasizing that “opinions that are close together are similar” should be intuitively understandable. It is designed as infrastructure for democracy, with the goal of being usable even by people without specialist knowledge.

By combining the technologies learned in Chapter 12, the essential processing can be implemented in about 100 lines of code. The source code for Broad Listening AI is all published as open source, so once you understand the processing flow explained in this chapter, you can customize and extend it in your own way.
