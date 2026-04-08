# Chapter 12 Core Technologies Behind Broad Listening

## 12.1 Learning Objectives for This Chapter

In this chapter, we explain the foundational technologies that support broad listening AI in a way that can be understood even without specialized knowledge of data science.

By the end of this chapter, you will understand the following:

- How computers handle the “meaning” of words
- The history of the development of technologies that convert text into numbers (vectors)
- Why large language models (LLMs) can behave in ways that resemble “intelligence”
- Technologies for organizing and visualizing large amounts of data
- How these technologies are combined in broad listening AI

For each technology, we will explain it from the perspective of “what goes in, what comes out, and what it can be used for.”

---

## 12.2 The Big Picture of the Technology

The technologies used in broad listening AI are rooted in philosophical questions such as: What does it mean for things to be “the same”? What does it mean for things to be “similar”?

Humans can intuitively understand that “cat” and “kitty” refer to the same thing. We also understand that while “dog” and “cat” are different, they are similar in that both are “pets.” But for a computer, these are nothing more than completely different strings of characters. How can we make a computer understand this “identity of meaning” and “similarity”? Many techniques in computer science were born from attempts to answer this question.

Computer science has approached this question in three different ways. The figure below shows the overall landscape of the technologies explained in this chapter.

![Relationship diagram of the core technologies that make up broad listening](images/12_tech_overview.png)
*Figure 12-1: Relationship diagram of the core technologies that make up broad listening*

The left side of the figure represents the approach that **“things used in similar ways have similar meanings.”** If we learn from a large amount of text how the word “cat” is used in context, we can discover that “kitty” and “feline” are used in similar contexts. In the past, people manually created synonym dictionaries and registered correspondences such as “cat = kitty = feline.” But this idea evolved: Word2Vec made it possible to automatically acquire word vectors from surrounding context, BERT enabled context-aware word vectors (different vectors for the same word depending on context), Sentence-BERT made it possible to vectorize the context of entire sentences, and GPT (LLMs) further developed this into text generation.

The right side of the figure represents the approach that **“things that are close in distance are similar.”** Data is represented numerically as vectors, and similarity between data points is judged by measuring distance. Cosine similarity quantifies closeness, Ward’s method and K-means group nearby items together, and UMAP projects high-dimensional data down into two dimensions that humans can view.

Slightly off to the side is the approach that **“values that are correlated are similar.”** PCA (principal component analysis) compresses dimensions by finding the directions in which the data spreads the most—that is, the directions with the strongest correlation.

Broad listening AI is realized by combining these approaches. First, the input text is cleaned and formatted using an LLM from the left-side family of technologies, and Sentence-BERT converts the words into vectors. Next, right-side technologies (cosine similarity, clustering, and UMAP) measure distances, group similar items together, and visualize them. Then the process returns to the left side again, where an LLM assigns labels. This is the basic mechanism of broad listening AI.

UMAP appeared in 2018, Sentence-BERT in 2019, and the practical deployment of LLMs began in 2023 and after. It was only a few years ago that all of the technologies needed to build broad listening AI finally came together. In other words, broad listening AI is a new method that has only just become technically possible.

---

## 12.3 Vectorizing the Meaning of Words: From Word2Vec to Sentence-BERT

### 12.3.1 From Synonym Dictionaries to Word2Vec

For humans, “cat” and “kitty” refer to the same thing. But for computers, characters are nothing more than sequences of numbers, so “cat” and “kitty” are completely different data. So how can we make a computer understand the closeness of words?

The conventional approach was to manually maintain a correspondence table of synonyms. This meant painstakingly registering relationships one by one, such as “PC” = “personal computer” and “automobile” = “car” = “vehicle.” However, manual dictionary maintenance had fundamental limits: the coverage problem caused by new words appearing every day, the context problem of not being able to tell whether “Apple” refers to the fruit or the IT company, and the scalability problem of needing separate dictionaries for each language.

The breakthrough came with **Word2Vec**, announced by Google in 2013. It implemented the linguistic distributional hypothesis—“words used in similar surrounding contexts probably have similar meanings”—through large-scale data and machine learning. For example, if we analyze a large amount of text, we find that words like “cat,” “kitty,” and “feline” appear in contexts such as “I have a ___ as a pet” or “The ___ curled up on my lap.” From such patterns, Word2Vec automatically learns relationships between words without being explicitly taught by humans.

Word2Vec represents words as points (vectors) in a high-dimensional space. For example, “cat” might be represented as a combination of 300 numbers—a 300-dimensional vector—and words used in similar contexts are placed near one another. In practice, if we calculate the words closest to “cat,” we find “kitty,” “feline,” and “kitten” near the top, with “rabbit” and “dog” also placed nearby.

### 12.3.2 Cosine Similarity: Measuring “Closeness” Between Vectors

Once words can be represented as vectors, we can numerically measure “how similar two words are.”

![How distance between vectors is measured](images/12_vector_distance.png)
*Figure 12-2: How distance between vectors is measured*

There are several ways to measure closeness between vectors, but for word vectors, **cosine similarity** is commonly used. This is a metric that measures how similar the “directions” of two vectors are, and mathematically it takes values in the range from -1 to 1. The closer the value is to 1, the more similar the usage; the closer it is to 0, the less related the words are. In high-dimensional embedding vectors, unrelated vectors tend to be nearly orthogonal, so in practice values are distributed mostly between around 0 and 1. For example, the cosine similarity between “cat” and “feline” might be high at 0.74, while the similarity between “cat” and “Tokyo” would be low.

What matters here is that vectorization transforms a language problem into a mathematical problem. Once “similarity” can be calculated numerically, large amounts of text can be compared and classified automatically.

---

### 12.3.3 The Limits of Word2Vec and the Arrival of BERT (2018)

Word2Vec was a groundbreaking technology, but it had one major limitation. Because the same word always maps to the same vector, it cannot properly handle polysemous words whose meanings change depending on context.

For example, the English word “bank” changes meaning depending on context.

- Sentence 1: "He sat by the bank of the river." (riverbank)
- Sentence 2: "She deposited money in the bank." (financial institution)

Japanese also has many polysemous words. For example, *amai* can mean sweetness in “This cake is sweet,” leniency in “The screening is too lenient,” or over-optimism in “That forecast is too optimistic.” Likewise, *atama* can refer to a body part in “My head hurts,” a leader in “the head of the organization,” or intelligence in “She is smart.” *Kin* can mean money, gold as a metal, or Friday, depending on context.

But with Word2Vec, all of these different meanings of “bank,” *amai*, *atama*, and *kin* collapse into a single vector as long as the word form is the same. If even such simple words are treated without regard to context, there are clear limits to practical natural language processing.

BERT solved this problem. BERT (Bidirectional Encoder Representations from Transformers), announced by Google in 2018, generates different vector representations depending on context by looking at both the preceding and following context of a word at the same time, as suggested by the word “Bidirectional.”

- If the word “river” is nearby → “bank” is represented as a vector meaning “riverbank”
- If “money” or “deposited” is nearby → “bank” is represented as a vector meaning “financial institution”

With the arrival of BERT, Google is said to have significantly improved the accuracy of natural-language search around 2019.

### 12.3.4 From Words to Sentences: The Arrival of Sentence-BERT (2019)

BERT can generate context-aware word vectors, but it was not well suited to efficiently representing an entire sentence as a single vector. To calculate the similarity between two sentences, both sentences had to be input into BERT at the same time, making the computational cost enormous when comparing large numbers of sentences.

This problem was solved by Sentence-BERT (S-BERT). Introduced in 2019, this method modified BERT so that an entire sentence could be efficiently represented as a single vector. As a result, once the vector for each sentence has been computed in advance, similarity can be calculated simply by comparing vectors.

Let us vectorize the following nine sentences and calculate cosine similarity between all pairs (Figure 12-3).

| Category | Sentence |
|---------|------|
| Cooking | I like making pasta with tomato sauce |
| Cooking | I’m good at cooking Italian food |
| Cooking | Spaghetti carbonara is easy to make and delicious |
| Weather | It’s sunny today and the weather feels great |
| Weather | Tomorrow’s forecast says it will rain |
| Weather | The weather looks good this weekend, perfect for going out |
| Technology | The new smartphone has become faster |
| Technology | The latest laptop has better battery life |
| Technology | The sound quality of wireless earbuds is improving |

![Category-wise cosine similarity using Sentence-BERT](images/12_sentence_bert_similarity.png)
*Figure 12-3: Calculating sentence similarity with Sentence-BERT*

Figure 12-3 shows the cosine similarity for every pair among the nine sentences, displayed in matrix form. The diagonal values are 1.00 because each sentence is being compared with itself, but they have been omitted for readability. The colors indicate the level of similarity: red to orange indicates high similarity, while green to yellow indicates low similarity.

Sentences within the same category (for example, the three cooking sentences) show relatively high similarity, around 0.45 to 0.73. By contrast, sentences across different categories (for example, cooking and technology) have lower values, around 0.06 to 0.29. Even without direct word overlap, Sentence-BERT captures semantic relatedness and assigns high similarity to “pasta with tomato sauce” and “Italian food.”

For this calculation, we use the **Sentence Transformers** library, an advanced implementation of Sentence-BERT (discussed in more detail later).

As an aside, this combination of “context vectors” and “cosine similarity” is also the core technology behind vector search. It made it possible to discover “semantically similar documents” that conventional keyword search could not find.

---

**[Column] What Is an Embedding?**

The English word *embedding* means “to embed,” and in Japanese it is sometimes translated as “embedding” or “embedded representation.” But what exactly is being embedded into what?

In short, it means **forcibly bringing things from the world of language into the world of vectors**.

Words are originally discrete symbols. Terms like “cat,” “dog,” and “Tokyo” are independent symbols, and concepts such as “distance” or “direction” do not naturally exist between them. There is no intermediate form like “cat.5” between “cat” and “dog.”

By contrast, the world of vectors is continuous. Between the coordinates (1,1) and (2,2) lies (1.5, 1.5), and between those points there are infinitely many more. The distance between (1,1) and (2,2) can be calculated as √2 ≈ 1.41. In the world of vectors, distance, direction, and angle can all be computed using ordinary arithmetic.

Through machine learning, embeddings transform words and sentences so that **similar words or sentences are placed near one another in a high-dimensional space**. By converting the symbol “cat” into a vector with hundreds of dimensions such as [0.23, -0.15, 0.87, ...], we can measure the distance between “cat” and “dog,” or numerically confirm that “cat” and “feline” are close.

This transformation makes language computable. Addition, subtraction, distance measurement, clustering, visualization—all of these become possible because of embeddings.

There is some variation in terminology for this concept. Depending on the literature or the engineer, you may see expressions such as “embedding,” “embedding vector,” “context vector,” “context embedding vector,” and so on. These all refer, in essence, to the same thing. In this book, we use these terms according to context, but please do not let that cause confusion.

Today, embeddings are easy to use through a variety of services, such as OpenAI’s Embeddings API (1,536 dimensions) and the open-source Sentence Transformers library (768 dimensions). You simply pass in a sentence, and a vector with hundreds or thousands of dimensions is returned. One important point is that **embeddings are model-specific and are not compatible across different models**. Even for the same sentence, different models produce completely different vectors. In broad listening AI, both the OpenAI API and Sentence Transformers can be used interchangeably, but once a model is chosen, it must be used consistently through the end of the analysis.

---

### 12.3.5 The Limits of Embeddings: Difficulty Distinguishing Negation and Support/Opposition

Embeddings have an important limitation. Even when opinions on the same topic are opposed, they tend to be embedded into nearby vectors.

Let us verify this. Using the sentence “Nuclear power plants should be restarted as soon as possible” as the reference sentence, the following are the cosine similarities calculated with Sentence Transformers against various other sentences.

| Cosine Similarity | Comparison Sentence | Relationship |
|------|------|------|
| 0.91 | Nuclear power plants should resume operation promptly | Same meaning (paraphrase) |
| 0.69 | Nuclear power plants should **not** be restarted as soon as possible | Negative sentence |
| 0.63 | I oppose nuclear power; it should be phased out completely | Same topic, opposed |
| 0.66 | Please restart nuclear power plants to lower electricity prices | Same topic, supportive (different wording) |
| 0.50 | We should increase investment in renewable energy | Related topic |
| 0.42 | I want the waiting list for daycare centers eliminated | Unrelated topic |
| 0.33 | I support raising the consumption tax | Unrelated topic |

As expected, the paraphrase (0.91) is the highest. However, opinions that are the exact opposite (0.63–0.69) are judged to be far more “similar” than unrelated topics (0.33–0.42). Even more striking is that “Please restart nuclear power plants to lower electricity prices” (0.66), which supports the same position as the reference sentence but uses different wording, has almost the same similarity score as the opposing opinion “I oppose nuclear power; it should be phased out completely” (0.63). For embeddings, differences in wording matter more than differences in support or opposition.

This stems from the distributional hypothesis introduced in Section 12.3.1. “Promote nuclear power” and “abolish nuclear power” are used in the same context—debates over energy policy—so the model learns them as having “similar meaning.” In the process of learning semantic similarity, the shared topic becomes the dominant signal, while subtle semantic differences such as support versus opposition are buried. This property has important implications when clustering large numbers of opinions. We will discuss this in detail in Chapter 13.

---

## 12.4 Understanding and Generating Text: Large Language Models (LLMs)

### 12.4.1 From BERT to GPT

BERT excels at “understanding” text, but it is not good at “generating” text. GPT (Generative Pre-trained Transformer), by contrast, was designed for generation. Developed by OpenAI, it can generate text, as suggested by the word “Generative” in its name.

GPT’s processing is based on a surprisingly simple principle:

> **Based on the input text, calculate the probability of the “next word.”**

For example, given the sentence "He sat by the bank of the ???":

| Word | Probability |
|------|------|
| river | 35% |
| stream | 25% |
| creek | 15% |
| pond | 6% |
| lake | 6% |
| ... | ... |

The model selects the next word based on these probabilities, appends the selected word, and then predicts the next word again. By repeating this process, it generates a sentence.

Why does merely “predicting the next word” produce abilities that seem worthy of being called “intelligence”? Because to correctly predict the next word, the model must in some sense “understand” the content of the sentence. To predict “Tokyo” after “The capital of Japan is,” it needs geographic knowledge. To continue “According to the Pythagorean theorem,” it needs mathematical knowledge. In the process of learning “next-word prediction” from vast amounts of text, LLMs acquired not only patterns of language but also enormous amounts of knowledge about the world.

As the ability to “predict the next word” was pushed to the extreme, behavior emerged that can reasonably be described as intelligent. The intelligence of today’s LLMs rests on this simple mechanism.

### 12.4.2 Few-Shot Learning: Solving Many Tasks as Fill-in-the-Blank Problems

GPT operates through the simple mechanism of “predicting the next word,” but this mechanism can be used to solve many tasks such as translation and classification. The GPT-3 paper, published in 2020, introduced a method called few-shot learning.

“Few” in few-shot means “a small number.” By showing just a few examples, an LLM can learn a new task. Below is the English-to-French translation example from the GPT-3 paper.

```
Translate English to French:

sea otter => loutre de mer
peppermint => menthe poivrée
plush giraffe => girafe peluche
cheese =>
```

For GPT, this is not a translation problem. It is a fill-in-the-blank problem in which it predicts the word that comes after “cheese =>”. From the three examples, it learns the pattern “English => French” and predicts that “fromage” should come next.

This discovery was groundbreaking. Without building a dedicated translation model, translation could be done simply by showing examples. Classification, summarization, and question answering could all be solved by “showing appropriate examples and converting the task into a fill-in-the-blank problem.”

Because language models are trained on all kinds of text from across the internet, they can draw on broad knowledge, and it is not an exaggeration to say that they possess a kind of “common sense.” As a result, tasks that can be solved with “common sense” may be solvable through prompt engineering. The cheese translation above is one example. Since there is a vast amount of English and French text on the internet, this falls within the range of “common sense” for GPT-3.

However, the “common sense” referred to here means patterns learned from information that exists in large quantities on the internet. By contrast, information not included in the training data—such as “this quarter’s sales for Company X,” “the details of City Y’s ordinance,” or “yesterday’s news”—should be called “knowledge,” and LLMs do not possess it. LLMs have “common sense,” but they lack “knowledge.” This distinction is extremely important when using LLMs.

If only one example is shown, it is called one-shot; if no examples are shown and only an instruction is given, it is called zero-shot. The title of the GPT-3 paper was “Language Models are Few-Shot Learners.”

Let us look at another example.

```
Please classify the following opinions:

“We want more parks” => Environment
“Reduce the daycare waiting list” => Childcare
“Traffic congestion is terrible” => Transportation
“Expand facilities for older adults” =>
```

Because the example categories are simple one-word labels like “Environment,” “Childcare,” and “Transportation,” GPT will also follow that pattern and output a one-word label such as “Welfare.” If the example categories were written in more detail, such as “Environmental and Greening Policy” or “Childcare Support Programs,” the output would likewise become more detailed—perhaps something like “Welfare and Senior Support Programs.” In few-shot learning, the format of the examples determines the format of the output.

### 12.4.3 The Birth of Prompt Engineering

The discovery of few-shot learning fundamentally changed the nature of software development. Previously, solving a new problem required developing a dedicated program. Translation required a translation engine, sentiment analysis required a sentiment model, and classification required a classifier—each built over months by specialist engineers.

But with few-shot learning, tasks could be solved simply by writing prompts. Translation, sentiment analysis, and classification could all be achieved by showing just a few appropriate examples. There was no longer any need to develop dedicated programs.

This new approach—solving tasks by carefully designing prompts—came to be known as **prompt engineering**. What examples should be shown? In what order? What wording should be used in the instructions? Such choices can dramatically change the quality of the output, even with the same model.

Broad listening AI also uses few-shot learning when extracting opinions. By including input-output examples in the prompt, it communicates the expected output format and level of granularity to the LLM (see Chapter 13 for details).

### 12.4.4 From GPT to ChatGPT: The Birth of Conversational AI

The GPT-3 API had been available since 2020, but using it effectively required programming knowledge and prompt engineering skills. Users had to design appropriate few-shot examples and write code to call the API.

Think back to the earlier French translation example. Only engineers who loved this kind of puzzle would write such a strange instruction. As a result, GPT-3 became a topic of discussion among some engineers, but it was not widely known to the general public.

Then, in November 2022, OpenAI released ChatGPT. Ask a question in ordinary language, and you get an answer back in ordinary language. That difference was decisive.

The technical breakthrough that made ChatGPT possible was Instruction Tuning. GPT-3 had been trained to “predict the next word,” but ChatGPT was additionally trained to “generate responses in human dialogue.”

This made natural conversation possible while taking context into account. Even without showing few-shot examples, users could simply say “Please do X,” and the model could carry out the task.

In March 2023, GPT-4 appeared and achieved scores at the level of passing the bar exam. It also passed Japan’s national medical licensing exam, and by early 2025, a model called o3 had surpassed the passing threshold for the University of Tokyo’s most selective science-track entrance exam, including the medical track. Since then, models such as GPT-5 have appeared, and the evolution of LLMs has not slowed; as of early 2026, fierce competition continues.

### 12.4.5 What LLMs Are Good At, Their Limits, and What Hallucination Means

LLMs are not all-purpose. Their strengths and weaknesses are quite clear.

They are good at tasks that rely on “common sense”: summarization and paraphrasing, translation, classification, logical reasoning, and code generation. By contrast, they are weak at tasks that require “knowledge”: the latest news, internal information about specific companies, and specialized domain knowledge not included in the training data.

| Common Sense | Knowledge |
|------|------|
| Information sufficiently included in the training data | Information not included in the training data |
| The sun rises in the east | This quarter’s sales for Company X |
| Water boils at 100°C | Details of City Y’s ordinance |
| The capital of Japan is Tokyo | Today’s exchange rate |
| Light is faster than sound | The latest CPI figure |
| Natsume Soseki was a novelist | Mr. Tanaka’s employer |
| Newton discovered universal gravitation | Mr. Yamada’s alma mater |
| Dogs are mammals | Company A’s customer count |
| The angles of a triangle sum to 180 degrees | City B’s budget allocation |
| Vitamin C is good for colds | The dominant influenza strain this season |
| Exercise is good for health | Trends in infection counts in Prefecture C |

This distinction between “common sense” and “knowledge” is also useful for understanding the difference between LLMs and search engines. The two are two sides of the same coin, with opposite strengths and weaknesses.

|  | LLM | Search Engine |
|------|------|------|
| **Strengths** | Tasks that use “common sense” (summarization, translation, classification, reasoning) | Tasks that find “knowledge” (latest news, specialized information, proper nouns) |
| **Weaknesses** | Tasks that require “knowledge” (latest information, local information, specialized knowledge) | Tasks that require “common sense” (context understanding, summarization, intent interpretation) |
| **Update Frequency** | Once every few months (requires retraining the model) | Every few hours to days (crawlers update automatically) |

In the era of search engines, we could reach “information that someone had published on the web.” Search engines themselves do not possess knowledge, but they guide us to where knowledge is written. That made them seem as if they knew everything.

LLMs, however, do not guide us to information; they generate answers themselves. The problem is that even though they do not possess the necessary “knowledge,” they will still produce some kind of answer when asked. If you ask an LLM “Tell me about X” with the same mindset you use for a search engine, it may return an answer that sounds plausible but is not true. This phenomenon is called hallucination.

Instead of saying “I don’t know,” an LLM may generate something that sounds right. If you use LLMs without understanding this trait, there is a real risk of believing false information.

As an aside, the hallucination problem has been improving in the latest models. In the GPT-5 family of models, alignment work was done to reduce hallucinations. Technically speaking, systems that had previously been trained with a binary choice between “correct” and “incorrect” were given a third option: “abstain” (do not answer), and the models were trained to refrain from answering questions when they lacked confidence.

As a result, they have become more likely to say things like “I don’t know” or “I can’t answer based on this information alone.” Earlier models tended to “say something no matter what,” but models with stronger reasoning ability are increasingly recognizing their own limits and expressing uncertainty.

https://openai.com/ja-JP/index/why-language-models-hallucinate/

That said, hallucinations have not disappeared completely. When using LLMs, it remains important to verify factual information against primary sources.

### 12.4.6 Political Bias in LLMs: Implications for Broad Listening

Another limitation of LLMs is political bias. Because broad listening relies on LLMs for summarizing and classifying opinions, this is not an issue that can be ignored.

Multiple studies have pointed out that today’s major LLMs (GPT, Claude, Gemini, and others) exhibit a liberal-leaning political bias. A 2025 study by Professor Andrew Hall of Stanford University and colleagues[^llm_bias_1] asked 30 political questions to 24 LLMs from eight companies, then had more than 10,000 U.S. respondents evaluate the political orientation of the answers. The results showed that on 18 of the 30 questions, the responses of nearly all LLMs were perceived as leaning left.

[^llm_bias_1]: Stanford Report, "Study finds perceived political bias in popular AI models" (2025) https://news.stanford.edu/stories/2025/05/ai-models-llms-chatgpt-claude-gemini-partisan-bias-research-study

One major factor behind this bias is thought to be RLHF (reinforcement learning from human feedback). The values and political tendencies of the human annotators who rank LLM outputs as “good answers” or “bad answers” are reflected in the model.

In addition, much of the training data for LLMs consists of English-language text, which reflects Western values. Even when opinions are written in Japanese, there is a risk that during summarization or classification they may be rephrased in a more liberal direction, or that conservative opinions may be undervalued.

A practical way to address this issue in broad listening is to avoid dependence on any single LLM. It is important for neutrality to build in the flexibility to compare results across multiple LLMs or use different models for different purposes. This is why broad listening AI adopts an architecture that allows the LLM in use to be switched easily.

### 12.4.7 RAG: Strengthening LLMs with External Knowledge

Broad listening AI does not use RAG, but it is an important technology for understanding LLM-based systems, so we introduce it here.

One response to the hallucination problem is RAG (Retrieval-Augmented Generation).

The mechanism of RAG is simple. It searches a database or the web for information relevant to the user’s question, then passes that information to the LLM to generate an answer. This allows the LLM to answer based on retrieved information, even if it does not already “know” it itself.

All major current LLM services (ChatGPT, Claude, Gemini, and others) are integrated with search engines and can generate answers that include up-to-date information. Microsoft Copilot has gained a large share of the business-user LLM market because it can search internal documents stored in SharePoint. This makes it possible to answer using company-specific knowledge, such as “respond based on our company’s past proposals” or “give advice in light of internal regulations.”

However, if the retrieved information is wrong, the LLM will also generate an answer that reflects that error. It is important not to accept LLM outputs uncritically and to check the source of the information.

### 12.4.8 Structured Output: Integrating LLMs into Systems

In August 2024, OpenAI added Structured Output to its API. This made it possible to have the model return output in a user-specified JSON format.

Why is this important? Because programs cannot directly process natural language.

Even if a program receives the string “We want more parks,” it cannot determine whether that is an opinion about the environment or about welfare. You cannot write a conditional branch like `if opinion == environment`, nor can you count “how many opinions were about the environment.” In raw natural language, it is just a sequence of characters from the program’s point of view.

Structured data, by contrast, can be processed. If the output is JSON such as `{"category": "Environment", "sentiment": "Support"}`, then all kinds of processing become possible: aggregation by category, analysis of support and opposition, routing based on conditions, and so on. To integrate LLMs into systems, their outputs must be converted into structured data.

Previously, however, LLM outputs were natural language strings, and even if you instructed the model to output JSON, it only followed the format about 90% of the time. A module that fails 10% of the time cannot be integrated into a production system.

Structured Output solved this problem. By simply “defining the output data structure (schema) and passing it to the LLM together with the prompt,” you can reliably obtain structured data. For example, if you define a schema such as “opinion category (environment/welfare/economy/education/other), support/opposition, summary,” you can automatically extract structured data from citizens’ free-text responses.

Previously, even by combining complex processing such as regular expressions and morphological analysis, it was difficult to make a system recognize that “We want more parks” and “I’d like to see more parks” mean the same thing. With Structured Output, LLMs can now be treated as “functions that take natural language as input and return structured data in a predefined format.” There is no longer any need to write the processing logic by hand.

As of 2025, Structured Output is supported by all major LLM providers, including OpenAI, Anthropic (Claude), and Google (Gemini).

Broad listening AI also uses Structured Output when extracting opinions and assigning labels. Converting thousands of free-text responses into structured data such as category, sentiment, and summary would not have been possible without Structured Output.

### 12.4.9 Reasoning Models: Improving Accuracy Through Multi-Step Inference

From the latter half of 2024 onward, Reasoning Models began to appear—models that perform multiple rounds of “reasoning” before answering. Major providers have all released them, including OpenAI’s o1 and o3, Anthropic’s extended thinking, and Google’s Gemini Thinking.

Whereas conventional LLMs generate an answer immediately after receiving a question, Reasoning Models analyze the problem, consider multiple possibilities, organize them logically, and then produce an answer.

![How a Reasoning Model works](images/12_reasoning_model.png)
*Figure 12-4: A Reasoning Model builds up inference step by step*

As shown in the figure, a Reasoning Model first generates reasoning from the prompt, then builds further reasoning on top of that, and finally returns only the conclusion that integrates all of the reasoning to the user. This “think before answering” approach has dramatically improved accuracy on tasks that require logical reasoning. However, because it increases processing time and cost, it is most practical to use it selectively in situations where accuracy is especially important.

---

## 12.5 Organizing and Visualizing Data: Clustering and Dimensionality Reduction

As we saw in Section 12.3, embeddings convert text into vectors with hundreds of dimensions. Once text becomes numerical data, we can apply clustering to “group nearby points together” and dimensionality reduction to “compress hundreds of dimensions into two and display them as a scatter plot.” In this section, we explain these technologies, which broad listening AI uses to organize and visualize opinions.

### 12.5.1 What Is Clustering?

Clustering is a technique for automatically grouping similar things together. Let us begin with a simple example.

Figure 12-5 shows 200 points plotted on a plane.

![Distribution of data](images/12_clustering_two_gaussians.png)
*Figure 12-5: Data divided into two groups*

Looking at this figure, you probably recognize two groups: a cluster in the upper left and a cluster in the lower right. To the human eye, the data appears naturally divided into two groups.

So where should we divide them in a “natural” way? As in Figure 12-5, it seems reasonable to draw circles around each cluster.

![Natural grouping](images/12_clustering_two_gaussians_clustered.png)
*Figure 12-6: Natural grouping*

Clustering is the technology that enables a computer to perform this kind of grouping automatically—the kind of grouping that humans recognize intuitively. The English word “cluster” means a bunch or group, and it may help to imagine a bunch of grapes, where similar things are gathered together.

### 12.5.2 Data Shape and Algorithm Selection

Broad listening AI uses two clustering algorithms: K-means and Ward’s method. Why these two? TTTC Scatter, the project from which broad listening AI was forked, used a different algorithm (spectral clustering), but it produced an “island” problem in which opinions far apart on the scatter plot were classified into the same cluster (see Chapter 13 for details). With K-means, the result matches the intuition that “opinions that are close together belong to the same group.”

There are many clustering algorithms, and each is suited to different data shapes. Figure 12-7 shows the results of applying three algorithms (K-means, Ward’s method, and DBSCAN) to four types of data structure.

![Algorithm comparison](images/12_clustering_algorithms_comparison.png)
*Figure 12-7: Differences in clustering results depending on data structure and algorithm (created with reference to the official scikit-learn documentation)*

Several important points become clear from this figure.

| Data Structure | K-means | Ward’s Method | DBSCAN |
|-----------|---------|--------|--------|
| One cluster | Forcibly splits into two | Forcibly splits into two | Recognizes it as mostly one cluster (with possible outliers at the edges) |
| Circular | Cannot separate well | Cannot separate well | Separates cleanly |
| Crescent-shaped | Cannot separate well | Cannot separate well | Separates cleanly |
| Three clusters | Merges two into one | Merges two into one | Cleanly separates into three |

The most important rows are “one cluster” and “three clusters.” If instructed to “split into two,” K-means and Ward’s method will forcibly divide even a single cluster, and if there are three clusters, they may merge them into two. DBSCAN, by contrast, defines clusters as “high-density regions,” so it can recognize the natural structure of the data. However, depending on its parameter settings (the density threshold), points at the edges may be classified as outliers (noise).

In other words, **there is no universally best algorithm**. You need to choose an algorithm that matches the shape of the data. Because the opinion data handled by broad listening AI tends to form “blob-like” distributions in vector space, K-means and Ward’s method are well suited.

### 12.5.3 The K-means Algorithm

K-means is the most widely used clustering method. The “K” represents the number of groups, and you specify in advance how many groups you want to divide the data into.

Let us look at how the algorithm works in Figure 12-8.

![Steps of K-means](images/12_kmeans_steps.png)
*Figure 12-8: How the K-means algorithm works*

1. **Step 0**: We start with the initial data
2. **Step 1**: Two “center points” (★) are placed at random
3. **Step 2**: Each point is assigned to the group of the nearest center point
4. **Step 3**: The center points are moved to the centroid of each group
5. **Step 4-7**: Repeat “assignment → center movement.” Stop when the center points no longer move

The key features of K-means are that it is simple and fast. However, you must decide in advance how many groups to divide the data into. Another important feature is that it can **forcibly split a single cluster into multiple parts**. In the “one cluster” example mentioned earlier, when K-means was asked to divide the data with “K=2,” the cluster was forcibly cut in two. At first glance this may seem like a disadvantage, but as we will see later, broad listening AI takes advantage of this property by first dividing the data finely and then integrating it hierarchically.

### 12.5.4 Hierarchical Clustering (Ward’s Method)

Ward’s method is a type of hierarchical clustering. Unlike K-means, it does not require the number of groups to be specified in advance.

Let us look at how the algorithm works in Figure 12-9.

![Steps of Ward’s method](images/12_ward_steps.png)
*Figure 12-9: How Ward’s method (hierarchical clustering) works*

1. **Step 0**: At first, every point stands alone
2. **Step 1**: The two closest points (A and B) are merged
3. **Step 2-3**: The next closest pairs are merged one after another
4. **Step 4-5**: Groups are then merged with other groups, until eventually everything becomes one group

This process can be represented as a **dendrogram** (tree diagram), as shown in Figure 12-10.

![Dendrogram](images/12_hierarchical_dendrogram.png)
*Figure 12-10: Hierarchical clustering and the dendrogram*

The key feature of a dendrogram is that **the number of groups changes depending on where you cut it**.

- Cut at the red line (distance = 3) → 2 groups
- Cut at the orange line (distance = 1.5) → 4 groups

In other words, once clustering has been performed, you can later adjust it by saying “I want to divide it more finely” or “I want to merge it into larger groups.” This is a major advantage of hierarchical clustering.

### 12.5.5 What Is Dimensionality Reduction?

In broad listening AI, 1,536-dimensional vectors are compressed into two dimensions for visualization. Humans can intuitively understand up to three dimensions, so it is impossible to directly view high-dimensional data as it is. By reducing it to two dimensions, it can finally be visualized as a scatter plot.

Let us think about this “mapping from high dimensions to low dimensions” using familiar examples (Figure 12-10).

![Analogy for dimensionality reduction: fish prints and maps](images/12_dimension_reduction_analogy.png)
*Figure 12-11: Analogy for dimensionality reduction — fish prints and maps*

A fish print is a technique for transferring the shape of a fish, which has a complex curved surface, onto paper (two dimensions). A real fish quickly rots, but a fish print can be preserved for years and compared side by side with others. Thickness and three-dimensionality are lost, but length is preserved, so it is still useful for showing off your catch.

Maps are similar. You cannot carry a globe around with you, but a map can be folded and put in your pocket. You can also hang it on a wall and see the whole world at a glance. But something is sacrificed in exchange for that convenience. In the Mercator projection, angles are accurate, but area is distorted, making Greenland look larger than Africa. Also, on maps centered on Europe, Alaska and the Russian Far East—which are actually adjacent across the Bering Strait—are pulled apart to the far left and far right edges.

Dimensionality reduction is a technology that, in the same way, sacrifices some information while trying to preserve as much of the essential structure as possible. However, information loss is unavoidable. It is important to keep in mind that exceptional cases that deviate from the overall trend may become harder to see.

A classical dimensionality reduction method is PCA (principal component analysis). It reduces dimensions by finding “the direction in which the data spreads the most” and projecting onto that direction.

![Dimensionality reduction with PCA](images/12_pca_2d_to_1d.png)
*Figure 12-12: Dimensionality reduction with PCA (2D → 1D)*

In the context of broad listening, Polis uses PCA to compress participants’ voting patterns into two dimensions and generate a “map of public opinion.”

### 12.5.6 Dimensionality Reduction with UMAP

PCA is well suited to capturing linear correlations, but it has limits when dealing with data that has complex structure. PCA finds “the direction in which the overall variance of the data is greatest,” but that is not necessarily ideal for the goal of “placing similar things close together.”

UMAP (Uniform Manifold Approximation and Projection), introduced in 2018, prioritizes preserving relationships between nearby points. Points that are close in high-dimensional space are placed close together even after being compressed into two dimensions. This makes it possible to create intuitive visualizations in which “similar opinions are placed near one another.” In broad listening AI scatter plots, opinions on the same topic appear grouped together thanks to UMAP.

Let us explain the UMAP algorithm metaphorically. From each data point, attach **rubber bands** to several nearby points in high-dimensional space. Nearby points are connected by an attractive force. At the same time, **repelling magnets** create a repulsive force between randomly chosen distant points. The balance between attraction and repulsion naturally arranges similar things close together and dissimilar things farther apart. Strictly speaking, this is not a physical simulation, but it is a useful intuitive image.

Now let us look at a concrete example to compare PCA and UMAP. We will use the MNIST handwritten digit dataset.

MNIST is a dataset of handwritten digits from 0 to 9 and is widely used in machine learning research. Figure 12-12 shows some samples.

![MNIST samples](images/12_mnist_samples.png)
*Figure 12-13: MNIST handwritten digit dataset*

Each image is a 28×28-pixel grayscale image. In other words, one image is represented by 784 numerical values (the brightness of each pixel). This is what it means to say the data is “784-dimensional.” Even for the same digit, the shape differs depending on who wrote it.

Figure 12-14 compares the result of compressing this 784-dimensional data into two dimensions using PCA and UMAP.

![PCA vs UMAP](images/12_pca_vs_umap.png)
*Figure 12-14: Comparison of PCA and UMAP (MNIST handwritten digit data)*

In the PCA projection on the left, the digits from 0 to 9 are mixed together, making it difficult to tell which digit is where. PCA merely projects onto the direction of greatest variance, so it cannot capture the semantic structure of “placing the same digits near one another.”

In the UMAP projection on the right, the same digits are grouped together, while different digits are placed farther apart. For example, “1” and “0” are clearly separated.

Even more interesting is that **digits that humans often confuse in handwriting are placed next to each other**.

- **0 and 6**: both have rounded shapes and can look similar depending on how they are written
- **2, 3, 5, 8**: all are curve-heavy digits and can resemble one another when written messily
- **4, 9, 7**: all are composed of vertical and diagonal strokes and are easy to confuse in handwriting

This is not a coincidence. UMAP is arranging the points based on similarity in the 784-dimensional image data (28×28 pixels), so digits that are visually similar are naturally placed near one another. This visualization suggests that there is a meaningful correspondence between human intuitive judgments of similarity and mathematical similarity based on pixel data.

Figure 12-15 zooms in on the boundary region where 4s and 9s are mixed. The actual handwritten images are shown at each point.

![Zoomed-in boundary region](images/12_mnist_boundary_zoom.png)
*Figure 12-15: Zoomed-in view of the UMAP boundary region where 4s and 9s are mixed*

The lower part contains many 4s (red boxes), while the upper part contains many 9s (blue boxes). In the middle region, the two are mixed together. Looking at the images, we can see that a 4 with a closed top resembles a 9, while a 9 with an open lower part resembles a 4. UMAP places such “could-be-either” samples appropriately near the boundary.

The important point is that **for ambiguous data like this, the correct answer is itself ambiguous**. The same is true in broad listening AI: opinions that span multiple topics or are open to multiple interpretations are placed near cluster boundaries.

However, care is needed when interpreting UMAP results. Because UMAP prioritizes “placing nearby things close together,” **points that are close can be interpreted as similar, but the distance between far-apart points has no particular meaning**. For example, in Figure 12-13, the cluster for “1” is far from the cluster for “0,” but that does not mean that “1 and 0 are more different than 4 and 9.”

Also, **UMAP axis 1 and axis 2 have no specific meaning**. The orientation and rotation of the axes are arbitrary, so you cannot interpret them as “the farther right, the more X.” What UMAP provides is only relative positional information: similar things gather near one another.

#### A Concrete UMAP Example: Visualizing Word Vectors

In Section 12.3, we introduced Word2Vec. Let us now look at an example in which these high-dimensional word vectors are visualized in two dimensions using UMAP.

Figure 12-16 shows 80 words—animal names, place names, food names, and sports names—converted into 300-dimensional vectors with Word2Vec and then arranged in two dimensions with UMAP.

![UMAP of word vectors](images/12_word2vec_umap.png)
*Figure 12-16: UMAP visualization of word vectors (Word2Vec, 300 dimensions)*

Animals (red circles), place names (blue squares), foods (green triangles), and sports (purple diamonds) each form clear clusters. One particularly interesting detail is that within the animal cluster, livestock such as cows, pigs, sheep, and horses gather on the side closer to the food cluster, while wild animals such as elephants, tigers, and owls are placed farther away. Because livestock frequently appears in food-related contexts such as “eat beef” or “grill pork,” their vectors become closer under the distributional hypothesis.

Broad listening AI uses the same basic approach. Each opinion is converted into a high-dimensional embedding vector, then compressed into two dimensions with UMAP, producing a scatter plot in which semantically similar opinions are placed near one another.

---

In this chapter, we have explained the core technologies that support broad listening AI. In the next chapter, we will look in detail at how these technologies are combined within broad listening AI and function together as a single pipeline, down to the implementation level.
