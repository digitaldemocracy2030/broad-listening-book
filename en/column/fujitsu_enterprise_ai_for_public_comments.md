# Big-Business AI Takes On Public Comments: Fujitsu’s Initiative

## 120,000 Characters in 10 Minutes: Fujitsu’s Demonstration

Fujitsu issued a press release titled, “Using the large language model ‘Takane’ for public comment operations, Fujitsu conducts a proof-of-concept on workflow efficiency with central government ministries and agencies”[^1]. In short, the company worked jointly with central government ministries and agencies on a demonstration project to improve the efficiency of public comment processing.

![Overview of the proof-of-concept](images/11_06_富士通_実証実験フロー.png)
*Figure: Image of the proof-of-concept (Source: Fujitsu press release)*

In public comment operations, highly contentious regulations can attract thousands of submissions. Staff must read each one, determine which provision of the regulation it relates to, classify it as supportive or opposed, and summarize it. Grouping similar opinions together, considering how to respond, and publishing the results can take more than a month. What is supposed to be “an important mechanism through which citizens can directly express their views” has, in practice, become a war of attrition for frontline staff.

The AI handled four tasks: classifying comments as supportive or opposed, summarizing opinions, mapping comments to specific provisions, and analyzing trends. The reported results were straightforward: it processed roughly 120,000 characters of public comments in about 10 minutes. Accuracy in mapping comments to provisions exceeded 80 percent overall. Fujitsu stated that it aims to launch a full-scale service during fiscal 2026.

As for future development, the company says it plans to expand beyond public comment operations to the broader legislative drafting process. It also plans to build “AI workflows” that systematically combine appropriate AI models and tools, and to develop “AI agents” that can autonomously support complex tasks such as researching relevant laws and coordinating with stakeholders, with service rollout also targeted for fiscal 2026.

## Public Comments Turning into DDoS, and DD2030’s Trial and Error

When he read the press release, Nakayama, the developer of the public engagement AI, reached out immediately. That was because this was news that a major corporation and central government ministries and agencies had begun seriously tackling something we had already tried once ourselves.

In recent years, the public comment system has run into serious problems. For example, in the public comment process on a ministerial ordinance revision concerning the reuse of decontaminated soil contaminated with radioactive materials after the Fukushima Daiichi nuclear accident, more than 200,000 comments were submitted[^2]. This happened because large-scale mobilization took place through social media, and huge numbers of comments were sent in by copying and modifying template text.

Because government officials are obligated to review every submission, this functioned, in a sense, like a DDoS attack[^3]. Staff were forced to spend their time removing duplicate comments and classifying and consolidating submissions, leaving less time for substantive policy review. After deduplication, the actual number of distinct comments in this case was about 8,000. In other words, more than 96 percent of the submissions were mass-posted, similar comments.

To confront this problem head-on, DD2030 obtained the original text of public comments through freedom of information requests and conducted experiments in detecting mass submissions, identifying duplicates, and consolidating opinions using natural language processing technology. The thinking was simple: if the volume of comments exceeds what humans can realistically process, then AI is needed to improve efficiency. Adapting the public engagement AI to public comments was already included in its development roadmap.

In other words, the question of “using AI to process large volumes of public comments” had long been an important theme for us as well. That is why Fujitsu’s press release did not feel like someone else’s story. Seeing that a major IT vendor and central government ministries and agencies had begun seriously confronting the same question, we requested an interview.

## What the Interview Revealed: Convergent Evolution in AI

What emerged from the interview was a simple approach. No contextual vectorization, no RAG. No custom application development, and no model tuning. The design treated the task as a clustering problem: which legal provision does a user-submitted comment correspond to? The set of legal provisions and the set of comments were simply pasted directly into the prompt and sent straight to Takane.

![An illustration of Takane determining which legal provision each comment corresponds to. The comment text and legal provisions are shown side by side, with arrows indicating the relationships.](images/11_06_富士通_条項対応付けイメージ.png)
*Figure: How provision mapping works (Source: Provided by Fujitsu)*

They said they had tried RAG, but found that putting both the legal provisions and the comments directly into the prompt produced higher accuracy.

This approach is almost identical to TTTC-turbo and Jigsaw SenseMaker. Those systems also use a design in which the LLM directly classifies opinions. According to the person in charge, they had not directly referred to those earlier examples, but they had arrived at essentially the same design. It is fitting to call this convergent evolution.

In the course of the demonstration, Takane went beyond classification and summarization. It also presented distributions based on the proportions and characteristics of opinions, organized both supporting and opposing views for each issue with concrete examples, analyzed trends in background factors and policy impacts, and generated reports. This suggests the possibility that AI may evolve from being merely a tool into becoming a partner in policy deliberation.

## AI Becoming Embedded in Government Infrastructure

What follows is the author’s analysis based on the interview. The significance of Fujitsu’s move goes beyond simply making public comment processing more efficient. It can be seen as a benchmark for how much work sovereign AI can actually do.

In May 2025, the Digital Agency began operating “Gennai,” a generative AI platform for government employees, and moves are already underway to adopt domestically developed LLMs in government. Fujitsu has positioned Takane as one candidate, and has also announced the start of domestic production of “Made in Japan” sovereign AI servers, in which data does not pass through overseas servers[^4]. In handling confidential information at government institutions with AI, Fujitsu is moving to address the challenges of data sovereignty and national security from both the software and hardware sides.

Against this backdrop, this proof-of-concept for public comment processing can be seen as an important step not just in workflow efficiency, but in evaluating the capabilities of domestically developed LLMs. By showing that major efficiency gains are possible through prompt design alone, without custom applications or complex technologies, this experiment is likely to point the way toward how AI will be used in future government services.

If you have a sufficiently powerful LLM, it can handle complex tasks with a single prompt, even without dedicated software. This offers a glimpse of a future in which AI is embedded in government infrastructure. The era in which AI reads laws and regulations, organizes opinions, and proposes policy drafts has already begun.

---

[^1]: Fujitsu Limited, “Using the large language model ‘Takane’ for public comment operations, Fujitsu conducts a proof-of-concept on workflow efficiency with central government ministries and agencies” (February 3, 2026) https://global.fujitsu/ja-jp/pr/news/2026/02/03-01

[^2]: e-Gov, “Results of the public comment solicitation on the draft ministerial ordinance revising part of the Ordinance for Enforcement of the Act on Special Measures Concerning the Handling of Environmental Pollution by Radioactive Materials Released by the Accident at the Nuclear Power Plant Caused by the 2011 Great East Japan Earthquake and Tsunami on March 11, 2011” https://public-comment.e-gov.go.jp/pcm/1040?CLASSNAME=PCM1040&id=195240105&Mode=1

[^3]: A DDoS (Distributed Denial of Service) attack is a cyberattack in which massive numbers of requests are sent to a server to render a service inoperable. Here, the term is used metaphorically to describe a situation in which a flood of public comment submissions—whether intentional or not—overwhelms the government’s processing capacity and paralyzes operations.

[^4]: Fujitsu Limited, “Start of domestic production of AI servers that realize sovereignty” (February 12, 2026) https://global.fujitsu/ja-jp/pr/news/2026/02/12-01
