# Chapter 1: What Is Broad Listening?

Author: @tokoroten

English translation by Luke Closs

## Purpose of This Book

This book explains “broad listening,” as it has been practiced in Japanese elections and public administration from 2024 to 2025. As Takahiro Anno noted in the preface, broad listening is still a very new technology and a practical activity in its infancy. This book aims to help readers understand broad listening and put it into practice themselves by systematically covering its core concepts, technical mechanisms, and real-world use cases.

This book is intended for readers such as the following:

- **Government officials and policymakers**: people interested in how to gather citizens’ voices and reflect them in policy
- **Politicians and election practitioners**: people exploring new ways to communicate with voters
- **Members of political parties promoting broad listening**: people who want to understand what their party is doing
- **Students and researchers in sociology and political science**: people who want to study new forms of democracy and civic participation
- **Citizens**: people interested in digital democracy as a new form of social participation
- **Engineers and data scientists**: people who want to understand and implement the technologies that support broad listening

## What Is Broad Listening?

### The Trade-off Between “Depth” and “Scale”

Why is it so hard to achieve both “listening broadly” and “listening deeply”? Deep dialogue takes time and labor. Because available resources—time, budget, and personnel—are finite, pursuing depth limits you to a small number of people, while prioritizing participation from many people makes the dialogue shallower. This is the trade-off that has long stood in the way of democratic practice.

Let’s think through a concrete example. Suppose a city of one million residents wants to “hear residents’ views on urban development.”

**Option A: Listen deeply**  
Have in-depth one-hour conversations with 10 residents. You can gain a deep understanding of each person’s feelings and background. But that is only 10 people out of one million. If they say, “We want a park in our neighborhood,” you still do not know whether that reflects the broader public or just those 10 individuals.

**Option B: Listen broadly**  
Send a survey to all one million residents asking, “Do we need a park? Yes/No.” This gives you statistical representativeness. But you do not learn the context: “Why do they want a park?” “What kind of park do they want?” “Should something else take priority over a park in the first place?”

This is the “depth-versus-scale trade-off.” Broad listening is an attempt to overcome that trade-off, but there is no single approach. Some methods gather input broadly from large, unspecified populations, while others involve listening deeply to experts or directly affected stakeholders. This book focuses mainly on the former.

In *Plurality*, a book written by Audrey Tang and others, this problem is expressed by quoting Nobel Prize–winning economist Herbert Simon:

> “A wealth of information creates a poverty of attention.”

There is a limit to how many perspectives any one person can thoughtfully absorb. Even in a discussion among 30 people, it is not easy to deeply understand every participant’s view. Listening to the voices of thousands or tens of thousands of people was, until recently, impossible.

### Broad Listening: A Technology for Hearing Many Voices

In *Plurality*, “broad listening” is positioned as a way to overcome this trade-off.

> “Advances in statistics (often associated with AI) make broad listening possible, allowing millions of people to attend to the essential claims extracted from each other’s views. This technology enables democratic deliberation at large scale.”

Broad Listening is a concept paired with broadcast. If broadcast is a technology for “one person speaking to many” (mass media such as radio and television), broad listening is a technology for “hearing the voices of many people and integrating that collective intelligence into a form people can understand.”

## Three Forms of Broad Listening

The term “broad listening” is used in different ways depending on context. Sometimes it refers to an AI tool (a component) for structuring and visualizing opinions; other times it refers to an entire deliberative cycle that incorporates such tools. In this book, we organize broad listening into the following three forms.

![Three forms of broad listening](images/01_broadlistening_three_types.png)

In this diagram, the left side is “broad listening as a tool,” the center is “broad listening as support for augmented deliberation,” and the right side is “politician- and government-led broad listening.” The next figure shows how these three forms emerged historically.

![The genealogy of deliberative democracy and broad listening](images/01_jukugi_minshushugi_keifu.png)

The starting point is deliberative democracy, proposed by thinkers such as the German philosopher Jürgen Habermas. This political theory, which aims at consensus-building through dialogue, later developed into practical forms of citizen deliberation. Then, with the development of ICT, the concept of “augmented deliberation” emerged to scale up deliberation, which had previously only worked in small groups. In the 2020s, with the arrival of large language models (LLMs) such as ChatGPT, form ②—broad listening as support for augmented deliberation—emerged as a technology for further supporting augmented deliberation with AI. Form ①—broad listening as a tool—then developed by separating out the tool component itself.

Meanwhile, from the tradition of parliamentary democracy came form ③—politician- and government-led broad listening—in which politicians or public agencies use the tools directly without going through a deliberative process. In practice, these forms did not develop in a neat sequence; technology and practice influenced each other and emerged in parallel. But as a matter of intellectual roots, this is a useful way to organize them.

### ① Broad Listening as a Tool

The figure below was presented in a 2023 article by Yasukazu Nishio, one of this book’s co-authors. In that article, which predicted that the arrival of LLMs (ChatGPT) would transform knowledge work, organizational structures, and social structures, he invoked the idea of “broad listening” from *Plurality* and argued that AI-mediated information summarization would strengthen the listening capacity of individuals and organizations.

![Nishio’s broad listening diagram](images/01_broadlistening.png)[^1]

[^1]: Yasumasa Nishio, “Not Subjective or Objective, but From One Subjectivity to Many Subjectivities—Knowledge Co-Creation Using AI = Integrating Individual Perspectives—,” *IPSJ Magazine* Vol.64 No.9, pp.e23 (Sep. 2023) https://ipsj.ixsq.nii.ac.jp/records/227307

This figure became a standard way of explaining broad listening and was widely cited. But as a result, it also created the misunderstanding that “broad listening = this diagram.” For example, people sometimes ask, “Where does Polis fit into this diagram?” Polis is a tool that collects and visualizes opinions at the same time, but this figure only represents one aspect—“AI-based information summarization”—so it does not fit neatly. What this figure shows is only “broad listening as a tool.”

Broad listening as a tool is a technology that uses AI to aggregate, structure, and visualize large volumes of opinions—a “component” that can be used in many different settings.

- **Election platform development**: helping candidates understand voters’ voices and reflect them in policy
- **Public outreach work in local government**: organizing and analyzing public comments and citizen feedback
- **Corporate marketing**: structuring customer feedback and opinions on social media
- **Media visualization of public opinion**: giving an overview of diverse opinions in election coverage and similar contexts
- **Stakeholder analysis for NPOs**: understanding the voices of beneficiaries and related stakeholders

In this way, broad listening as a tool is a general-purpose technology not tied to any one context.

Tools that fall into this category include the following:

| Tool name | Overview |
|----------|------|
| Talk to the City (TTTC) | An open-source tool developed by the AI Objectives Institute. It clusters large volumes of natural-language text using LLMs and visualizes them as a scatterplot |
| Public Listening AI | A Japanese-language broad listening tool developed by Digital Democracy 2030. Built on TTTC, it was turned into a web system so that people can use it without programming skills |
| Sensemaker | An open-source tool developed by Google Jigsaw. Using Gemini, it classifies and summarizes large-scale opinion data by topic and extracts points of agreement and disagreement |
| AffinityBubble | A tool developed by Korea’s pxdAX. It uses AI to analyze large volumes of qualitative data and visualizes them with a hierarchical Voronoi treemap |
| Embedding Atlas | An open-source visualization tool developed by Apple. It projects large-scale embedding data into two dimensions and enables interactive exploration through automatic clustering and labeling |
| Nomic Atlas | A cloud-based visualization platform developed by Nomic AI. It projects large-scale text data into two dimensions using embeddings and enables interactive exploration |

In this book, we mainly use TTTC and Public Listening AI as examples to explain the technical mechanisms and use cases of broad listening as a tool, while also referring to other tools where relevant. Chapter 2 explains in detail the characteristics of broad listening as a tool (such as the analysis of open-ended questions and its nature as qualitative analysis). Chapter 12 covers the underlying technologies in detail, including embeddings (techniques for converting text into numerical vectors), and Chapter 13 explains implementation methods.

### ② Broad Listening as Support for Augmented Deliberation

Broad listening as a tool began attracting attention because practices that incorporated it into the cycle of “augmented deliberation” proved successful. In *Plurality* as well, broad listening appears in the chapter on augmented deliberation as a technology that supports it.

Deliberative democracy, simply put, is the idea that “before deciding by majority vote, we should first have a proper discussion.” It is a form of democracy in which citizens aim for consensus through dialogue and debate. Rather than relying on simple majority rule, it emphasizes participants listening to one another, giving reasons for their views, and sometimes changing their own minds in order to reach better conclusions. But deep dialogue takes time and labor. Because resources are finite, there is an unavoidable limit to how many people can participate at once. This is the “depth-versus-scale trade-off” described at the beginning.

To overcome this limit, efforts emerged to use information and communication technology to scale up deliberation. This is what we call “augmented deliberation.” It expands deliberation, which traditionally only worked in small groups, along **three dimensions: scale, frequency, and format**. It enables discussions among thousands or tens of thousands of people (expanded scale), ongoing deliberation rather than once-a-year events (expanded frequency), and synchronous and asynchronous discussion online and in text, not just face-to-face (expanded format). Broad listening tools emerged with the rise of LLMs as a technology for scaling augmented deliberation even further.

Taiwan’s vTaiwan process and other citizen-participation processes using Polis are representative examples of augmented deliberation. They combine a range of IT tools: collecting a wide range of opinions through the online opinion-aggregation tool Polis, enabling direct dialogue among citizens through video conferences and online forums, and visualizing discussion outcomes on the web in real time. In this model, the government acts only as a facilitator of the process, while citizens autonomously work toward consensus.

Look again at the center panel of the “three forms of broad listening” diagram. It shows a cycle of three elements: “opinion collection,” “deliberation,” and “policy decision.” In the deliberation stage, issues are organized through AI facilitation and broad listening. This is the augmented deliberation cycle. As the cycle turns, it creates an ongoing loop of dialogue: rather than simply collecting opinions and stopping there, the process moves through deliberation and policy decision, which then lead to new rounds of opinion collection.

Tools that support augmented deliberation include the following:

| Tool name | Overview |
|----------|------|
| Polis | An opinion-aggregation tool used in Taiwan’s vTaiwan. Participants post opinions and vote on others’ opinions with agree, disagree, or pass. It visualizes opinion groups based on voting patterns and helps identify points of agreement across groups |
| Agora Citizen Network | An open-source consensus-building platform from France. It inherits the spirit of Polis while improving the bridging-based algorithm and rebuilding the experience with a modern UI |
| JOIN | An administrative platform operated by the Taiwanese government. Citizens can post policy proposals, and if a proposal gains 5,000 supporters within 60 days, the relevant ministry is required to respond within two months. As of December 2023, it had about 1.29 million registered users and more than 10 million cumulative visitors |
| Decidim | An open-source citizen-participation platform developed by the City of Barcelona. It integrates multiple functions, including citizen proposals, participatory budgeting, public debate, and citizen initiatives. It has been adopted by cities and organizations around the world |
| Loomio | An open-source deliberation tool from New Zealand. It allows groups to discuss, make proposals, and vote on a single platform. It is used for decision-making in cooperatives and local governments |
| Consider.it | A deliberation tool that visualizes both pros and cons. Participants express their views on a slider and write reasons for both support and opposition. It allows users to see the distribution of support and opposition, along with reasons, for each issue |
| All Our Ideas | An open-source opinion-collection tool developed by Princeton University. By repeating pairwise comparison voting, it efficiently identifies group priorities across a large number of options |
| Stanford Online Deliberation Platform | A video deliberation platform developed by Stanford University. AI moderators help ensure fairness in speaking opportunities and support structured deliberation in small groups of 8–15 people. It has been used in more than 120 deliberations across over 50 countries |
| Baisoku Kaigi | Baisoku Kaigi, an AI-supported “fast-track meeting” tool, developed by Multiverse Reality, founded by Aoyama, the developer of “Idobata.” Based on participants’ responses, AI generates real-time follow-up questions and automatically structures responses into “points of agreement,” “points of disagreement,” and “unclear points.” It delivers the quality of interviews at the speed of surveys, supporting everything from pre-meeting alignment to automatic generation of deliberation analysis reports. It is used in corporate management meetings and local government resident meetings (discussed in detail in Chapter 11) |
| Idobata | A Japanese-language deliberation platform developed by Digital Democracy 2030. Inspired by Taiwan’s vTaiwan, it includes AI-based opinion organization and visualization functions to support large-scale citizen-participatory policy discussions |
| D-Agree | An online AI discussion system developed by AGREEBIT. It uses AI and multi-agent technology to support discussion facilitation and consensus-building |
| Liqlid | A citizen-participation platform developed by Liquitous. It includes AI-based comment filtering and clustering functions and has been introduced in local governments in Japan |

From a functional perspective, these tools can be classified along multiple dimensions, such as large-scale text analysis, synchronous discussion support, agreement discovery, and process management. In actual projects, multiple tools are often used together. Chapter 3 explains in detail how to combine them and how to operate augmented deliberation in practice. Chapter 10 introduces examples from overseas.

### ③ Politician- and Government-Led Broad Listening

In Japan, however, a different approach took hold first—one that differs from the full cycle of augmented deliberation. In the 2024 Tokyo governor election, Takahiro Anno used broad listening to collect and analyze citizens’ voices. This helped rapidly spread a style in which politicians and political parties take the lead in gathering diverse public opinions and using them in policy development.

For example, Takahiro Anno presents his own policy process in the following diagram.

TODO: Brush up this diagram https://codezine.jp/article/detail/21308
![Politician- and government-led broad listening cycle](images/01_anno_zyukugi_cycle.png)

The Democratic Party for the People and the Japan Innovation Party have also reflected supporters’ voices in policy through their own processes. Furthermore, when the Japan Innovation Party entered the ruling coalition with the Liberal Democratic Party in October 2025, it created a situation in which parties practicing broad listening existed both in government and in opposition.

In particular, in the 2025 House of Councillors election, Team Mirai (referred to as "Team Future" in this book) operated a “talkable platform,” updating its electoral platform through dialogue with voters, while the Japan Innovation Party used broad listening to frame the issue of social insurance premium reform. Broad listening is thus becoming a de facto standard strategy for discovering campaign issues and drafting campaign pledges (see Chapter 6).

The same pattern is spreading in public administration. In Tokyo’s “Shin Tokyo 2050” (“New Tokyo 2050”) project, approximately 27,000 opinions were collected from residents and used in formulating a long-term strategy (see Chapter 5). More and more local governments are also incorporating broad listening into their planning cycles (see Chapter 8).

Although this differs in form from augmented deliberation, in which citizens debate directly with one another, this style—in which politicians and public agencies actively listen to citizens and respond—is consistent with the spirit of deliberative democracy. The case-study chapters beginning with Chapter 4 discuss this in detail.

Also in Japan, as many organizations—including major political parties—have adopted broad listening tools for opinion aggregation and policy discussion, the image of broad listening as “collecting opinions from social media and using them in policymaking” has spread rapidly.

## This Book’s Hidden Concept: A Watered-Down *Plurality*

In *Plurality*, the chapter on “augmented deliberation” is only 19 pages long, and the term “broad listening” appears only three times in that chapter.[^2] The chapter merely cites Talk to The City and Takahiro Anno’s activities in the 2024 Tokyo governor election as examples, without offering a detailed explanation of what broad listening is. Moreover, the book is extremely dense and abstract, and understanding it requires a great deal of prior knowledge.

In addition, as discussed above, “broad listening” is a polysemous term. It is a problem, a tool, part of augmented deliberation, and also a politician- or government-led policy development process. These aspects are all present in the chapter in *Plurality*, but they are not explicitly organized; instead, they are scattered across 19 pages of dense argument. Indeed, even the author of this book only gradually realized the many meanings of the term while writing it.

So in this book, we aim to reorganize the overall picture of broad listening and introduce concrete examples from Japan in 2024 and 2025, so that readers can understand broad listening through real cases.

To put it more casually: *Plurality* is a straight pour of intellectual whiskey, aged for years in the minds of geniuses—far too strong for ordinary people to drink. So let’s make a watered-down version.

[^2]: *Plurality Book*, chapter on augmented deliberation (Japanese edition) https://www.plurality.net/v/chapters/5-4/

## Why Broad Listening, and Why Now?

The idea of gathering many people’s voices and using them in decision-making is not new. But broad listening only began attracting attention as a practical method in the 2020s. With the spread of the internet, tools for augmented deliberation such as Polis, and initiatives such as vTaiwan that used them, already existed. However, the work of reading and structuring large volumes of open-ended text still required enormous human effort. Broad listening did not become possible simply because AI got smarter. It became possible only because the following technologies matured in quick succession and were combined.

- **Sentence-BERT** (2019): a technology for converting the meaning of sentences into vectors (arrays of numbers)
- **UMAP** (2018): a technology for compressing high-dimensional data into lower dimensions (usually two or three) for visualization
- **LLMs** (2020–): technologies for understanding, summarizing, and generating large volumes of text

In particular, OpenAI’s release of ChatGPT in November 2022 marked a turning point in the history of natural language processing. Tasks that computers had previously struggled with—reading, understanding, and structuring large volumes of natural language—became possible at realistic cost and speed. Talk to the City and Public Listening AI, which are central to this book, were developed using the ChatGPT API. It became possible to collect thousands or tens of thousands of opinions, cluster them, summarize them, and visualize them without mobilizing expert teams for months on end.

In other words, broad listening is a new method that has only just become technically feasible. Chapter 2 explains in detail how LLMs made large-scale analysis of open-ended questions possible. Chapter 12 covers the underlying technologies, and Chapter 13 covers implementation methods.

## Structure of This Book

This book explains broad listening in the following structure.

**Part I: Concepts (Chapters 1–3)**

- Chapter 1: What Is Broad Listening? (this chapter)
- Chapter 2: Broad Listening and Surveys: From Quantitative Analysis to Qualitative Analysis
- Chapter 3: Digital Democracy and Broad Listening: New Ways of Delivering the Public Will

**Part II: Case Studies (Chapters 4–11)**

- Chapter 4: The Spread of Broad Listening in Japan
- Chapter 5: Tokyo, Shin Tokyo 2050, and Policy Change Through Broad Listening
- Chapter 6: Use Cases in the 2025 House of Councillors Election
- Chapter 7: Use in Local Elections
- Chapter 8: Use in Local Governments
- Chapter 9: Use in Companies and NPOs
- Chapter 10: Broad Listening Overseas
- Chapter 11: The Ecosystem Supporting Broad Listening

**Part III: Technology (Chapters 12–13)**

- Chapter 12: Core Technologies Behind Public Listening AI
- Chapter 13: Implementing Public Listening AI

Depending on the reader’s interests, we recommend the following reading paths:

- **If you want to understand the concepts**: read Chapters 1–3, then move to the case studies as needed
- **If you want to know the use cases**: read Chapters 4–11, then move to the technology section if technical questions arise
- **If you want to know what kind of tool Public Listening AI is and why it was created**: start with Chapter 11, then move to the technology section as your interest dictates
- **If you want to understand the AI and data science behind it**: focus on Chapter 12
- **If you only want to grasp the flow of Public Listening AI’s processing pipeline**: start with Chapter 13 and refer to Chapter 12 as needed

### Scope of This Book

There are many approaches to broad listening. This book focuses primarily on Talk to the City (TTTC) and “Public Listening AI,” which developed from it.

We also refer as needed to other tools such as Polis, Idobata, and Jigsaw Sensemaker, but the main focus of this book is TTTC and Public Listening AI. By understanding these tools deeply, readers will be able to acquire ways of thinking that apply to broad listening more generally.

## Looking Ahead to the Next Chapters

In this chapter, we introduced the three forms of broad listening: broad listening as a tool, broad listening as support for augmented deliberation, and politician- and government-led broad listening. Chapters 2 and 3 go deeper into these aspects.

**Chapter 2: Broad Listening and Surveys: From Quantitative Analysis to Qualitative Analysis**

This chapter answers the question: “Can opinions gathered through broad listening be used like opinion polls?” It clarifies the characteristics and limits of AI-based opinion structuring and visualization from the perspectives of open-ended versus closed-ended questions, qualitative versus quantitative analysis, and the presence or absence of representativeness. It also explains how to combine qualitative and quantitative analysis appropriately in order to hear diverse voices while maintaining representativeness.

**Chapter 3: Digital Democracy and Broad Listening: New Ways of Delivering the Public Will**

This chapter answers the questions: “Why is this technology needed now?” and “How should it be operated?” It explains how to think about augmented deliberation that incorporates broad listening tools, including the limits of the election cycle, gaps in existing methods, and the design of operational workflows for real-world use.
