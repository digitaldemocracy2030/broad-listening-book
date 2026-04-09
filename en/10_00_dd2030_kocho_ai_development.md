# Development Work on Public Listening AI by DD2030

Written by tokoroten, nishio, and sumino

English translation by Luke Closs

## About Digital Democracy 2030

Digital Democracy 2030 (DD2030) is a nonprofit open-source project jointly launched in January 2025 by AI engineer Takahiro Anno and Ken Suzuki, Chairman of the Board at SmartNews and Project Researcher at the University of Tokyo. Ken Suzuki is known as the author of *Namerakana Shakai to Sono Teki* (*The Smooth Society and Its Enemies*), and has long explored the possibilities of digital democracy from the perspectives of complex systems science and natural philosophy. He has been deeply involved in shaping DD2030’s intellectual foundation.

In the 2024 Tokyo governor election, the volunteer group “Team Anno,” which supported Anno’s campaign, advanced efforts in broad listening. After that, practical applications gradually expanded, including opinion analysis for Nippon TV’s special lower house election program and use in formulating Tokyo’s long-term strategy. To ensure that these activities would not remain one-off projects, but instead be delivered continuously to society in a form that municipalities, political parties, and politicians could use neutrally, Digital Democracy 2030 was organized on the foundation of Team Anno.

What DD2030 has consistently emphasized is political neutrality. Rather than backing any particular politician or political party, it takes the position of building the infrastructure that underpins democracy, and in that capacity supports multiple political parties and municipalities regardless of whether they are in government or opposition. Under the goal of “updating democracy through information technology by 2030 and making it normal for every individual’s voice to reach politics and government,” DD2030 announced three development projects at its founding: Public Listening AI, the deliberation platform “Idobata,” and the political funding transparency tool “Polimoney.” Through these, it has been advancing work that supports digital democracy.

Later, in May 2025, Anno founded Team Mirai (meaning “Team Future”) and ran in the 2025 House of Councillors election. To preserve DD2030’s political neutrality, he stepped down from its board. However, development activities as a community have continued, and even after Anno’s departure, work to improve and promote each project, including Public Listening AI, has carried on.

This book, incidentally, has been written by volunteers as part of DD2030’s activities.

## What Is Public Listening AI?

Public Listening AI is an open-source AI analysis tool designed to broadly collect and visualize the voices of diverse citizens and put them to use in decision-making. It was developed within the Digital Democracy 2030 project based on Talk to the City (TTTC).

Simply by uploading a CSV file, the AI clusters and summarizes large volumes of opinion data and visualizes them in a way that makes the overall picture understandable. This makes it possible to analyze opinions efficiently without requiring enormous amounts of manual work, and to generate useful insights. It is intended for use in a wide range of settings, including analysis of public comments, policy development by political parties, and regional planning.

### From Talk to the City to Public Listening AI
![Technical lineage of Public Listening AI](images/10_kouchouai_technical_genealogy.png)

The origins of Public Listening AI trace back to “Talk to the City (TTTC),” an open-source tool for collecting and visualizing large volumes of opinions developed by the U.S.-based AI Objective Institute. TTTC has two variations: “TTTC Scatter,” which classifies and visualizes opinions using contextual vectors and scatter plots, and “TTTC Turbo,” which uses direct classification by an LLM to present results in a list-view format. Both have been used in practice in places such as Taiwan and elsewhere around the world.

Through the Tokyo governor election and Nippon TV’s election special, Team Anno repeatedly put broad listening into practice using TTTC Scatter. In the process, however, several challenges with TTTC Scatter became clear.

One issue was the high barrier to use. Because TTTC Scatter is a command-line tool, it required software engineering skills such as setting up a Python environment.  
Since TTTC Scatter uses an LLM (large language model) internally for analysis, users also need to write the instructions given to the LLM during analysis—that is, prompts. In principle, writing prompts should be possible without software development knowledge, but as noted above, because the tool was command-line based, analysts were effectively required to have software engineering skills.

There were also issues with the clustering algorithm. TTTC Scatter performs clustering using a single fixed number of clusters. If the cluster granularity is too coarse, differences between opinions are obscured; if it is too fine, the overall picture becomes hard to see. As a result, it was often necessary to rerun the analysis many times in search of an appropriate level of granularity.

To address these issues and needs—and to enable more users to conduct broad listening and gain more insight—development of Public Listening AI began as a fork of TTTC Scatter.

As an aside, “fork” here is the same word as the utensil “fork.” It carries the meaning of “something branched off,” and in software development it refers to creating a new project by inheriting code from an existing one.

At first, Public Listening AI was developed mainly by developers from Anno’s team, but it was released as open-source software (OSS) on March 16, 2025, after which many developers from the DD2030 community joined the effort.

### The Design of Public Listening AI: Implemented as a Web Application

The biggest feature of Public Listening AI is that it transformed TTTC Scatter from a command-line tool into a web application that can be operated through a browser. Hosting the application still requires engineering skills, but once it is up and running, analyses can be performed without development knowledge, and anyone can access the results simply by sharing a URL.

The main differences between Public Listening AI and TTTC Scatter are as follows.

| Perspective | TTTC Scatter | Public Listening AI |
|------|------|--------|
| Delivery format | CLI tool | Web app |
| Environment setup | Required on the user side | Completed on the server side |
| Running analysis | Command line | Operated from a browser |
| Sharing results | Host HTML | Just share a URL |

Public Listening AI also introduced several improvements to the analysis algorithm.

**Hierarchical clustering**

TTTC Scatter performed clustering at a single level, but Public Listening AI performs clustering hierarchically. This makes it possible to understand the structure of opinions at multiple levels of granularity. In addition to the conventional scatter-plot visualization, a treemap was also added to visualize the hierarchical structure.

**Extraction of dense clusters**

A feature was also implemented to extract only high-density clusters. Low-density clusters may contain a mix of weakly related data. Excluding them makes it possible to obtain clearer analysis results.

**Changes to the clustering algorithm**

TTTC Scatter and Public Listening AI use different clustering algorithms.

TTTC Scatter uses spectral clustering. Spectral clustering is a method that builds a neighborhood graph over the data points and performs clustering based on that graph structure. In this algorithm, points connected in that graph are assigned to the same cluster, so points that are far apart in the two-dimensional plot may still be included in the same cluster. As a result, clusters can appear scattered like detached islands on the plot, which can make interpretation difficult for users.

Public Listening AI, by contrast, uses K-means. K-means is a method that clusters points based on Euclidean distance, so spatially close points are grouped into the same cluster. This makes clusters appear more compact on the scatter plot and easier to interpret visually.

That said, K-means also has drawbacks. Because K-means assumes spherical clusters, it may fail to properly capture groups that are semantically related but have complex shapes in embedding space. Clusters that spectral clustering could have detected may end up split apart by K-means. Public Listening AI prioritizes visual clarity and accepts that trade-off by design.

### The Significance of Public Listening AI

With Public Listening AI, broad listening can now be carried out without programming skills or specialized knowledge. You can run an analysis simply by uploading a CSV, and publish the results simply by sharing a URL. Although environment setup still requires engineering skills, the barriers to running and sharing analyses have been greatly reduced.

Its open-source nature is also well suited to government use, where transparency in the analysis process is essential. Because the code makes it possible to verify at the code level how opinions are classified and summarized, it becomes easier to meet accountability requirements for the analysis results. The detailed implementation is explained in Chapters 12 and 13, but since this is an open-source product, we encourage you to read through the source code with AI by your side.

## The Turning Point of May 2025

May 2025 became a turning point for the development of Public Listening AI—a moment when “deceleration and acceleration happened at the same time.”

### The Community’s Dispersion

On May 7, 2025, Anno, who had launched DD2030, stepped down from DD2030’s board. The next day, May 8, he formally founded Team Mirai and began full-scale political activity for the 2025 House of Councillors election. He made the decision to step down in order to preserve DD2030’s political neutrality, but it had a major impact on the organization’s operating structure.

Because DD2030 had been organized on the foundation of Team Anno, when Anno launched Team Mirai, many of the engineers and designers who had originally been active in Team Anno naturally shifted their activities to the Team Mirai community. In addition, one of Public Listening AI’s core developers decided to run as a candidate from Team Mirai, and development resources declined sharply.

Some people moved to Team Mirai, some remained with DD2030, and some moved back and forth between both communities. Public Listening AI had been released as an open-source project, so this turnover among developers can even be seen as part of the nature of the community. Precisely because the project had a structure in which anyone could contribute code without depending on any particular individual, development did not stop completely and was able to continue.

### The Birth of the Team Mirai Version

To meet its own requirements for the 2025 House of Councillors election, Team Mirai forked the Public Listening AI repository and proceeded with independent development. By forking it, Team Mirai’s engineers could add and release features based solely on their own judgment, without waiting for approval from the DD2030 side. In the time-constrained environment of an election, this agility was a major advantage.

Team Mirai used “Idobata Policy,” a version of the deliberation platform “Idobata” developed by DD2030 and specialized for policy proposals, to broadly solicit citizens’ suggestions for changes to its party platform. The large volume of proposals collected was managed as GitHub PRs, and when visualizing them in Public Listening AI’s scatter plot, Team Mirai implemented a “source link feature” that allowed users to jump from each data point directly to the original GitHub PR with a single click. This was an important improvement from the standpoint of transparency, because it allowed citizens themselves to verify which opinions had been classified into which clusters.

The features first developed in this fork were later merged back into DD2030’s mainline. Freely developing the features they needed, and then contributing them back to the mainline if they proved broadly useful—the relationship between Team Mirai and DD2030 embodied the healthy cycle of branching and integration envisioned by open source.

### The Rise of AI Programming Tools

In developing Public Listening AI, DD2030 had been using the AI coding agent “Devin” since the organization’s founding. After Anno published an introductory video about Devin on YouTube, he received a large amount of credit (points used to run AI) as a referral fee, and the team used those credits collectively to make use of Devin. Even after Anno stepped down, DD2030 entered into a new contract and has continued using Devin in the development of its various products.

Out of a total of 2,756 commits over the full project period (February 2025 to February 2026), 210 commits (about 7.6%) were made by Devin. This ranked fifth among 24 contributors including humans, giving it a presence on par with core developers such as Sumino (389 commits) and Nishio (354 commits).

In addition, GitHub Copilot Agent and Claude Code became generally available in May 2025, and AI-assisted development styles spread further. However, this did not immediately mean that the overall pace of development recovered. In reality, while the difficulties of the slowdown period continued, advances in AI programming tools functioned as support that helped a small number of people keep development from grinding to a halt.

A concrete example is Public Listening AI’s “attribute filter feature.” This feature allows data to be filtered by attributes such as age, gender, and supported political party, and visualizes the distribution of opinions by attribute. It was a large-scale implementation involving 2,203 added lines, but most of the feature was developed by GitHub Copilot Agent.

Then, in September 2025, it became possible to assign a GitHub issue to AI and have code written automatically. Development has accelerated further as we entered an era in which AI programming can be completed without even opening an editor.

### Keeping Development Alive During the Slowdown

In OSS development, when work continues with only a small number of people, there are times when the back-and-forth of review and decision-making thins out and progress becomes difficult. In Public Listening AI as well, that phase surfaced after the community dispersed.

After the turning point in May 2025, development of Public Listening AI remained in a difficult phase for some time. The density of day-to-day communication declined, and the exchanges needed for PR review and decision-making increasingly stalled. As reviews became harder to get, changes were less likely to be merged and more likely to sit idle.

During this period, there were times when the number of people who could effectively continue development was extremely limited. As a result, even when PRs could be created, there was sometimes no one available to review them, so they could not be merged and would remain stalled for a while. In long cases, as much as two and a half months could pass before review and merge, making it easy for the development tempo to slow down.

But even as the community dispersed, a small number of members deliberately remained on the DD2030 side and tried to preserve continuity in both development and community.

Specifically, they made a point of continuing regular development meetings whenever possible, even when there was no clear agenda, and of preserving a place where participants could share the current situation or simply chat, even on days with few attendees—a place where people could know “what is happening right now.” It would have been possible to decide not to hold meetings when there was no progress to report, but in this phase it was necessary to maintain the entry point itself in order to keep the flame alive. In fact, some people came to understand the situation through this venue and, though few in number, newly joined the development effort.

### Outreach Through On-the-Ground Engagement

On the outreach side, DD2030 did not simply “wait” for municipalities it had made contact with, but proactively reached out to them. In addition to gathering feedback during trial runs, it emphasized hearing about issues that do not directly appear in the product itself—such as data preparation, internal coordination within government offices, accountability, and the workflow of public listening—in order to understand the context on the ground.

That said, adoption did not always proceed smoothly. Municipalities did not always have strong incentives to prioritize trying new initiatives, and there were limits to how much DD2030 alone could drive adoption.

For example, even when a mayor or governor was highly enthusiastic, implementation could still lose momentum because frontline workloads and organizational capacity could not keep up. In some cases, staff transfers changed the person in charge, requiring all the background assumptions to be shared again from scratch. In addition, on the user side considering adoption, it was often difficult to secure an ongoing engineering structure, making the transition from trial to operation prone to interruption.

For that reason, in the early stages of outreach, the team did not wait for “well-developed case studies” to emerge naturally. Instead, it consciously repeated small-scale trials wherever cooperation could be obtained, recorded what was learned, and preserved those lessons as case studies. At the same time, because of constraints on staffing and time, there were also situations where it was not possible to sustain the full cycle of designing, documenting, and sharing trials, and fragmented lessons sometimes disappeared within one-off exchanges.

One practical insight repeatedly confirmed through these field connections was that in broad listening, if you do not first articulate what kind of insight you want to obtain, collecting opinions itself easily becomes the goal. If the question remains vague while the number of responses simply increases, the result is that the analysis gets buried in a mass of opinions, making it difficult to extract insights that connect to decision-making.

Another recurring issue was that in many municipalities, the challenge lay not in analysis but in collecting enough opinions in the first place. In some municipalities, there were situations where participation in calls for public input did not spread easily because there was not enough accumulated day-to-day dialogue between government and residents.  
This appears to be a structural issue overlapping with the reasons public comment processes so easily become hollow formalities. Simply creating a “place to collect opinions” is not enough; what matters is designing for motivation to participate and for the ongoing building of trust.

Even so, through dialogue with these municipalities, the team accumulated a sense of what tends to become a barrier in practice and which preconditions matter most. This work was intended not only to promote adoption itself, but also to incorporate user-side feedback into the direction of development. By continuously gathering voices from the field and feeding them back to the development members as shared assumptions, the team worked to ensure that development outcomes did not drift away from users—whether political parties, municipalities, or private-sector organizations considering adoption. At the same time, it aimed to work together with them to create conditions in which the tool could continue to be used in ways suited to local realities.

Also as part of outreach activities, technical presentation slides on Public Listening AI[^techslide] were released in June 2025. These slides helped broaden awareness of Public Listening AI through university lectures and corporate talks, and they became the basis for Chapters 12 and 13 of this book.

[^techslide]: “Technical Explanation of Public Listening AI: The Technology Supporting Broad Listening,” June 2025. https://www.docswell.com/s/tokoroten/ZL1M88-2025-06-14-014546

## The Future of Public Listening AI

Public Listening AI began as a fork of TTTC Scatter and, despite community dispersion and resource depletion, grew into a product with 807 merged PRs in about one year from its first commit in February 2025. Several of the case studies introduced in this book used Public Listening AI, which is evidence that it functioned as a tool practical enough for real-world use.

Public Listening AI is published as open source, and anyone can contribute to its codebase. Just as Team Mirai forked it to develop election-oriented features and those features were later contributed back to the mainline, people who need improvements can make them and share the results with the community. It is precisely because this structure exists that development has been able to continue without depending on any particular individual or organization.

The challenges ahead are clear. As outreach activities have shown, simply providing the tool is not enough for it to take root in practice. It will be necessary to accumulate operational knowledge in areas such as how to design opinion collection, how to share analysis results within government offices, and how to connect them to decision-making, thereby lowering the barriers to adoption. There is also a need for mechanisms that reduce the burden of hosting and maintenance so that municipalities and political parties can operate it sustainably on their own.

Public Listening AI is not a finished product. But there is meaning in its continuing to exist as a foundation that anyone can use and anyone can improve. Even if DD2030 were to cease operations, the code would remain, and someone who needs it could fork it and carry it forward. That is the strength of open source, and it is also why Public Listening AI has been developed in this form.
