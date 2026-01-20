# Introduction in English

We are writing a book that documents broad listening practices in Japan.

"Broad Listening That Changed Elections -- Visualizing and Analyzing Public Voice with Generative AI"(working title) records how Broad Listening, as the opposite of Broadcast, was implemented in political and administrative settings, what worked, and where unintended side effects appeared. It traces a new democratic protocol that only became feasible with LLMs: structuring large volumes of voices by context and returning them to decision making. The book is grounded in Japan's social experiments in 2024-2025.

The preface is written by AI engineer and politician Takahiro Anno, who frames broad listening as "the operating system for future democracy."

This book is written as part of Digital Democracy 2030. Drafts are published under CC BY 4.0 (the Japanese edition is scheduled for publication by Impress in summer 2026). It is a practical book for policymakers, civic technologists, researchers, media, and engineers, with an emphasis on the back-and-forth between theory and field practice.

# Table of Contents Overview

- Preface: Takahiro Anno (AI engineer and politician). Against the backdrop of Plurality and LLMs, he surveys why "listening broadly" is essential for democracy and reflects on practice from the Tokyo governor election to national politics.
- Part 1 (Concepts): Organizes broad listening into two layers -- a component that summarizes large volumes of information and a process of augmented deliberation. It clarifies differences from surveys, lays out design principles, and provides operational know-how as a "map" for practice.
- Part 2 (Japan case studies): Deep dives into broad listening projects during the Tokyo governor election (including Talk to the City), the Tokyo Metropolitan Government's long-term strategy work, a Polis-based public opinion visualization (YoronChizu) during the 2025 House of Councillors election, and the "talkable manifesto." It covers concrete cases across elections, administration, and media, including use in the National Diet and trials by newspapers and TV broadcasters.
- Development and ecosystem: Covers the development of Koucho AI by Digital Democracy 2030, its open-sourcing, and on-the-ground challenges and improvements. It also includes perspectives from support organizations that help local governments adopt broad listening, such as Boots and Code for Japan.
- Local and corporate: Introduces local elections (e.g., the Yatsushiro City case in Kumamoto) and multiple corporate applications, including in-house workshops and call centers for analyzing customer voice.
- Part 3 + columns: Explains component technologies and the implementation stack in a two-layer structure. Columns discuss adjacent themes such as "freedom to choose AI" and "collective vs connective action."

---

# Preface: The Present and Future of Broad Listening

Commentary: Takahiro Anno (Translated by NISHIO Hirokazu using GPT5.2)

## What Is Broad Listening

What is "Broad Listening (Broadlistening)"? This unfamiliar term is positioned as the antonym of the familiar "Broadcast."

Broadcast is a technology and culture where a single sender delivers information to many receivers at once. Our democracy, nurtured through newspapers, television, radio, and social media, has been designed around media that "deliver broadly."

In contrast, broad listening aims to place "listening broadly" at the center of democracy again. Instead of leaving many voices scattered, it receives them as a single system, transforms them into a structure that can be understood, and ultimately returns them to society in the form of policy and institutional change. It is an attempt to redesign the entire loop, both technologically and institutionally.

The technologies and practices described in this book are still in their infancy. They are far from perfect, and there are only a handful of successes. It would not be wrong to say there are more things that did not work and more unintended side effects. Even so, we are convinced that broad listening is a critical technology for the "operating system of future democracy."

Behind this is a major tectonic shift in technology: the idea of Plurality advocated by Audrey Tang and others in Taiwan, which seeks coexistence and consensus among multiple positions rather than a simple majority vote, and the emergence of large language models (LLMs) that can articulate tacit knowledge and structure massive amounts of text.

Against that backdrop, the author has been experimenting with implementing broad listening in the arenas of elections, administration, and the National Diet in Japan. Originally an AI engineer who founded an AI startup, the author ran in the 2024 Tokyo governor election and proposed bringing technology into elections. In the same year, he became an advisor to GovTech Tokyo, a digital-affiliated organization of the Tokyo Metropolitan Government, and explored how such technology could be used for local government policy making. In January 2025, he co-founded a civic tech organization called Digital Democracy 2030 with Ken Suzuki. In May 2025, he founded a political party called Team Mirai, won a seat in the July House of Councillors election, and now, as the leader of a national party, is trying to update democracy digitally from within Nagatacho.

## Trajectory of Trial and Error

To implement broad listening, the author would like to introduce four initiatives he has pursued: (1) "Manifesto as Code" and AI-based listening in the Tokyo governor election, (2) "Shin Tokyo 2050," which applied the know-how to administration, (3) the "Idobata Policy" in the House of Councillors election, and (4) the use of an AI interviewer after becoming a member of the National Diet.

### 1. Tokyo Governor Election: Treating the Manifesto as Code

The first and most symbolic practice was the challenge of the 2024 Tokyo governor election. Here we built a system that treated the candidate's manifesto like software development, updating it during the campaign based on citizen feedback.

Three components were central.

First, the manifesto was published on GitHub and managed with Issues and Pull Requests (PRs). Comments from voters and stakeholders were registered as Issues, and the policy team reviewed them and reflected actual wording changes as PRs. During the campaign alone, hundreds of issues and change proposals were submitted, and many were adopted as manifesto revisions. The manifesto was treated not as a static PDF but as a living document evolving through versions 1.0, 1.1, 1.2, and so on.

Second was the AI avatar "AI Anno." It was an AI town meeting that answered questions from voters 24/7 through YouTube streams, with nearly 10,000 questions in total. These conversations used RAG (retrieval-augmented generation) to reference the manifesto hosted on GitHub, and the logs were accumulated as material for policy revisions.

Third was broad listening using Talk to the City (TTTC). Reactions scattered across online spaces such as X (formerly Twitter) posts and YouTube comments were collected and clustered with TTTC to visualize what topics were discussed and how much. It also served as a filter to extract constructive points from a noisy online space and organize them as GitHub Issues.

The resulting policy manifesto earned the best evaluation among candidates in a university research assessment, demonstrating tangible results, but looking back, each initiative was still rough around the edges and many challenges remained.

### 2. Working with Government: Broad Listening in "Shin Tokyo 2050"

The experiment in the governor election later led to implementation within government. In the Tokyo Metropolitan Government's long-term strategy project "Shin Tokyo 2050," we collaborated with the in-house team at GovTech Tokyo to apply broad listening to real policy design.

Channels for gathering opinions expanded beyond online forms, X, and YouTube comments to include postal mail, email, and in-person interviews at Ueno Zoo and science museums. In total, over 10,000 voices were collected and integrated and analyzed through a TTTC-based system.

GovTech Tokyo built the infrastructure for data collection, integration, and visualization, while we advised on prompt design, parameter tuning, and how to read analysis results. During the project, new functional demands emerged, such as deeper dives and reorganization of clusters, which later led to the development of "Koucho AI."

### 3. House of Councillors Election and the "Talkable Manifesto"

The next phase was the "talkable manifesto" for the House of Councillors election. In the governor election, we accepted policy improvement proposals directly through GitHub, but this was simply too high a barrier for non-engineers.

To solve this, we developed "Idobata Policy," combining AI with MCP (Model Context Protocol). Users no longer needed to think about GitHub at all. They could simply express their opinions in a chat interface, and behind the scenes the AI organized the content and automatically generated a Pull Request on a GitHub repository.

This enabled policy revision proposals not only from people who write code but from a much broader group. Ultimately, more than 8,000 proposals were submitted, an order of magnitude larger than in the governor election.

But new challenges emerged. The quality of PRs varied widely, and the burden of reviewing them and selecting what to adopt became extremely heavy. The feedback loop to original proposers, explaining acceptance or rejection, was also insufficient. We succeeded in widening the entry, but the design of "organize, use, and return" remained an unresolved task.

### 4. Practice as a Member of the National Diet: The AI Interviewer

Now, as a member of the National Diet, the author is trying to connect broad listening more directly to the protocol of politics.

On the surface, committee questioning in the Diet seems to allow ample preparation time, but in practice the time sense is very different. Once schedules and topics are set and materials arrive, the lead time to submit questions can be as short as two or three days. There are also many situations where one must ask questions about areas outside one's expertise.

In this context, the AI interviewer was introduced. It is a mechanism where AI acts as the interviewer in semi-structured interviews, collecting deep voices from practitioners and stakeholders in a short time. When participation links are distributed through SNS or email lists, within a few hours dozens of hours of dialogue are gathered. LLMs organize and summarize the logs and provide reports that directly inform the structure of Diet questioning.

For example, in AI interviews related to legal revisions for the digitization of bills of lading (B/L), we gained cooperation from many practitioners. Specific concerns from the field and bottlenecks like "this will clog here if we keep going" surfaced. In the theme of Diet DX, vivid episodes about the pain of paper and fax that Diet staff and parliamentary offices face daily were gathered. This yielded tactile information for thinking about where and how to probe in the Diet, rather than just numbers for or against. Still, the number of cases is not yet large, and challenges remain.

## Future Tasks: How to Define the "Quality" of Broad Listening

As we have seen, tools and mechanisms for listening broadly have been explored in many ways, and improvements are gradually accumulating. The key to continuing improvement is how to define the "quality" of broad listening.

We are currently trying to redefine that quality along five dimensions: (1) breadth, (2) depth, (3) speed, (4) the quality of insights that lead to action, and (5) completion rate of action.

### A) Breadth: Quantitative Expansion and Its Limits

The most straightforward indicator is how many people participate. The number of participants and the number of opinions and PRs submitted are easy to count as KPIs. In fact, the number of proposals gathered through our systems increased by orders of magnitude from the governor election to the House of Councillors election, a huge step forward in the volume of voices.

However, in politics it is dangerous to treat quantity as a direct representation of public opinion. Intentional majority manipulation and incentive asymmetry always exist. A minority that strongly opposes a policy tends to send opinions repeatedly with much higher energy than a majority that is indifferent. Estimating the overall structure of approval and disapproval based on the share among respondents requires great caution.

Moreover, it is often difficult to increase the number of respondents in the first place. Those who will take the time to fill in a form about political issues are only a tiny fraction of the whole. Even if a place to speak is provided, speaking there still imposes a cost on citizens. The author believes that building trust that the voices we raise might change society is one key.

Taiwan's civic participation platform "JOIN" is a good example. It is a government-sanctioned web service where citizens can post proposals such as "it would be nice to have this policy or law." Anyone can write, but there is an important rule: proposals that collect 5,000 or more endorsements must be reviewed and answered by the responsible ministry. Good proposals are implemented, and when they are not adopted the reasons are explained.

Over the past ten years, about 10,000 citizen proposals have been submitted, around 370 have passed the 5,000 endorsement threshold, and about 200 policies or laws have actually been implemented. Because the recognition has spread that using JOIN leads to tangible outcomes, citizens feel it is worth paying the cost to speak.

Conversely, if you only provide a place to speak and it does not lead to action, citizens are left with a negative experience of "I worked hard to answer a survey, but it was pointless." Trust leads to more participants, and lack of trust leads to rapid decline. Whether you enter a good cycle or a bad cycle is decisive.

At this point, Team Mirai chose the path of becoming a national political party. It ensures that action can be taken within the Diet. The existence of this exit provides credibility and helps secure a certain level of participation in broad listening.

Of course, many mechanisms are needed. Civic tech organizations, companies, politicians, and government must each steadily build trust that "if I speak here, something might change."

Breadth is a difficult indicator: it should be neither overvalued nor undervalued. If policymakers refer to it without recognizing bias, there is a risk of producing undesirable policies that do not align with reality.

### B) Depth: From Surveys to Dialogue

The second dimension is how deeply we can listen. Conventional surveys and public comments often end with one question and one answer. They reveal only surface attitudes such as "support," "oppose," or "not sure," and do not reach deeper layers such as why people think so, what experiences their opinions are rooted in, or whether they would keep the same view after hearing counterarguments.

We focus on the approach of the AI interviewer. AI acts as interviewer and advances the discussion interactively. It does not end with a single question; it asks "Why is that?" "Please tell me a concrete episode," "Here is a counterargument; what do you think?" It can also provide explanations of rules or data as needed and can ask again after aligning factual understanding, asking "Even so, what do you think?"

Such semi-structured interviews or debate-like communication used to require human participation. However, current LLMs are gaining the ability to do this. Similar AI interview functions are beginning to be released by frontier model companies overseas such as Anthropic by the end of 2025. As a method to extract tacit knowledge sleeping in the minds of experts and stakeholders at scale and speed, AI interviews are likely to develop in politics as well as business.

Analyzing dialogue logs after interviews reveals insights directly connected to policy design: "This concern arises from a misunderstanding of the premise," or "If this condition is met, many people can accept it." The depth of information that elicits assumptions, reasons, and conditions together, rather than just for or against, is essential to broad listening.

### C) Speed: Shortening the Lead Time of Democracy

The third dimension is speed, or how quickly we can collect and organize these voices. Social issues often surface suddenly. Accidents, disasters, new technologies, and changes in international affairs all require the Diet and local governments to devise responses within limited time.

As noted earlier, the time available for preparing Diet questions can be as short as two or three days. With traditional processes that rely on manual interviews, memo writing, and issue organization, it is often impossible to keep up.

Combining broad listening with AI interviews can drastically shorten this lead time. In the author's recent experience, spreading an interview link on SNS prompted hundreds of stakeholders to start responding within an hour, and the total dialogue time quickly reached dozens of hours. This level of speed was difficult with previous methods.

To increase speed further, we need to reach stakeholders and experts quickly and help them respond quickly. For political parties and civic communities, this means preparing lists of stakeholders and experts in advance, maintaining engagement and attention from them, and building trust so their networks will help spread participation.

### D) Extracting Insights That Lead to Action

The fourth dimension is whether we can extract insights that lead to action. Here, it is more important to find "one line that is valuable even if n=1" than simple majority ratios such as "X percent support, Y percent oppose." The latter can be manipulated intentionally or heavily biased. Quantitative information can be useful in some cases, but the more you recruit broadly on the internet, the harder it becomes to rely on such numbers.

So what kind of information leads to action? One idea is to find information that represents a difference from the baseline. Policy makers usually have a current hypothesis or a draft plan. Extracting information that can affect that hypothesis or draft has real value. For example, a policy designed with good intentions can be met with unexpected concerns from the field. When such information appears, it must be verified and the hypothesis may need revision.

Another factor that cannot be ignored is the power of emotional episodes. A high-resolution episode from a single stakeholder can help grasp reality beyond statistics. Ground-level, living episodes that are not available in the market can carry special persuasive power.

Of course, anger based on factual misunderstanding or conspiracy-like narratives also carries strong emotion, so governance is needed to verify facts and decide how far such information should be used in policy judgment.

Even so, being able to handle both logical revision proposals and emotional episodes is an important destination for broad listening. Politicians and administrators move when rational logic and a story that moves the heart are both present.

### E) Completion of Action: Are We Actually Creating Change?

The last and most important dimension is the completion of action. No matter how widely, deeply, and quickly we gather voices and organize insights that lead to action, the broad listening loop is not complete unless action happens in the real world.

In this sense, Team Mirai becoming a national political party with the authority of Diet members has meaning. Members of the National Diet are granted by law the following options and powers:

- Vote for or against laws, treaties, and budgets
- Question and speak in Diet committees
- Submit official questions to the government in the form of written questions
- Submit citizens' voices to the Diet as petitions or requests
- Submit amendments to existing bills
- Submit new bills themselves

We have a hypothesis that, at this early stage of broad listening, a structure that integrates four actors -- political parties, tech platforms, policy designers, and media -- will maximize its value. If the team that collects voices, the team that analyzes and interprets them into bills or institutions, and the team that pushes those bills forward in politics exist as separate organizations, there is a high chance that information will be lost at some connection point.

Team Mirai is in a unique position globally in that it can vertically integrate "listen," "think," "decide," and "act" within a single organization. There are very few examples worldwide of a national political party that has a team of software engineers working in this way. For that reason, we believe Team Mirai has a responsibility to conduct many experiments.

At the same time, the roles of government and the civic tech community are also very large. Because civic tech communities do not hold political positions, there are many things they can do freely, and there are also many things that only local governments or central ministries can achieve. It would be ideal if what each player explores could be shared and connected organically for the development of broad listening.

## Closing: Cultivating a New System Together

As we have seen, broad listening is still an immature technology and practice. Talk to the City, AI Anno, the GitHub manifesto, Idobata Policy, the AI interviewer, Koucho AI -- none of the projects mentioned here is something we can confidently say is complete.

Even so, by following the trial and error traced in this book, readers will be able to sense the possibilities of these technologies and concepts. Their development requires not only engineers but also the cooperation of each citizen as a sovereign. I hope this book becomes a trigger for readers to think about the future of democracy.
