# Preface: The Present State and Future of Broad Listening

Commentary by Takahiro Anno

English translation by Luke Closs

*Translation note (English edition): Original Japanese authorship is preserved throughout this book. Any errors in the English translation are the translator’s own.*

TODO: Incorporate the latest version as appropriate: https://docs.google.com/document/d/13P7PDVu1IlM4EEr43dnW8KUGDi5HiZuamTut3XbOBP4/edit?tab=t.h9b0y89f3vv3


## What Is Broad Listening?

What is “Broad Listening”? This unfamiliar term can be understood as the opposite of the “broadcast” we all know so well.

Broadcast is a technology and culture in which a single sender delivers information to a large number of recipients all at once. Our democracy, shaped through newspapers, television, radio, and social media, has fundamentally been designed around this model of “reaching broadly.”

Broad Listening, by contrast, seeks to place “listening broadly” at the center of democracy. Rather than leaving the voices of many people scattered and disconnected, it aims to receive them as part of a single system, transform them into a structure that can be understood, and ultimately return them to society in the form of policy or institutional change. It is an attempt to redesign that entire loop from both the technological and institutional sides.

That said, the technologies and practices discussed in this book are still in their infancy. They are far from perfect, and there are only a handful of success stories. In fact, it would be fair to say that there have been more failures and unexpected side effects than clear wins. Even so, we are convinced that Broad Listening will be a critically important technology for the “operating system of democracy” in the years ahead.

Behind this conviction lies a major technological and intellectual shift. One element is the idea of “Plurality,” advocated by Audrey Tang and others in Taiwan: not a democracy based merely on majority rule, but one in which multiple perspectives coexist while seeking agreement. Another is the emergence of large language models (LLMs), which can articulate tacit knowledge and structure vast amounts of text.

Against this backdrop of new ideas and technologies, the author has been experimenting with how to implement Broad Listening in the real-world settings of Japanese elections, government administration, and the Japanese Parliament. Originally an AI engineer who founded an AI startup, the author ran in the 2024 Tokyo governor election and proposed bringing technology into the electoral process. That same year, he became an advisor to GovTech Tokyo, a Tokyo metropolitan digital-policy affiliate, and explored whether these technologies could be used in local government policymaking. In January 2025, he co-founded the civic tech organization *Digital Democracy 2030* with Ken Suzuki. In May 2025, he launched Team Mirai (referred to as “Team Future” in this book), a political party, which won seats in the July House of Councillors election. He is now working from within Nagatacho, Japan’s political center, as the leader of a national political party, to “update democracy through digital technology.”

## A Trail of Trial and Error

To illustrate the effort to implement Broad Listening, I would like to introduce four initiatives the author has worked on: (1) “Platform as Code” and AI-powered public listening in the Tokyo governor election, (2) “Shin Tokyo 2050” (“New Tokyo 2050”), which brought that know-how into government administration, (3) “Idobata Policy” in the House of Councillors election, and (4) the use of an AI interviewer after becoming a member of the Japanese Parliament.

### 1. Tokyo Governor Election: Treating the Platform as “Code”

The first, and most symbolic, experiment was the 2024 Tokyo governor election. There, we built a system in which the candidate’s electoral platform itself could be updated during the campaign in response to citizens’ voices, using a process similar to software development.

At the core were three components.

The first was a mechanism for publishing the platform on GitHub and managing it through Issues and Pull Requests (PRs). Feedback and suggestions from voters and stakeholders were logged as Issues, and the policy team reviewed them and reflected actual wording changes through PRs. During the campaign alone, hundreds of issues and proposed changes were submitted, and many were adopted as revisions to the platform. Campaign pledges were treated not as a static PDF, but as a “living document” that evolved through versions 1.0, 1.1, 1.2, and so on.

The second was the AI avatar “AI Anno.” Through YouTube streams and other channels, it functioned as an AI town hall that answered voters’ questions 24 hours a day, 365 days a year, ultimately handling nearly 10,000 questions. These conversations were conducted using a method known as RAG, referencing the platform hosted on GitHub, and the resulting logs were accumulated as material to inform policy revisions.

The third was Broad Listening using Talk to the City (TTTC). Reactions scattered across the internet—such as posts on X (formerly Twitter) and YouTube comments—were collected and clustered with TTTC to visualize “what issues were being discussed, and how much.” It also served as a filter, extracting constructive issues from the internet’s mix of abuse and noise and organizing them into GitHub Issues.

This approach produced some meaningful results—for example, the resulting policy platform received the highest evaluation among the candidates in an assessment by a university research institution. At the same time, looking back, every one of these measures was still rough around the edges, and many challenges remained.

### 2. Working with Government: Broad Listening in “Shin Tokyo 2050”

The experiment in the governor election later led to implementation within government itself. In the Tokyo Metropolitan Government’s long-term strategy project *Shin Tokyo 2050*, we worked with GovTech Tokyo’s in-house team to apply Broad Listening mechanisms to a real policy development setting.

Channels for gathering input expanded beyond online forms, X, and YouTube comments to include postal mail, email, and in-person interviews at places such as Ueno Zoo and science museums. In the end, more than 10,000 voices were collected and integrated and analyzed through a TTTC-based system.

GovTech Tokyo built the infrastructure for data collection, integration, and visualization in-house, while we advised on AI analysis prompt design, parameter tuning, and how to interpret the results. As the project progressed, new feature requests emerged—such as deeper drill-downs and reclustering—which later led to the development of “Public Listening AI.”

### 3. The House of Councillors Election and the “Talkable Platform”

The next phase was the “talkable platform” in the House of Councillors election. In the Tokyo governor election, we had introduced a system for accepting policy improvement proposals directly through GitHub, but this inevitably posed a high barrier for anyone who was not an engineer.

That led to the development of “Idobata Policy,” which combined AI with MCP (Model Context Protocol). With this system, users no longer needed to think about GitHub at all; they could simply express their views naturally through a chat interface. Behind the scenes, AI organized the content and automatically generated Pull Requests in the GitHub repository.

This made it possible to gather policy revision proposals not only from “people who write code,” but from a much broader public. In the end, more than 8,000 proposals were submitted—several times the scale of the governor election in sheer volume.

But here too, new challenges emerged. The quality of the large number of PRs varied widely, and the burden of “selection”—who would review them, how, and which proposals would be adopted—became extremely heavy. The loop for feeding back adoption or rejection decisions to the original contributors was also still inadequate. We succeeded in “widening the front door,” but the design of the “organize, use, and return” part remained the next major task.

### 4. Putting It into Practice as a Member of Parliament: The “AI Interviewer”

Now, as a member of the Japanese Parliament, the author is trying to connect Broad Listening more directly to the “protocols of politics.”

Committee questioning in the Japanese Parliament may appear, from the outside, to allow ample time for preparation. In reality, the sense of time on the ground is very different. Once the schedule and topic are set and materials arrive, the lead time before submitting formal questions can be as short as two or three days. And often, one must ask questions in fields outside one’s own expertise.

In that context, we introduced the “AI Interviewer.” This is a mechanism in which AI serves as the interviewer in semi-structured interviews, enabling the rapid collection of deep, substantive input from practitioners and affected stakeholders. When participation links are distributed through social media, mailing lists, and similar channels, dozens of hours’ worth of dialogue can be gathered within just a few hours. LLMs then organize and summarize the logs and provide them as reports that feed directly into the structure of parliamentary questioning.

For example, in an AI interview conducted in connection with legal reform on the digitization of bills of lading (B/L), we were able to gain the cooperation of many practitioners. It surfaced concrete concerns from people in the field, as well as bottlenecks—places where “this is where things will get stuck if nothing changes.” On the topic of digital transformation in the Japanese Parliament, many vivid episodes emerged about the daily “pain of paper and fax machines” faced by parliamentary staff and lawmakers’ offices. We have been able to obtain a certain amount of tangible, practical information—not just numbers showing support or opposition, but insight into “where and how to press the issue in Parliament.” That said, the number of cases is still limited, and many challenges remain.

## Future Challenges: How Should We Define the “Quality” of Broad Listening?

As we have seen, a variety of tools and mechanisms for “listening broadly” are being tested and gradually improved through trial and error. In that process, one key question for continued improvement is: how should we define the “quality” of Broad Listening?

At present, we are trying to redefine the “quality” of Broad Listening from five perspectives: (1) breadth, (2) depth, (3) speed, (4) the quality of insights that lead to action, and (5) the completion rate of action.

### A) Breadth: Quantitative Expansion and Its Limits

The most intuitive metric is simply: “How many people participated?” The number of participants, and the number of submitted opinions or PRs, are easy to count as KPIs. In fact, from the Tokyo governor election to the House of Councillors election, the number of proposals gathered through our systems increased by orders of magnitude. In terms of the sheer volume of voices, this was a dramatic advance.

But in politics, it is dangerous to treat “quantity” as a direct proxy for “the will of the people.” Deliberate attempts to manufacture majorities and asymmetries of incentive are always present. A small minority strongly opposed to a policy is often far more motivated than a large number of indifferent people, and may submit opinions repeatedly with much greater intensity. Any attempt to infer society-wide patterns of support and opposition from “the proportion among those who responded” requires great caution.

Moreover, in many cases, simply increasing the number of respondents is itself difficult. Those willing to go out of their way to fill out a form on a political issue are only a tiny fraction of the whole population. Even if a venue for speaking up is provided, speaking up still imposes a cost on citizens. The author believes that one key lies in fostering trust that “the voices people actually raise might help change society.”

Taiwan’s civic participation platform “JOIN” is a good example. It is an officially recognized government web service where citizens can post ideas such as “I wish there were a policy or law like this.” Anyone can submit an opinion, but there is one important rule: if a proposal gathers support from more than 5,000 people, the responsible ministry is guaranteed to review it and respond. Good proposals are actually reflected in policy, and even when they are not adopted, the reasons are clearly explained.

Over the past decade, roughly 10,000 citizen proposals have been submitted, of which about 370 surpassed the 5,000-support threshold. Of those, around 200 policies or laws have actually been put into effect. Because people recognize that “if you use JOIN, your input will be received and can lead to real results,” citizens feel that it is worth paying the cost to speak up.

Conversely, if you merely provide a venue but it does not lead to real action, citizens are left with the negative experience of “I worked hard to answer the survey, and it was all for nothing.” If trust is earned, participation will continue to grow; if trust is not earned, the system will quickly wither. Whether one enters a “good cycle” or falls into a “bad cycle” is decisively important.

In this context, Team Future chose the path of becoming a national political party. That ensures the ability to take action within the Japanese Parliament. The existence of this “exit” helps guarantee credibility and, in turn, helps secure a certain level of participation in Broad Listening.

Of course, this requires many different mechanisms. Civic tech groups, companies, politicians, and government institutions all need to steadily build trust, from their respective positions, that “if you voice your opinion here, something might actually change.”

“Breadth” is a difficult metric—one that must be neither overestimated nor underestimated. It is worth emphasizing again that if policymakers rely on it without recognizing its biases, there is a real danger of producing undesirable policies that do not align with reality.

### B) Depth: From Surveys to Dialogue

The second perspective is “depth”: how deeply are we actually hearing people’s voices? Traditional surveys and public comment processes often end with one question and one answer. They reveal only surface-level attitudes—“support,” “oppose,” or “neither”—without reaching deeper questions such as: Why do you think that? What experiences is your view rooted in? Would you still make the same argument after hearing the opposing side?

What we are focusing on here is the AI interviewer approach. AI acts as the interviewer and advances the discussion interactively. It does not stop at one question and one answer, but continues asking: “Why is that?” “Can you give a specific example?” “Here is a counterargument—what do you think about it?” It can also explain institutions or data as needed while moving the discussion forward. After aligning on the facts, it can ask again: “Given that, what do you still think?”

In the past, this kind of semi-structured interview—or even debate-like communication—required human participation. But today’s large language models are increasingly acquiring the ability to do this. Similar AI interview functions have begun to be released by frontier-model companies overseas, such as Anthropic, by the end of 2025. As a way to “draw out the tacit knowledge sleeping in the minds of experts and stakeholders, at scale and with speed,” AI interviews seem poised to develop in earnest not only in politics but in business as well.

When the dialogue logs from these interviews are analyzed afterward, insights directly relevant to policy design begin to emerge: “This concern arises from a misunderstanding of these assumptions,” or “If this condition is met, many people would find it acceptable.” What matters is not simply support or opposition, but eliciting the “assumptions,” “reasons,” and “conditions” together. That kind of informational “depth” is essential to Broad Listening.

### C) Speed: Shortening Democracy’s Lead Time

The third perspective is “speed”: how quickly can these voices be gathered and organized? Social issues often surface suddenly. Accidents, disasters, new technologies, and changes in international affairs all require the Japanese Parliament and local governments to devise responses within limited timeframes.

As noted earlier, the time available to prepare for parliamentary questioning can be as short as two or three days. With conventional methods—conducting interviews manually, compiling notes, and organizing issues by hand—it is often impossible to keep up.

By combining Broad Listening with AI interviews, this “lead time” can be dramatically shortened. In the author’s recent experience, when an interview link is shared on social media, well over a hundred stakeholders may begin responding in less than an hour, and the total amount of dialogue quickly reaches dozens of hours. This level of speed was difficult to achieve with previous methods.

To increase this speed even further, there is still room for improvement in how quickly one can reach the relevant stakeholders and experts, and how quickly they can be encouraged to respond. For political parties and civic communities, for example, it matters whether they have prepared lists of stakeholders and experts in advance, whether they have already earned the attention and trust of such people, and whether they have built enough credibility for the request to spread through networks among stakeholders and experts themselves.

### D) Were We Able to Extract Insights That Lead to Action?

The fourth perspective is whether we are extracting insights that actually lead to action. Here, what matters is not simply majority-style information such as “X% support, Y% oppose,” but how to discover “a single line of value, even if n=1.” As noted earlier, the former kind of information may be intentionally manipulated or heavily biased. Of course, quantitative information can be useful depending on how voices are collected, but the more broadly one solicits input on the internet, the harder it becomes to rely on such quantitative data.

So what kind of information leads to action? One way to think about it is to identify information that represents a meaningful deviation from the “baseline.” Policymakers usually already have a “current hypothesis” or a draft proposal serving as a starting point. Information that could alter that hypothesis or draft is meaningful. For example, a policy proposal created with good intentions may reveal completely unexpected concerns when viewed from the field. When that happens, the facts need to be verified and the hypothesis may need to be revised.

Another important factor is the power of “emotional episodes.” A high-resolution story from a single stakeholder can sometimes help us grasp reality more effectively than statistical information. Concrete, lived episodes—local, on-the-ground information that never appears in the market of public data—carry a special persuasive force.

Of course, anger based on factual misunderstandings, or conspiracy-laden narratives, can also come wrapped in “strong emotion,” so governance is needed to determine how facts are verified and to what extent such material should be used in policymaking.

Even so, being able to handle both logical proposals for revision and emotional episodes is, I believe, one of the key milestones for Broad Listening. Politicians and public officials move when both rational argument and emotionally compelling narrative are present.

### E) Completion Rate of Action: Are We Actually Producing Change?

The final, and most important, perspective is the completion of action. No matter how broadly, deeply, and quickly voices are gathered, and no matter how well they are organized into actionable insights, the Broad Listening loop is not complete unless it ultimately results in action in the real world.

From this perspective, it matters that Team Future became a national political party and gained the authority that comes with having members in the Japanese Parliament. By law, members of the Japanese Parliament are given options and powers such as the following:

- Voting for or against bills, treaties, and budgets
- Asking questions and speaking in parliamentary committees
- Submitting formal written questions to the government
- Officially submitting citizens’ voices to the Japanese Parliament as petitions or requests
- Submitting amendments to bills already under consideration
- Introducing entirely new bills

At this early stage in the development of Broad Listening, we have a hypothesis: that a structure integrating four elements—“political party,” “tech platform,” “policymaker,” and “media”—may maximize the value of Broad Listening. If the team that builds the system for gathering voices, the team that analyzes and interprets those voices into bills or institutions, and the team that pushes those bills or institutions forward in the political arena all exist separately in disconnected organizations, then the probability rises that information will be lost at some point in the chain.

Team Future, which has vertically integrated “listen,” “think,” “decide,” and “execute” within a single organization, occupies a unique position even by global standards. There are almost no examples anywhere in the world of a national political party undertaking this kind of effort while also maintaining a team of software engineers. Precisely for that reason, we believe Team Future has a responsibility to engage in extensive trial and error.

At the same time, of course, government institutions and the civic tech community also have major roles to play. Civic tech communities can do many things precisely because they do not hold political positions, and there are also many things that can only be achieved by local governments or central ministries. Ideally, the explorations of each player will be shared organically and connected with one another as Broad Listening develops.

## Conclusion: Cultivating a New System Together

As we have seen, Broad Listening is still a very immature technology and practice. Talk to the City, AI Anno, GitHub platforms, Idobata Policy, AI Interviewer, Public Listening AI... Not a single one of the projects mentioned here is something we can proudly declare “complete.”

Even so, by following the trial and error traced in this book, I believe readers will be able to sense the potential of these technologies and concepts. Their development depends not only on engineers, but on the cooperation of each and every citizen. I hope this book will serve as an opportunity for readers to think about the future of democracy.
