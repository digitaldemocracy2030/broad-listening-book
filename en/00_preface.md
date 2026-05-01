# Preface: The Current State and Future of Broad Listening

Commentary by Takahiro Anno

What is "broad listening"? This unfamiliar term is positioned as the opposite of the more familiar "broadcast."

Broadcast is a technology and culture in which one sender delivers information to many recipients at once. Our democracy, shaped through newspapers, television, radio, and social media, has fundamentally been designed around media that "deliver broadly."

Broad listening, by contrast, aims to place "listening broadly" at the center of democracy. Rather than leaving many people's voices scattered, it receives them as one system, transforms them into an understandable structure, and ultimately returns them to society in the form of policy or institutional change. It is an attempt to redesign that whole loop from both the technological and institutional sides.

That said, the technologies and practices discussed in this book are still in their infancy. They are far from perfect, and there are only a handful of successful cases. It may even be fair to say that the failures and unexpected side effects still outnumber the successes. Even so, we are convinced that broad listening will be an extremely important technology for the future of democracy.

Behind this conviction is a major technological shift. One element is the idea of "Plurality" advocated by Audrey Tang and others in Taiwan: not democracy as simple majority rule, but a view of democracy in which plural positions coexist and search for agreement. Another is the arrival of large language models (LLMs), which can verbalize tacit knowledge and structure vast amounts of text.

Against that background of ideas and technologies, I have been experimenting with implementing broad listening in real settings: elections, public administration, and the Japanese Parliament. I originally founded an AI startup as an AI engineer. In 2024, I ran in the Tokyo governor election and proposed bringing technology into elections. That same year, I became an advisor to GovTech Tokyo, a Tokyo metropolitan digital-policy affiliate, and explored whether such technologies could be used in local government policymaking. In January 2025, I co-founded the civic tech organization Digital Democracy 2030 with Ken Suzuki. In May 2025, I launched Team Mirai, a political party whose name means "Team Future"; in the July House of Councillors election, it won seats. I am now working from inside Nagatacho, Japan's political center, as the leader of a national political party, trying to "update democracy through digital technology."

## A Trail of Trial and Error

I would like to introduce four initiatives I have worked on toward implementing broad listening: (1) an open-source-style manifesto and AI-powered listening in the Tokyo governor election, (2) Shin Tokyo 2050, which brought that know-how into public administration, (3) Idobata Policy in the House of Councillors election, and (4) use of an AI interviewer after becoming a member of Parliament.

### 1. Tokyo Governor Election: Polishing a Manifesto Like Source Code

The first experiment was the 2024 Tokyo governor election. There, we built a system for updating the candidate's manifesto during the campaign in response to citizens' voices, using a method similar to software development.

At the core were three components.[^component]

[^component]: A component is a part or element that makes up a system, machine, or piece of software. This footnote was added to make the text easier to read for people unfamiliar with technical terms. Responsibility for this and the following footnotes: Yasukazu Nishio.

The first was a mechanism for publishing the manifesto on GitHub,[^github] a service commonly used by engineers, and managing issues and proposed changes there. Feedback from voters and stakeholders was registered as issues, and the policy team reviewed it while reflecting actual wording changes as pull requests.[^issue_pr] During the campaign alone, hundreds of issues and proposed changes were submitted, and many were adopted as manifesto revisions. The policy document was not a static PDF; it became a "living document," evolving as markdown files through versions 1.0, 1.1, 1.2, and so on.[^versioning]

[^github]: The name of a web service commonly used by IT engineers to share software source code. As of 2024, it had 180 million users.

[^issue_pr]: The names of GitHub mechanisms used to discuss problems and propose improvements.

[^versioning]: "Ver." is short for version, meaning an edition or release. Here, it can be read as "first edition," "second edition," "third edition," and so on. In software development, updates happen more frequently and in smaller units than in printed books, so this kind of versioning is customary. It is known as semantic versioning.

The second was the AI avatar "AI Anno." Through YouTube streams and other channels, it functioned as an AI town hall that kept answering voters' questions 24 hours a day, 365 days a year. In the end, it received nearly 10,000 questions. The AI had been trained on the policies, and its log data was accumulated as material for further policy revisions.

The third was opinion visualization using Talk to the City[^tttc] (TTTC). We used TTTC to collect online reactions scattered across X (formerly Twitter), YouTube comments, and other sources, separate them into clusters, and visualize on a website "what issues were being discussed and how much." It also functioned as a filter for extracting and organizing issues from an online space full of abuse and noise.

[^tttc]: Talk to the City is the name of a tool created and released by AI Objectives Institute, a US nonprofit, for visualizing large volumes of opinions.

The resulting policy manifesto received the highest rating among the candidates in an evaluation by a university research institution. It is fair to say that this produced some results. Looking back, however, every measure was still rough around the edges, and many challenges remained.

### 2. Working with Government: Broad Listening in Shin Tokyo 2050

The experiment in the governor election later led to implementation inside government itself. In the Tokyo Metropolitan Government's long-term strategy project Shin Tokyo 2050, we worked with GovTech Tokyo's in-house team to apply broad listening mechanisms to a real policy development setting.

Channels for gathering opinions expanded beyond online forms, X, and YouTube comments to include postal mail, email, and in-person interviews at places such as Ueno Zoo and science museums. In the end, more than 10,000 voices were collected and integrated and analyzed through a TTTC-based system.

GovTech Tokyo built the infrastructure for data collection, integration, and visualization in-house, while we were involved as external advisors on AI-analysis prompt design, parameter tuning, and how to interpret the analysis results. As the project progressed, new feature requests emerged, including deeper drill-downs and cluster reorganization. These later led to the development of Kouchou AI (広聴AI, Broad Listening AI).

### 3. The House of Councillors Election and the "Chat-Enabled Manifesto"

The next phase was the "chat-enabled manifesto" in the House of Councillors election. In the Tokyo governor election, we introduced a system for accepting direct policy-improvement proposals through GitHub, but GitHub inevitably created a high barrier for people who were not engineers.

That led to the development of Idobata Policy, which combined AI with MCP (Model Context Protocol).[^mcp] With this system, users no longer needed to think at all about the opinion-organizing machinery behind the scenes. They could simply express their views naturally through a chat interface. Knowledge of GitHub operations was no longer required; instead, the AI organized and summarized the content behind the scenes.

[^mcp]: The name of a communication method for connecting AI to external tools. In the chat-enabled manifesto, MCP allowed the AI to connect to and operate GitHub.

This made it possible to gather policy revision proposals not only from people with high IT literacy, but from a broader public. In the end, more than 8,000 proposals were submitted, several times the scale of the governor election in sheer volume.

But here too, new challenges emerged. The quality of the large number of proposed changes (pull requests) varied widely, and the process of coordinating who would review them, how, and which proposals would be adopted created an extremely heavy burden. There was also no pathway for feeding back the results or reasons for adoption or rejection to the original contributors. We succeeded in "widening the front door," but the design of the "organize, use, and return" part remained the next major task.

### 4. Putting It into Practice as a Member of Parliament: The AI Interviewer

Now, as a member of Parliament, I am trying to connect broad listening more directly to parliamentary debate.

Committee questioning in the Japanese Parliament might appear from the outside to allow a long preparation period. In reality, the time sense on the ground is very different. Once the schedule and bill are set, the lead time before submitting formal questions can be only a few days. Often, one must also ask questions in fields outside one's own expertise.

In that situation, we introduced the AI interviewer. This is a mechanism in which AI acts as the interviewer, gathering deep input from practitioners and affected people in a short period of time. In ordinary use of AI such as ChatGPT, the human asks questions and the AI answers. With an AI interviewer, the reverse happens: the AI asks questions about a specific theme, and the human answers. Unlike a normal survey, the AI keeps asking follow-up questions and digging deeper into the conversation. As a result, although it cannot match a human-to-human interview, it can hear deeper voices than a survey.

When AI interview URL links are distributed through social media, mailing lists, and similar channels, dozens of hours of dialogue can accumulate within a few hours. Large language models (LLMs) then organize and summarize those logs and generate reports that feed directly into parliamentary questioning.

Using this method, many previously unrecognized ideas, issues, and vivid episodes can be collected. The result is not just numbers showing support or opposition, but tangible information for thinking about "where and how this should be discussed in Parliament." There are still not many cases, and we are continuing to experiment, but the method so far feels promising.

## Future Challenges: How Should We Define the "Quality" of Broad Listening?

As we have seen, tools and mechanisms for "listening broadly," or practicing broad listening, have been tested in many ways and gradually improved. One question I personally consider important is how to define the "quality" of broad listening.

At present, I think the quality of broad listening can be defined from five perspectives: (1) breadth, (2) depth, (3) speed, (4) whether actionable insights were extracted, and (5) how much action was actually completed.

### A) Breadth: Quantitative Expansion and Its Limits

The most intuitive metric is "how many people did we hear from?" The number of participants and the number of submitted opinions or pull requests are easy to count as KPIs.[^kpi] In fact, from the Tokyo governor election to the House of Councillors election, the number of proposals gathered through our systems increased by orders of magnitude. In terms of the sheer quantity of voices, this was a dramatic advance.

[^kpi]: KPI is a management term and an abbreviation for key performance indicator. It is a metric used to check whether the process toward achieving an objective is on track and to identify and improve gaps from the goal early.

In politics, however, it is extremely dangerous to treat "quantity" as a direct proxy for "the will of the people." Deliberate attempts to manufacture majorities and sampling bias are always present. A small minority strongly opposed to a policy is often far more motivated than a large number of indifferent people and may submit opinions repeatedly with much greater intensity. Any attempt to estimate society-wide support, opposition, or structure from a biased sample requires considerable caution.

Moreover, simply increasing the number of people who respond is difficult in the first place. People willing to go out of their way to fill out a form on a political issue are only a tiny fraction of the whole population. Even if a place for raising voices is provided, doing so still imposes a cost on citizens. I believe the important thing is to steadily cultivate trust that "the voices people actually raise might change society."

Taiwan's civic participation platform JOIN is a good example. It is an officially recognized government web service where any citizen can post an idea such as "I wish there were a policy or law like this." What makes the service special is its prior promise: if a proposal gathers more than 5,000 supporters, the responsible ministry must review it and respond. Good proposals are actually reflected in policy, and even when they are not adopted, the reasons are explained.

In fact, over the past decade, about 10,000 citizen proposals have been submitted, of which about 370 surpassed the 5,000-support threshold. From those, around 200 policies or laws have actually been enacted and implemented. Because the recognition has spread that "if you use JOIN, information will reach the government and can produce real results," citizens can feel that it is worth paying the cost to speak up. It has reportedly even appeared in Taiwanese textbooks recently.

Conversely, if a venue is merely provided but does not lead to real action, citizens are left with the negative experience of "I worked hard to answer the survey, and it came to nothing." If trust is earned, participation will keep growing; if trust is not earned, the system will quickly wither. Whether the system enters a good cycle or falls into a bad cycle is decisively important.

This is why I chose the path of creating a national political party. Doing so guarantees the ability to take action inside Parliament. The existence of this exit establishes credibility and helps secure a certain number of respondents for broad listening.

Of course, other actors, including civic tech groups, companies, politicians, local governments, and public administration, should also be able to build trustworthy places where people feel that "if I voice my opinion here, something might change." In fact, many efforts cannot be sustained by any one actor alone. It is desirable for many players to steadily accumulate their own efforts.

"Breadth" is a difficult metric that must be neither overestimated nor underestimated. It is worth emphasizing again that if policymakers refer to it without recognizing its biases, there is a risk of creating undesirable policies that do not align with reality.

### B) Depth: From Surveys to Dialogue

The second perspective is "how deeply are we hearing people's voices?" Traditional surveys and public comment processes often end with one question and one answer. They reveal only surface-level attitudes such as "support," "oppose," or "neither," and do not reach deeper questions such as "Why do you think that?" "What experiences is your view rooted in?" or "Would you still make the same argument after hearing opposing views?"

Here, we are focusing on the AI interviewer approach described above. AI acts as the interviewer and advances the discussion interactively. It does not stop at one question and one answer; it keeps asking questions such as "Why is that?" "Can you give a specific example?" and "Here is a counterargument. What do you think?" while also explaining institutions or data as needed. After aligning on the facts, it may even be possible for the AI to present counterarguments.

In the past, this kind of semi-structured interview or debate-like communication required a human. Today's large language models, however, are increasingly acquiring the ability to conduct these dialogues in place of humans. As a way to draw out tacit knowledge sleeping in the minds of experts and affected people at scale and speed, AI interviews seem likely to develop in earnest not only in politics but also in business.

By analyzing the dialogue logs obtained from interviews, insights directly relevant to policy design can emerge: "This concern arises from a misunderstanding of these assumptions," or "If this condition is met, many people would find it acceptable." What matters is not simply support or opposition, but eliciting the assumptions, reasons, and conditions together. That kind of informational depth is essential for broad listening.

### C) Speed: Shortening Democracy's Lead Time

The third perspective is speed: how quickly these voices can be gathered and organized. Social issues often surface suddenly. Accidents, disasters, new technologies, and changes in international affairs all require Parliament and local governments to devise responses within limited time.

As noted above, the time available to prepare parliamentary questions can be only two or three days. Conventional processes of manually interviewing stakeholders, compiling notes, and organizing issues often cannot keep up.

Combining broad listening with AI interviews can greatly shorten this lead time. When an interview link is shared on social media, hundreds of affected people can begin responding in less than 24 hours, and the total interview time quickly reaches dozens of hours. Achieving this level of speed was difficult with previous methods.

To increase that speed further, we need to think carefully about how to approach affected people and experts. Whether relationships have been built in advance and whether trust has been steadily accumulated are extremely important.

### D) Were We Able to Extract Actionable Insights?

The fourth perspective is whether actionable insights have been extracted. Here, what matters is not simply majority-style information such as "X% support, Y% oppose," but how to discover "information that is valuable even if n=1."[^n1] As noted above, the former kind of information may be intentionally manipulated or heavily biased. Of course, quantitative information can be useful depending on how voices are collected, but the more broadly one solicits input on the internet, the harder it becomes to rely on quantitative data.

[^n1]: Here, n means the sample size. The point is that there are cases where an opinion is valuable even if only one person wrote it, and the important question is how to find those cases.

So what kind of information leads to action? One way to think about this is to find information that differs meaningfully from the "baseline."[^baseline] Policymakers usually already have a current hypothesis or draft proposal. It is meaningful to extract information that could affect that hypothesis or draft. For example, a policy proposal created with good intentions may reveal unexpected concerns from people in the field. When that happens, the facts need to be verified and the hypothesis may need to be revised.

[^baseline]: A baseline is a reference line or starting point for comparison or evaluation. Here, it refers to the current hypothesis or draft proposal held by policymakers.

Another important factor is the power of "emotional episodes." A high-resolution episode from one affected person can sometimes help us grasp reality more effectively than statistical information. Local, on-the-ground information that does not circulate in public data markets, and concrete episodes with human texture, can have special persuasive force.

Even so, being able to handle both logical proposals for revision and emotional episodes is, I believe, one important milestone for broad listening. Politicians and public officials move when both rational logic and a story that moves the heart are present.

### E) Completion Rate of Action: Are We Actually Producing Change?

The final and most important perspective is the completion of action. No matter how broadly, deeply, and quickly voices are gathered, and no matter how well they are organized into action, the broad listening loop is not complete unless action ultimately occurs in the real world.

From this perspective, I believe it matters that Team Mirai became a national political party. The following kinds of action became possible:

- Voting for or against bills, treaties, and budgets
- Asking questions and speaking in parliamentary committees
- Submitting formal written questions to the government
- Officially submitting citizens' voices to Parliament as petitions or requests
- Submitting amendments to bills under consideration
- Introducing entirely new bills

At this early stage of broad listening, I believe there are things made possible by integrating four elements: political party, tech platform, policymaker, and media. If the team that builds the system for gathering voices, the team that analyzes and interprets those voices into bills or institutions, and the team that pushes those bills or institutions forward in the political arena all exist separately in disconnected organizations, the probability rises that information will be lost at some connection point.

Team Mirai, which is trying to consolidate "listen," "think," "decide," and "execute" within one organization, occupies a globally unique position. Around the world, there are very few examples of a national political party undertaking this kind of effort while maintaining a team of software engineers. Precisely for that reason, we believe Team Mirai has a responsibility to engage in many forms of trial and error.

At the same time, of course, the roles to be played by public administration and civic tech communities are also very large. Civic tech communities can do many things precisely because they do not hold political positions, and there are also many things that only local governments or central ministries can realize. Ideally, the explorations of each player will be shared organically and connected with one another as broad listening develops.

## Conclusion: Cultivating a New System Together

As we have seen, broad listening is still a very immature technology and practice. Talk to the City, AI Anno, GitHub manifestos, Idobata Policy, the AI interviewer, Kouchou AI... Not a single project mentioned here is something we can proudly declare complete.

Even so, by following the trial and error traced in this book, I believe readers will be able to sense the potential of these technologies and concepts. Their development depends not only on engineers, but also on the cooperation of each and every citizen. I hope this book will give readers an opportunity to think about the future of democracy.
