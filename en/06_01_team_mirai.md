# Team Mirai’s Efforts in the 2025 House of Councillors Election

Written by: Yasukazu Nishio  
(Footnote:  
This section is a reconstruction for this book based on the author’s blog post, [Insights Gained from Team Mirai’s Social Experiment, the “Interactive Policy Platform”](https://note.com/nishiohirokazu/n/nb35e8d526fd4).)


In May 2025, AI engineer Takahiro Anno launched a new political party, “Team Mirai” (meaning “Team Future”), and entered the House of Councillors election. When the votes were counted on July 20, Team Mirai won one seat and about 2.6% of the vote, meeting the legal requirements to become a national political party.

This chapter explains how the broad listening process evolved during that election period, focusing on the “interactive policy platform” that was operated throughout the campaign.

## From the Tokyo Governor Election to the House of Councillors Election: Integrating Three Projects

In the 2024 Tokyo governor election discussed in the previous chapter, three projects were running in parallel:

1. **AI Town Hall**: the AI YouTuber “AI Anno” answered questions 24 hours a day
2. **Broad Listening**: public opinion was visualized with Talk to the City
3. **Open-source policy improvement**: policies were published on GitHub, where improvement proposals were accepted

![alt text](images/06_01_tokyo_gov_manifesto.png)
Mr. Anno’s electoral platform in the 2024 Tokyo governor election

These projects were interconnected. The platform on GitHub served as the latest data source—the Single Source of Truth (SSoT). AI Anno referred to it when answering questions, while information from diverse channels was visualized and observed through Talk to the City.

![alt text](images/06_01_three_projects.png)
How the three projects were connected

Broad listening itself was explained in detail in Chapter 4, Section 1. During the Tokyo governor election period (June 21 to July 6, 2024), the GitHub-based policy improvement project received 232 issue submissions and 104 change proposals, of which 85 were incorporated into the policy. AI Anno answered a total of 8,600 questions.

However, this system had a major problem: many voters could not use GitHub[^github].

[^github]: This was, of course, already known before release. But because the schedule was tight, and because no matter what kind of service you build there will always be people unfamiliar with it, we chose a service that at least some people already knew how to use. As a result of this experiment, we learned that voters were able to use GitHub more than we had expected.

## The Interactive Policy Platform: Making Dialogue the Entry Point

In the 2025 House of Councillors election, the three projects were integrated in order to solve the problem that GitHub posed too high a barrier.

![Figure: The spoken manifesto flow, where AI conversations with voters become policy proposals on GitHub.](images/06_01_manifesto_flow_en.png)
The interactive policy platform

First, the AI chats one-on-one with each voter. As with the earlier AI Anno, the AI can answer questions—but this time the reverse is also possible. Acting like an interviewer, the AI listens to voters, asks follow-up questions, and helps them put their thoughts into words. In the end, the AI operates GitHub and submits a policy change proposal (a pull request) to GitHub.

This meant voters no longer needed to use GitHub directly.

This “interactive policy platform” was released on May 16, 2025. Four days later, on May 20, Mr. Anno posted what he described as a happy problem[^anno-ureshii].

> We have received more than 1,300 proposal documents from everyone regarding Team Mirai’s platform. Honestly, that’s more than ten times what we expected, so it’s a very welcome problem to have.

[^anno-ureshii]: Takahiro Anno, post on X, May 20, 2025. https://x.com/takahiroanno/status/1924760499729436716

In the Tokyo governor election, 104 change proposals had been collected. In just a few days, this new system gathered nearly an order of magnitude more. This shows that the effect of switching from a specialized entry point—posting directly through GitHub—to a conversational UI that anyone could use was even greater than expected.

In the end, 8,559 proposals related to policy alone were submitted. There were more than 9,600 change proposals in total, and the figure of 8,559 counts only those related to policy after excluding proposals about changes to the system itself.

## Visualization and Observation with Public Listening AI

Although it was not publicly announced, by May 22 the team had internally reached a point where they could visualize and observe submissions using “Public Listening AI,” a broad listening tool developed by the Digital Democracy 2030 Project[^dd2030].

[^dd2030]: The Digital Democracy 2030 Project and Public Listening AI will be explained again in detail in Chapter 11.

Public Listening AI is a Japan-developed system that improves on Talk to the City, which had been used during the Tokyo governor election. Based on experience from projects such as broad listening for the 2024 House of Representatives election in collaboration with Nippon TV, and broad listening conducted with the Tokyo Metropolitan Government for the formulation of a long-term strategy toward “Tokyo in the 2050s,” new features were added that were felt to be necessary for better broad listening.

When Talk to the City was used during the Tokyo governor election, the analysis program ran on the analyst’s local computer. As a result, one challenge was how to share the generated reports with policy staff and social media staff. With Public Listening AI, analysis is performed on a server, and as soon as report generation is complete, a shareable URL is issued. Staff members can view the report simply by opening that URL.

This time, dozens of reports were created internally, but most were never made public. Although generating reports had become easier, staff still needed to review each one before publication to confirm that its contents were suitable for release. For Team Mirai, which fielded ten candidates with a small staff and was contesting its first national election, pre-publication review of reports was not something they could prioritize highly.

At this point, recall the distinction introduced in Chapter 1 between “broad listening as a tool” and “broad listening as a cycle of augmented deliberation.” Ideally, the “cycle of augmented deliberation” should be realized. At the same time, broad listening technology can also function on its own as a tool that helps people better understand public opinion—in other words, like a pair of glasses. Team Mirai’s use of Public Listening AI in the House of Councillors election mainly delivered this latter benefit.

## Changing the Way Issues Are Categorized: The Value Broad Listening Brings

Looking at the actual voices gathered through the platform led to a discussion, and ultimately a decision, that it would be clearer to group opinions that had originally been scattered across multiple sections into a new section called “Welfare.” This kind of “change in categorization” is one of the effects produced by broad listening tools such as Public Listening AI and Talk to the City. A similar restructuring of the policy framework was also carried out in the Tokyo Metropolitan Government case discussed in Chapter 5.

In Team Mirai’s activities, the addition of a new category meant it became necessary to analyze existing change proposals and determine whether they should be reassigned to the welfare category. To do this, all roughly 2,000 change proposals that existed at the time were categorized using GPT-4o. The reclassification cost $0.021 per item, for a total cost of $9.34[^welfare-label].

[^welfare-label]: Welfare label analysis experiment report. https://github.com/team-mirai-volunteer/policy-pr-hub/blob/99917864d079e432053c94a86765d6edad850e5b/docs/welfare_label_analysis_experiment.md

## How to Process a Massive Volume of Proposals

In the end, 8,559 policy-related proposals were submitted. That number arrived in just a few weeks during the election period. Even if each one took only five minutes to read, it would require more than 700 hours to read them all. Even divided among ten staff members, that would still mean 70 hours per person.

Very few organizations in the world have received proposals on this scale in such a short period, so it is worth looking back at what actually happened.

### Improving Processing Efficiency with AI

Even slightly before Mr. Anno’s “happy problem” post, once the team saw the pace of activity in the first few days after release, it was already possible to foresee that they would receive many times more proposals than in the Tokyo governor election. The engineering team therefore began building support systems to process the proposals efficiently. For example, automatic labeling of proposals made it possible to filter and browse them by category even within GitHub’s web interface. The categories corresponded to teams of staff with relevant expertise, making it easier for each person to find what they needed to read.

The team also built a system to collect and organize proposal data posted to GitHub via the GitHub API. This made it possible to ask ChatGPT, for example, “Give me a rough summary of what people are saying,” and receive output like the report below, explaining which proposal numbers were related to each type of suggestion and what kinds of revisions they were trying to make.

![alt text](images/06_01_pr_overview.png)
A report generated by ChatGPT

This made it possible to work through the problem of “there are too many proposals, and we don’t know where to start” by consulting with an LLM. However, the mechanism for retrieving data from GitHub—the GitHub API—had a limit of 1,000 items per hour, and once the number of proposals exceeded 1,000, the system stopped working. It became necessary to change the system so that it would run hourly, retrieve only the differences, and update the aggregated results incrementally.

### The Problem of Change Proposals Being Submitted to the Table of Contents

When the actual change proposals were classified by target file, it became clear that many proposed edits were being submitted to the README file—the table of contents page.

In this “interactive policy platform,” users could choose individual pages of the platform from the table of contents, divided into categories such as “Healthcare” and “Education.” They could then ask the AI questions about the page being displayed or make change proposals through conversation. In other words, the system was designed on the assumption that users would click into each page, read it, and then ask questions or submit change proposals. From an engineer’s perspective, this was a simple and understandable implementation. But for general users, it was not the best possible experience design (UX design). Users who opened the prominently displayed page with the chat box often started chatting immediately. As a result, they might ask the table of contents page about something written on the healthcare page and be told it could not be found, or they might propose as a change to the table of contents something that was already written on the healthcare page.

For example, it would probably have been better to add features such as “if a user asks a question on the table of contents page, answer after reading the whole platform,” or “if a user tries to make a proposal from the table of contents page, first open the most relevant page.” However, implementing such features would have required major design changes, and it would have been difficult to bolt them on after the service was already running. There was no room to do that during the election period.

### The Problem of Multiple Proposals Colliding on the Same Passage

In the “interactive policy platform,” the AI automatically created concrete revision proposals. For example, a user might suggest “add ‘improved treatment for childcare workers’ to the section on childcare support.”

But problems arose when multiple different revision proposals were submitted for the same passage. If Person A proposed “add improved treatment for childcare workers,” and Person B proposed “add expanded after-school childcare,” both targeting the same section, then adopting one would not automatically incorporate the other. In some cases, dozens of proposals were trying to revise the same passage.

This time, Team Mirai solved the problem through sheer effort: policy staff read all the conflicting revision proposals, extracted the intended changes, and created a new revision proposal themselves. This was far too burdensome, and the author believes future systems will need to improve on it.

The author believes that creating “change proposals” was itself too specific a task from the outset. When AI interviewers go out and listen to large numbers of people, what should they really be asking? Should it be “How should the wording of the platform be revised”? Or should it be “What are you struggling with, and what do you want solved”? When broad listening is used to gather “the voices of many people,” the ease with which those voices can later be used depends on what kind of voices are collected.

In a keynote speech at LibreCon 2016[^librecon], Audrey Tang explained four steps for deliberation. In Step 1, observable facts are collected from all stakeholders. In Step 2, people’s emotional responses to those facts are collected. Only in Step 3 are ideas solicited. The quality of an idea is judged by whether it takes many people’s emotions into account. Seen in this light, past cases often seem to have asked people immediately for “ideas” or “requests,” and as a result ended up collecting self-centered ideas that reflected only the speaker’s own concerns. As we build systems for large-scale deliberation in the future, this four-step framework is an important perspective.

[^librecon]: YouTube video, "Keynote for LibreCon 2016" https://www.youtube.com/watch?v=ukna6wZg-8A

### A Sense of “Being Heard” and Bidirectional Traceability

In Team Mirai’s “interactive policy platform,” the AI interviewer listens to people and creates proposals for them. What was lacking was the design of the user experience *after* that listening and proposal creation had taken place.

The interactive policy platform sends change proposals to GitHub in the form of pull requests, and then it simply ends there. Users had little visibility into what happened afterward. For example, all change proposals were in fact being divided up and read by more than ten staff members, but there was no feedback indicating that someone had “started reading” a given proposal.  
As a result, comments expressing “it doesn’t feel like my opinion was really received” appeared here and there. Ideally, users should be able to see a list of the proposals they submitted and the processing status of each one. It would also be good to let users who wish to do so register an email address or similar contact information so that push notifications can be sent. In his book *Applied Imagination*, brainstorming advocate Alex Osborn recommends that “when ideas are adopted, the meeting participants should be informed and encouraged,” and that “a brief note of thanks from those who benefit from the suggestion should be distributed to everyone.”

In visualizations such as those produced by Public Listening AI, one can drill down into summarized opinions to confirm “where they came from.” By contrast, many existing systems have not provided users with any way to confirm “where their opinion went.” One of the major lessons from this case was the realization that traceability is needed not only for “where it came from,” but also for “where it went.”

![Figure: Diagram contrasting where opinions came from with where they went in the policy process.](images/06_01_opinion_flow_en.png)
Bidirectional traceability is needed for both “where it came from” and “where it went”

## When Discussion Starts on GitHub

At the planning stage of Team Mirai’s “interactive policy platform,” it was not assumed that ordinary users would use GitHub directly. The system was developed on the premise that doing so would be difficult. However, because GitHub was public, direct use was still possible. In this experiment, one of the 8,559 change proposals received an extraordinary 236 comments. The second-most-commented proposal had only 10 comments, which shows just how exceptional that was.

During the Tokyo governor election, an AI filter had been implemented to screen out abuse and harassment. This time, however, because lively discussion on GitHub had not been anticipated, no such filter was included. As a result, GitHub’s comment section ended up functioning like a classic web message board, with exchanges of sharp and hostile language. In addition, some people in the comment section used Mr. Anno’s face icon without permission, while others who were in no position to speak on behalf of the party made statements that could be mistaken for official party views. On X (Twitter), some influencers even posted misinformation after misidentifying this discussion as “internal debate within Team Mirai.”

Software engineers familiar with open-source development on GitHub know that GitHub is a place where anyone can write comments as long as they create an account, and that it is not an “internal discussion.” But people who arrived by following a direct link from a social media post may well have encountered the discussion without any explanation of the nature of the space.

The author believes that exposing GitHub directly should be avoided, and that it should instead be fully covered by a custom-designed interface. In an internet age where people can link in from anywhere, designing a good user experience requires being able to design every screen yourself.

## Summary of the Insights Gained

During the summer of 2025, several principles were articulated that are useful for realizing digital democracy.

1. **The importance of entry-point design**: switching from GitHub to a conversational UI increased the number of proposals by about 80 times
2. **The value of restructuring the framework**: observing real voices makes it possible to reconstruct the policy framework
3. **The need for bidirectional traceability**: both “where it came from” and “where it went” are necessary
4. **The importance of designing the space itself**: spaces for discussion need usage guidance and moderation

Broad listening as a component (such as visualization through Public Listening AI) appears in many places throughout this cycle, but those components alone are not the main actors. What matters is that the entire broad listening cycle—opinion gathering → structuring → deliberation → decision-making → feedback—was attempted as a single social experiment.

In the Tokyo governor's race, Mr. Anno demonstrated what had only recently become technically possible, and that had a major impact on society a year later. Team Mirai's demonstration in this House of Councillors election will likely have an even greater impact in the future.

## About the Idobata System

It would be undesirable for the contribution that brought this development into the world to be attributed solely to the most visible actor, “Team Mirai,” so I want to take the space here to explain this clearly.

One component of this “interactive policy platform” project—the part that chats with people and sends change proposals to GitHub—was developed by the “Digital Democracy 2030” project introduced in Chapter 11. It is best understood as a derivative of the “Idobata” system, which was created as a deliberation support system. During 2025, additional development was also underway to introduce this system to political parties other than Team Mirai, but unfortunately those efforts faded away after the person on the other side who had been enthusiastically promoting the project was reassigned. As an example of private-sector use, Chapter 9 discusses a case at Cybozu. The Idobata project is not “owned by Team Mirai”; it is a public asset that anyone can use.

The “Idobata” system is somewhat complex. Within the same repository, two related systems were developed and used for different purposes: “Idobata Policy,” which formed part of the “interactive policy platform,” and “Idobata Vision,” a town-hall-like mechanism in which people chat with AI and reports summarizing their views are generated.

### Shared Concept
In these systems, people are not “in the same chat”; rather, they are “in separate chats” but connected through AI. This is a new form of communication—one that was not even technically possible five years ago.

![alt text](images/06_01_idobata_chat_structure.png)
(A) People in the same place, (B) individual AI chats, and (C) people connected through AI

Let us dig a little deeper while looking at the figure.  
In A, people exist in the same communication space.  
This is the form taken by pre-computer “people gathering and talking.”  
Since the spread of the internet, it has also become widespread in the form of online bulletin board systems, chat, and social media.  
The dramatic increase in people with experience of this kind of communication was likely influenced by the 2011 Great East Japan Earthquake, when phone lines were down but social media remained stable, and by the rapid spread of smartphones and LINE in the years that followed.

B is AI chat. It became widely known after the release of ChatGPT in 2022.  
In this model, what each person says to the AI is not shared with others; it is a closed world between one user and the AI.

C is the new form of communication. People chat with AI just as in B, but the AIs share information with one another. In implementation terms, this can be done either by sharing the chat logs themselves or by sharing only the insights derived from those logs. In this figure, the fact that the chat logs between humans and AI (the squares) are not included within the shape representing information sharing among AIs (the gray shape on the right) indicates that the chat logs themselves are not being shared.

This figure is intentionally simplified to make the newly emerging communication structure easy to understand. In reality, there is plenty of room for careful design in how to implement the bidirectional arrows that represent the exchange of information. For example, in Idobata Policy, the square of information shared by the AIs contains “the current platform,” and the AI that has chatted with a person writes a proposed update to that platform on GitHub. A human then approves it, the platform is updated, and from that point on the AI reads the latest updated platform when speaking with new users. This fairly complex process is represented by the bidirectional arrows showing that “the AI reads and writes shared information.”

In *The Public and Its Problems* (1927), John Dewey argued that changes in technology alter the ways people interact, and that this in turn changes the shape of “the public.” The technological advances since ChatGPT are doing exactly that: they are reshaping the public itself.
