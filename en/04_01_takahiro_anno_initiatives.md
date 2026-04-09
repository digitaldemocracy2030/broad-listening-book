# Takahiro Anno’s Initiatives in the 2024 Tokyo Governor Election

Author: Hirokazu Nishio

In the July 2024 Tokyo governor election, candidate Takahiro Anno’s campaign adopted “digital democracy” as a central theme and fully introduced broad listening technology into its election activities. The author, Hirokazu Nishio, participated in this effort as one of Anno’s technical supporters. This section reports on the initiative in practice and the insights gained from it.

## Encountering Talk to the City

This initiative began about two months before the governor election.

The author and Mr. Anno had known each other for some time through their roles as mentors for Mitou Junior, a Japanese program for mentoring young tech talent run by the Mitou Foundation[^mitou-jr]. In the Mitou Junior community, the concept of Plurality—advanced by Audrey Tang, Taiwan’s Digital Minister, and others—had been a topic of discussion since 2023, partly because Rikuto Niimi, one of the organizers of “Plurality Tokyo” held in April 2023, was also a Mitou Junior alumnus. The author was leading the Japanese translation project for a book on Plurality by Audrey Tang and others, and on May 13, 2024, Mr. Anno contacted him saying he wanted to hear more about Plurality in detail.

The author introduced Mr. Anno to Haruyuki Seki of Code for Japan, one of the organizers of Plurality Tokyo. On May 23, a meeting was held between Mr. Seki, Mr. Anno, and Nishio, and during that meeting Mr. Seki introduced a tool called “Talk to the City” (TTTC). TTTC is an open-source tool developed by the AI Objectives Institute that uses AI to cluster large-scale text data and visualize the distribution of opinions[^aoi-tttc].

On May 30, the author tried TTTC. At Mr. Anno’s suggestion, the test dataset consisted of roughly 20,000 public comments related to AI. The next day, May 31, Mr. Seki gave a keynote lecture titled “Updating Society and Democracy Through Civic Tech” in the organized session “AI and Democracy” at the 2024 Annual Conference of the Japanese Society for Artificial Intelligence, where he presented the analysis results[^hal-sk-jsai].

[^hal-sk-jsai]: Haruyuki Seki, “Updating Society and Democracy Through Civic Tech,” keynote lecture in OS-8 “AI and Democracy,” 2024 Annual Conference of the Japanese Society for Artificial Intelligence, May 31, 2024. https://speakerdeck.com/halsk/sibitukutetukuniyoru-she-hui-tomin-zhu-zhu-yi-noatupudeto (analysis results appear on slide 23; the analysis report is available at https://github.com/nishio/tttc_aipubcom_japan)

At that time, the author also provided, under a CC0 license, a hand-drawn diagram of “broad listening” from an article he had contributed in 2023 to the journal of the Information Processing Society of Japan[^ipsj]. That diagram was later adopted into Mr. Anno’s electoral platform and was shown on television and in several other media outlets.

[^ipsj]: “Not Subjective or Objective, but from One Person’s Subjectivity to Many People’s Subjectivities,” an article in the special feature “Tools for a New Era, ChatGPT: Exploring Its Potential from 14 Perspectives,” Information Processing, Vol. 64 (2023), No. 9, journal of the Information Processing Society of Japan

![](images/04_01_broadlistening_concept.png)
Conceptual diagram of broad listening (CC0)

In this way, even before the official start of the governor election campaign, TTTC was actually run and its potential was confirmed. Based on that experience, Mr. Anno decided to conduct broad listening using Talk to the City during the election period. In addition to the author, the analysis team included a social media analysis researcher referred to here as M, who had been Mr. Anno’s contemporary in the same university lab and club[^m-note].

This preparation period made it possible to produce more than 30 reports during the campaign.

[^m-note]: M, “The Tokyo Governor Election from the Perspective of a Social Media Researcher: Here’s What’s Amazing About TTTC,” note, July 2024. https://note.com/m_datasci/n/n3a5f4b9cdee5

## Why Broad Listening in an Election?

In an election, it is important for candidates and campaign teams to understand what voters care about, but each conventional method has its limits. Street speeches and rallies tend to attract highly motivated supporters, while opposing views are rarely heard. Surveys are shaped by how the questions are designed, and they only yield information on the narrow topics the survey intended to ask about. On social media, the loudest voices stand out, making the silent majority difficult to see.

Broad listening does not eliminate these biases entirely. However, by gathering diverse voices through new channels different from conventional ones, it becomes possible to grasp the overall landscape of opinions that existing methods struggle to reveal. That is why we believed it would be useful in an election.

## Mechanisms for Gathering Voices

The Anno campaign collected voters’ voices through multiple channels.

**#TOKYOAI hashtag**: People were encouraged to post on X (formerly Twitter) using this hashtag, and posts were collected by searching via the X API. When necessary, searches could also be conducted using keywords other than this hashtag.

**Anonymous suggestion box**: Using Google Forms, the campaign added a mechanism allowing anyone to submit opinions anonymously. Many people do not like making political statements on social media, and many do not even have Twitter accounts. The anonymous suggestion box played an important role in capturing those voices.

**Questions for AI Anno**: The campaign offered “AI Anno,” a system that answered people’s questions via YouTube Live. Questions submitted to AI Anno also became a useful source of information for understanding what people were interested in. Technically, these were collected through the YouTube comments API.

**AI voice phone**: By combining IP telephony with real-time speech-to-text, the campaign connected telephone calls to the AI Anno question-and-answer system. This was designed as a way for people less comfortable with electronic text input to interact through the more familiar medium of voice. Like the questions submitted to AI Anno on YouTube, this also became a useful information source.

**My Number Voting**: POCKETSIGN released “My Number Voting,” an experimental electronic voting app that allowed users to express support or opposition to policies after verifying their identity with a My Number card under My Number, Japan’s national ID system[^mainatohyo]. Mr. Anno introduced it on his official website and also received the voting data. This later became input data for the Polis analysis discussed below.

[^mainatohyo]: https://prtimes.jp/main/html/rd/p/000000036.000110743.html

**GitHub**: People were also able to submit proposals regarding the candidate's policy platform on GitHub[^github-manifesto]. The process of improving the candidate's policy platform itself became observable on GitHub, and anyone could obtain the latest version of it. This was true not only for humans, but also for AI Anno.

The visualizations created with Talk to the City from data collected through `#TOKYOAI` and the anonymous suggestion box became widely recognized because of the strong visual impact of the scatterplots, which in turn helped raise awareness of broad listening. At the same time, however, this also led some people to think that broad listening was nothing more than this scatterplot-style representation.

Mr. Anno’s vision was a much larger system. In the governor election platform, it was described as a “cycle of listening, refining, and communicating.” He believed it was important to run this cycle quickly. Visualization was only one component of that cycle. GitHub and AI Anno were components mainly responsible for other parts of the system, but as operations progressed it became clear that they too could be made targets of Talk to the City visualization. In particular, AI Anno and Talk to the City were initially seen as unrelated components, but once question data began to accumulate, it became clear that this too was simply another form of information gathering.

![alt text](images/04_01_kiku-migaku-tsutaeru_en.png)
The cycle of “listening, refining, and communicating” (quoted from Mr. Anno’s electoral platform)

## Visualizing Opinions with Talk to the City

Opinions gathered from posts using the `#TOKYOAI` hashtag and from the anonymous suggestion box were analyzed with TTTC.

TTTC’s processing can be broadly divided into three stages:

1. **Preprocessing (extraction)**: a generative AI extracts opinions from posts
2. **Clustering**: the text is converted into vectors with thousands of dimensions, and similar opinions are grouped together
3. **Postprocessing (labeling and summarization)**: a generative AI assigns labels to each cluster and summarizes representative opinions

In the end, the distribution of opinions is visualized as a two-dimensional scatterplot, producing a report that allows users to survey what kinds of opinion groups exist.

More than 30 reports were produced during the election period. These included a visualization of 951 opinions from the anonymous suggestion box.

## Reflecting Analysis Results in the Platform

The analysis conducted with TTTC was actually used in policy formation.

For example, analysis of the anonymous suggestion box revealed, across multiple posts, calls to abolish income limits in child-care support. In response, when the platform was updated from ver. 1.0 to ver. 2.0, that request was incorporated.

To help readers form as concrete an image as possible, I will introduce some scatterplots from here on. However, it is important to note that Talk to the City is fundamentally an interactive browser-based system, and once it is presented as a static image, more than 90% of the information is lost.

Below is a visualization of 951 opinions extracted from submissions to the anonymous suggestion box. Each point corresponds to one opinion.

![](images/04_01_anonymous_opinion_scatter_en.png)
Scatterplot of submissions to the anonymous suggestion box

In practice, analysts do not usually look at images like this. More often, they open a full-screen map like the one below and observe the distribution on a large display. In this figure, opinion groups are color-coded, and the group titles appear on hover. However, this may be difficult to see on the printed page. If you open the author’s blog post[^shotokutaishi] on a PC, you can view a large full-color image.

![](images/04_01_fullscreen_map.png)
Full-screen map

[^shotokutaishi]: “Broad Listening: The Technology That Makes Everyone Prince Shōtoku” https://note.com/nishiohirokazu/n/n15a60978113d

Let us look more closely at one opinion group. Each point corresponds to an opinion, and the AI groups similar opinions together. The AI also generates an explanation of what kinds of opinions are contained in that group. In the figure below, the AI explains that this group contains voices calling for the abolition of income limits in the context of child-care support.

![](images/04_01_cluster_detail_en.png)
Detailed analysis

Each point can be traced all the way back to the original post. In other words, the system allows step-by-step elaboration: first gaining a rough understanding, and then examining the details.

![](images/04_01_comment_detail_en.png)
An example showing a specific comment. It conveys both one household’s situation under income limits and the urgency of their appeal.

Because similar opinions are visualized as gathering near one another, looking at the points around a given point makes it possible to find related opinions. Rather than reading only one person’s description of a single opinion, one can read it together with descriptions by others saying similar things. This makes it possible to observe an issue from multiple directions and gain a more multifaceted understanding. This is a major strength of visualizing opinions in a scatterplot, but it is also something that cannot really be experienced by merely looking at static images without using the interactive system.

By examining the area around this point, related opinions were discovered not just from this one post but from multiple posts. This suggested that the issue was important to many people. The policy team then examined the detailed circumstances and budgetary feasibility, concluded that a change should be made, and Anno made the decision to update the platform. This is one concrete example of a policy platform update driven by broad listening.

The revision history of this election policy platform is published on GitHub[^github]. It makes transparent when, why, and how changes were made. In addition, explanations of updates were periodically turned into articles by AI[^note]. The details are explained in Ito’s article, “We’ve Started Operating a System That Uses AI to Automatically Turn Takahiro Anno’s Policy Updates into Articles!”[^note2]. The approach made it easier for people to understand and keep up with how policies were updated in response to public opinion, thereby advancing policy improvement as a two-way process.

[^github]: https://github.com/takahiroanno2024/election2024/pull/221

[^note]: [2024 Tokyo Governor Election] Summary of Updates to Takahiro Anno’s Governor Election Platform (6/22–6/29) | Takahiro Anno Staff @ Team Mirai [Official] https://note.com/annotakahiro24/n/n2e69a38b41f3

[^note2]: https://note.com/jujunjun110/n/n2f855e5a6cd8?magazine_key=m3060242bb916

This process put into practice, in the setting of an election, the broad listening ideal of “building policy by listening to voters’ voices.”

## Visualizing Opinion Groups with My Number Voting and Polis

My Number Voting was not operated by the Anno campaign; it was an electronic voting experiment conducted by POCKETSIGN. On his official website[^directvote], Mr. Anno introduced it as follows: “Check the views of each Tokyo resident individually—we will also refer to POCKETSIGN’s My Number Voting! *My Number Voting is an external service operated by Pocket Sign Inc.”

[^directvote]: https://takahiroanno.com/directvote

When people hear that social media posts are being used as policy input, concerns often arise: “What if one person posts multiple times?” “What if someone creates multiple accounts and pretends to be multiple people?” “What if foreign operatives impersonate citizens?” Japan already has a technology capable of addressing these concerns: the My Number system, Japan’s national ID system.

In materials submitted by Mr. Anno to the second meeting of the Digital Administrative and Fiscal Reform Strategy Team (March 25, 2025), under the Cabinet Secretariat’s Digital Administrative and Fiscal Reform Council, he proposed combining Talk to the City with smartphone-based My Number functionality to enable one-person-one-vote participation, and also suggested that My Number card authentication could be used as a countermeasure against cognitive warfare on deliberative platforms.

On June 24, 2024, through cooperation between the Digital Agency and Apple, My Number card functionality became available on the iPhone. This made it possible to use the electronic certificates on the My Number card through integration with Apple’s systems, eliminating the need to physically scan the card when logging into services such as Mynaportal, Japan’s online government services portal. The author uses it as well, and it is extremely convenient.

Against this backdrop of steady progress in the My Number system, My Number Voting was released on June 21, 2024[^mynavote]. It allowed people to express direct support or opposition not to politicians but to policies. In this experiment, users answered questions such as, “Should redevelopment proceed in Meiji Jingu Gaien, the outer garden of Meiji Shrine, by rebuilding Jingu Stadium and Chichibunomiya Rugby Stadium and constructing new high-rise buildings?” and “Given the current concentration of population in Tokyo, should this population distribution continue to be maintained in the future?” The experiment demonstrated that it was possible both to restrict participation to one vote per person and to obtain analysis results limited to Tokyo residents.

[^minavote]: https://prtimes.jp/main/html/rd/p/000000037.000110743.html

The Anno team received anonymized voting data. Because this data consisted of support / neutral / oppose responses, as in Polis, it was possible to conduct similar analysis and extract and visualize opinion groups[^polis-analysis].

### Looking Back on My Number Voting

I am writing this manuscript in January 2026. As I looked back on the events of June 2024, which passed in a blur, I remembered My Number Voting. Reflecting on it now, I felt it was definitely something that should be written about. And yet I had forgotten the experiment so thoroughly that it had slipped off the original list of topics I planned to include in this manuscript.

Why had I forgotten it? Probably because we hardly talked about it at all after the election. During the campaign, it had been prominently featured on the official website with a large image, so why was it not discussed afterward? Presumably because it did not lead to especially good results. Thinking through why that happened should be useful for the future development of broad listening.

From the author’s own experience of actually using My Number Voting, it felt sufficiently interesting even at that time. Of course, given the technical level at the time, users had to scan a physical My Number card with a smartphone when starting to use it, and that raised the barrier to participation. But that was not the main problem.

The biggest problem was organizational structure. POCKETSIGN was conducting the social experiment independently of the Anno campaign[^POCKETSIGN], and the campaign was not involved in planning it. As a result, the questions in My Number Voting were not the priority items in Mr. Anno’s policy agenda, but rather issues that mass media at the time had framed as key points of contention. Consequently, the output suggested that the main divisions were “Meiji Jingu Gaien redevelopment” and “Tokyo overconcentration.” Even if such results were obtained late in the campaign, they were of little practical use to Mr. Anno. This is also related to the problems with conventional surveys discussed in Chapter 2. What you ask has a major impact on what you get. If you do not determine the questions by working backward from the information you actually want, it is difficult to obtain useful results.

[^POCKETSIGN]:
“This experiment is not an official Tokyo Metropolitan Government initiative, but an independent technical experiment planned by our company (a private enterprise). Therefore, it has no connection whatsoever with the actual election, the election administration commission, or any candidates.” / “Data obtained in this experiment may be provided, in anonymized statistical form, to politicians, government officials, researchers, and others for research, analysis, and other purposes.”
https://pocketsign.co.jp/news/46

If, hypothetically, a My Number Voting–type service had been operated not by an independent private company but by an organization directly under Anno’s control, what would have happened? There would likely have been more questions about technology rather than questions about Meiji Jingu. It also would likely have operated by “running the cycle quickly”: closing the vote promptly, publishing the analysis results, and then using the attention generated by that to conduct a second round on a larger scale. Based on the information obtained from the previous vote, it would likely have asked more detailed follow-up questions in areas judged worth probing further.

Broad listening is a means of gathering information for an organization, and what kinds of information are useful differs from one organization to another. What matters is to run the cycle quickly, learn rapidly from unknown information, and evolve in the direction of obtaining more useful information. If part of the cycle belongs to another organization, it cannot be run quickly. This shows that, for effective use, it is important to operate broad listening tools oneself and build a structure that can use them swiftly.

## Technical Challenges and Responses

In actually using Talk to the City in an election, we encountered several technical challenges.

**Building a preview server**:

This was not something I noticed when experimenting alone, but when using Talk to the City within a team, review before publication becomes necessary. To make “Person A creates a report” and “Person B reviews it” possible, Person A needs a way to share the results with Person B without publishing them to the entire world.

At first, we shared screenshots or printed the browser view to PDF, but because “being able to interact with it in the browser” accounted for most of its value, we ultimately built a hosting environment for internal team sharing. This experience later became one of the triggers for developing “Public Listening AI.”

**Japanese localization**:

Because TTTC was designed with English as the priority language, using it in Japanese required some ingenuity.
There was no function to make Japanese the default, so if a report were shared with Tokyo residents, it would likely first appear in English, requiring users to switch to Japanese with the language toggle button in the upper right. We worried that this would be cumbersome to explain and that many people would leave upon seeing an English screen, so we quickly created a Japanese-only version and used that instead. This too later became one of the triggers for developing “Public Listening AI.”

**Hallucination countermeasures**:

When people hear “hallucination,” many simply think it means “the AI makes mistakes,” but what actually happened was a bit more complicated.

One of the candidates in the Tokyo governor election was Shinji Ishimaru. Many users referred to him as “Ishimaru-san,” while one user posted the incorrect spelling “Shinji Ishimaru” using the wrong character for “Shin.” Seeing these posts, the AI interpreted “Ishimaru-san” as referring to that incorrectly written name and used the mistaken spelling in the summary of an opinion group. A reviewer of the generated report noticed the spelling error, and in the process of investigating the cause, this situation came to light.

In other words, what was needed was not merely the already difficult task of “preventing the AI from outputting incorrect information,” but the even harder task of “preventing the AI from outputting incorrect information when humans themselves input incorrect information into the AI.” At the technical level of summer 2024, this was not an easy problem to solve[^2026]. The options were either to tolerate this kind of error and publish reports as they were, or to have humans review and correct each report. In the 2024 Tokyo governor election, we chose human review. As a result, because the Anno campaign had limited human resources, only a small portion of the more than 30 generated reports were made public.

[^2026]: As of January 2026, when I am writing this, I believe that “if the final report is reviewed by a sufficiently capable model given sufficient contextual information, it should be possible to detect and correct such errors.”

## Election Results and What Comes Next

Candidate Takahiro Anno received 154,638 votes (2.3% of the vote)[^tochiji-result]. This was the highest vote total ever for a candidate in his 30s, and also an exceptional result for a first-time candidate without the backing of a party organization.

The practice of broad listening in this election became a pioneering example demonstrating the potential of digital democracy in Japanese elections. Although challenges remain, it was meaningful that the process of “listening to voters’ voices and building policy” was actually put into motion within the short time frame of an election campaign.

This result—more than 2% of the vote—also helped encourage Mr. Anno to launch the political party Team Mirai (meaning “Team Future”) in 2025[^team-mirai]. The ideals of “digital democracy” and the methods of broad listening cultivated during the governor election have continued to occupy a central place in his subsequent political activities.

Broad listening, moreover, is not a technology limited to elections. It can be applied in many settings, including analyzing customer feedback in private companies, gathering residents’ opinions in local governments, and facilitating dialogue with supporters in NPOs. This point will be discussed in more detail in the latter half of this book.

[^mitou-jr]: Mitou Junior, Mentor Introduction. https://jr.mitou.org/mentors/
[^aoi-tttc]: AI Objectives Institute, “Talk to the City.” https://ai.objectives.institute/talk-to-the-city-1
[^github-manifesto]: Takahiro Anno Tokyo Governor Election 2024 Platform Repository. https://github.com/takahiroanno2024/election2024
[^polis-analysis]: Hirokazu Nishio, “The Future Digital Technology Brings to Democracy,” note, July 2024. https://note.com/nishiohirokazu/n/n799c9818b33d
[^tochiji-result]: Tokyo Metropolitan Election Administration Commission, “Tokyo Governor Election (held July 7, 2024) Vote Count Results.” https://www.senkyo.metro.tokyo.lg.jp/election/tochiji-all/tochiji-sokuhou2024/result
[^team-mirai]: ITmedia, “AI Engineer Takahiro Anno Launches New Party ‘Team Mirai,’” May 8, 2025. https://www.itmedia.co.jp/aiplus/articles/2505/08/news146.html
