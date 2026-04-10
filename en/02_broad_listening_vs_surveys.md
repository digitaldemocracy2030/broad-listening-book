# Chapter 2: The Difference Between Broad Listening and Surveys, and the Shift from Quantitative to Qualitative Analysis

Author: @tokoroten


## The Difference Between Closed Questions and Open Questions

As discussed in the previous chapter, broad listening is an attempt to overcome the trade-off that "if you pursue scale, you sacrifice depth of opinion; if you pursue depth, scale becomes limited." In this chapter, we will unpack why this trade-off arises by examining the difference between closed questions and open questions.

This chapter focuses on **broad listening as a tool**—that is, the use of AI to structure and visualize opinions. Chapter 3 explains how expanded deliberation works in practice when this tool is incorporated.

The defining feature of broad listening is that it makes it possible to analyze and investigate open questions.

Traditional surveys typically ask something like, "What do you think about X?" and require respondents to choose from a limited set of options such as "strongly agree," "agree," "don't know," "disagree," and "strongly disagree." This type of question is called a closed question.

Broad listening, by contrast, can handle not only the closed questions described above, but also free-response prompts such as "Please share your views on X." These free responses effectively allow answers from an unlimited range of possibilities. This type of question is called an open question.

Open and closed questions each have their own strengths and weaknesses. The table below summarizes the differences.

|  | Open Questions | Closed Questions |
|:--|:--|:--|
| Question format | Free-text response | Multiple choice |
| Agenda setting | Raised freely by respondents | Set in advance by the survey designer |
| Burden on respondents | High | Low |
| Burden on analysts | High | Low |
| Ability to reproduce public opinion | Low | High |
| Type of analysis | Qualitative analysis (what views are expressed) | Quantitative analysis (what views are expressed and how prevalent they are) |

Because closed questions place a low burden on respondents, they make it easier to collect large numbers of responses, and they can also be analyzed efficiently. Since respondents only need to choose from a list of options, they can answer in a matter of seconds, making drop-off less likely. On the analysis side, responses are obtained as numerical data, so tabulation and statistical processing can be automated. In addition, combining responses with demographic attributes such as age, gender, occupation, place of residence, and educational background improves the reproducibility of public opinion.

For example, this makes it possible to analyze questions such as, "What percentage of women in their 30s support this policy?" or "Are there differences in opinion trends between urban and rural areas?" With appropriate sampling, it is even possible to estimate broader social trends from the responses of survey participants. In other words, quantitative analysis can show how many people in society as a whole hold a given view.

Open questions, on the other hand, pose a major challenge because they place an extremely high burden on respondents. Putting one's thoughts into writing takes time and effort, so many people drop out before completing their response. More fundamentally, people often do not have a clearly formed opinion on a given topic, in which case they can only answer, "I don't know," which lowers the rate of useful responses. In addition, extracting insight from responses written in natural language places a heavy burden on analysts, and with conventional technology it was difficult to analyze large volumes of open-ended responses.

That said, open questions also have major advantages. Because respondents are not constrained by predefined options and can express themselves in their own words, perspectives and issues that the survey designer had not anticipated can emerge. In other words, open questions are not well suited to producing quantitative results such as "how many people think this way," but they are highly effective for producing qualitative results such as "what kinds of opinions exist."


## Power to the Public: Free-Text Responses Open up Agenda Setting

The significance of being able to handle open questions goes beyond the depth of the responses alone. Audrey Tang, who served as Taiwan's Digital Minister, gave the following answer to the question, "How does the collective intelligence of the internet affect politics?"

> "If agenda-setting power is opened up to the public, we can show citizens what they agree and disagree on. The era in which government alone determines the agenda is over."
https://web.archive.org/web/20201011031722/https://www.vodafone-institut.de/aiandi/democracy-needs-to-evolve-into-a-real-time-system/

"Opening up agenda-setting power to the public"—this is the essential significance of broad listening.

As noted above, traditional surveys have been designed primarily around closed questions because they place a low burden on both respondents and analysts and allow for quantitative analysis. But closed questions have an inherent limitation: responses can only be collected from among the options prepared in advance by the survey designer.

For example, if the question is "Do you support or oppose this policy?", it cannot capture responses such as "I support it conditionally," "A different policy would be better," or "The framing of the issue itself is wrong." At the moment the answer choices are designed, responses are already constrained by the limits of the survey designer's imagination.

There is also the problem that the wording of the question and the order of the answer choices can influence the pattern of responses. Whether through lack of skill or by design, the results can change depending on how the question is framed.

Put differently, these problems reflect the fact that the power to decide "what becomes an issue" and "what options are presented" has long been monopolized by governments and survey designers. In traditional opinion polling, it was always the government or the mass media that decided what questions would be asked, while citizens could do no more than choose from the options they were given. Even if people had concerns or difficulties that genuinely mattered to them, they had no way to voice them unless those concerns appeared among the choices.

Broad listening changes this situation by making it possible to handle open questions at scale. Through free-text responses, citizens themselves can raise issues, and the themes of discussion can take shape starting from voices saying, "I want to talk about this problem," or "This is what I'm struggling with." By extracting common themes and unexpected points of contention from those responses, it has become possible to identify issues that survey designers had overlooked.

The real appeal of broad listening lies in discoveries such as, "I didn't realize anyone was concerned about that." By picking up voices that those conducting the survey never would have anticipated, it becomes possible to develop policies that respond to the concerns of a wider range of people. This is exactly what Tang means by "reflect[ing] back to people what they agree or disagree on."


## Broad Listening Is Qualitative Analysis, Not Quantitative Analysis

In the table at the beginning of this chapter, I touched on "the reproducibility of public opinion" and "the type of analysis." These two points are extremely important for understanding broad listening, so I will explain them in more detail here.

### What Is Public Opinion?

First, let us define the term "public opinion." The phrase is used in many different ways, but in this book, **public opinion means the distribution of views across society as a whole, as captured through representative data**.

So what does "representative" mean? Representativeness is the property that allows us to correctly estimate broader social trends from the responses of survey participants (the sample). For example, it is not realistic to ask all 120 million people in Japan, "Do you support this policy?" Instead, a survey is conducted using a sample of 1,000 or 2,000 people. If that sample is a miniature version of society as a whole—if its age distribution, gender ratio, regional distribution, and so on match those of the broader population—then the sample's responses can be said to reflect overall social trends. That is what it means for data to be representative.

The basic method for ensuring representativeness is **random sampling**. People are selected without bias, like drawing lots, from sources such as the resident registry or lists of phone numbers. This prevents the sample from being skewed toward people with particular attributes or opinions and makes it possible to obtain a sample that resembles society as a whole in reduced form.

With data sampled without bias through random sampling, it becomes possible to reproduce the overall distribution in forms such as "X% support, Y% oppose." This is what it means to measure public opinion. The purpose of quantitative analysis using closed questions is precisely to understand this public opinion. That is why traditional opinion polls, market research, and surveys have been designed primarily around closed questions.

### The Problem of the Noisy Minority

The defining feature of broad listening is that it can analyze natural-language responses to open questions at scale. Once natural language became usable as input, it became possible to collect data from a wide variety of existing communication channels and handle them in an integrated way. Input sources for broad listening include posts on X (formerly Twitter), Facebook, and Bluesky, YouTube comments, LINE, email, transcribed phone calls, website submissions, responses to surveys conducted by political parties, and input to AI-based listening systems. Because opinions can be collected from channels people already use in everyday life, it has become possible to capture naturally expressed voices without requiring the special act of "answering a survey."

However, these information channels have one important characteristic: they tend to attract the voices of people who have something clear they want to say. Writing out an opinion in free text takes time and effort. People who overcome that hurdle and post their views tend to be those with strong interest in the topic or clearly formed opinions. Conversely, people with no particular opinion have little motivation to post, so their views are not captured.

And in reality, most people do not have clearly formed opinions on any given topic. For example, if someone asked me, as a Tokyo resident, "What do you think about the Isahaya Bay land reclamation project (a long-running administrative and legal dispute involving ecosystem conservation and agricultural development)?", I could only answer, "I don't really know." This kind of "silent majority" does not appear in broad listening.

By contrast, in a closed-question survey, if randomly sampled respondents are asked to choose among "support," "oppose," and "don't know," then the issue can be understood quantitatively in terms such as "X% support, Y% oppose, and Z% don't know." "Don't know" is itself perfectly valid data. And because such data are obtained through random sampling, they have the representativeness needed to reproduce the distribution of views across society as a whole, making them something we can properly call public opinion.

In broad listening focused on natural language, only the views of people who have opinions are collected, so the existence of this "majority without an opinion" disappears from view. In other words, the voices gathered through broad listening tend to be skewed toward the **noisy minority**—a small group with loud voices. This is not so much a flaw as an inevitable consequence of broad listening being a form of qualitative analysis.

According to a 2019 Pew Research Center study, 73% of political tweets on Twitter (now X) were produced by just 6% of users[^1]. The "public opinion" we see on social media is shaped by a very small number of highly active posters.

![The noisy minority and the actual distribution of people](images/02_noisy_minority_en.png)

The figure above illustrates this phenomenon schematically. The horizontal axis represents the strength of opinion (the degree of positivity or negativity). The vertical axis of the red curve shows the distribution of the **actual number of people**, with the vast majority occupying a neutral position. By contrast, the vertical axis of the light blue curve shows the **volume of opinions**, where people with more extreme views post more frequently. What broad listening captures is the shape of the light blue curve, which differs greatly from the actual distribution of people.

### When to Use Qualitative Analysis and Quantitative Analysis

Qualitative analysis is a method used when "we do not yet know what should even be measured." Before creating answer choices, the first step is to explore what people are thinking and what issues exist. It is an exploratory form of inquiry, not a way to measure quantitative indicators such as "what percentage of people hold each opinion." The value produced by broad listening lies not in precise ratios, but in the discovery itself: "I hadn't considered that perspective." Its real strength is in surfacing unexpected issues, overlooked problems, and minority voices that may be small in number but important.

So how should broad listening be used?

The correct way to use it is to position broad listening in the **hypothesis-generation** phase. First, broad listening is used to understand what kinds of opinions exist and what issues are present. Then, based on those findings, survey questions are designed and quantitative analysis using random sampling is conducted to test how many people hold each view. Only with this two-step approach can qualitative depth and quantitative representativeness be combined.

For example, suppose broad listening on a certain policy reveals three issues: "the burden on child-rearing households," "transportation for older adults," and "environmental impact." At that stage, we still do not know which of these issues is shared by the largest number of citizens. So the next step is to conduct a randomly sampled survey using those three as answer choices and ask, "Which issue do you consider most important?" That is how quantitative validation is obtained.

Broad listening is not a method for measuring public opinion. It is a method for discovering the diverse perspectives that exist within public opinion. One sometimes hears the criticism, "Broad listening is useless because its results are not public opinion," but this confuses the purpose. Measuring public opinion and discovering new perspectives are fundamentally different goals.

### The Misconception That Broad Listening Automatically Produces Lots of Responses

At this point, it is worth clearing up one common misunderstanding: the mistaken belief that "if you do broad listening, lots of citizen opinions will naturally come in." This is clearly wrong.

Because free-response surveys have a higher barrier to participation than ordinary multiple-choice surveys, local governments that simply run a survey in the usual way will not collect very many responses. The case studies introduced in this book gathered large numbers of opinions, but that was not because of the power of broad listening itself.

In the 2024 Tokyo governor election, Takahiro Anno collected many opinions because he, as a high-profile AI engineer, conducted broad listening. In the Tokyo Metropolitan Government case introduced in Chapter 5, around 27,000 opinions were collected, but that too was the result of large-scale public outreach by the Tokyo Metropolitan Government, active calls for citizen participation, and the collection of opinions from social media.

Broad listening is only a tool for **efficiently analyzing and visualizing opinions that have already been collected**. It is not a tool for **collecting** opinions. If broad listening is to be implemented, it is necessary to think through how responses will actually be gathered as well—including outreach methods, ways to encourage participation, and coordination with existing mechanisms for civic participation.

That said, there are ways to gather opinions without running your own survey. One option is to collect and analyze conversations that are already taking place online, such as on X (formerly Twitter), Facebook, Bluesky, message boards, and comment sections on news sites. By moving away from the idea of "collecting opinions through a survey" and instead adopting the perspective of "discovering opinions that already exist," even organizations with limited outreach capacity or organizational resources can secure material for analysis.

However, collecting from social media comes with selection bias. For example, if you search for "childcare support," you may miss people discussing the same underlying issue using terms such as "daycare," "parental leave," or "solo parenting." But if the purpose of broad listening is to gain insight, this kind of bias is not necessarily fatal. As long as the bias is recognized, it can be mitigated by combining multiple keywords and diverse channels.


## LLMs Made Large-Scale Analysis of Open Questions Possible

So how were open questions analyzed before this? Word clouds were widely used as a way to visualize free-text responses, but a word cloud simply breaks text into words, counts how often each appears, and displays the most frequent words in larger type. It may show "what topics exist," but it does not show "what opinions exist." A word cloud was never really an analysis of open questions; it was merely a visualization of frequency.

What changed this situation was the large language model (LLM) discussed in Chapter 1.

Anyone who has used ChatGPT has probably experienced its capabilities firsthand. For example, try entering the following prompt into ChatGPT: "Express the theme of this statement in a single word: 'We're both working, and still can't get our child into daycare!'" It will respond something like this:

> **"Daycare waiting lists"**

The original statement does not contain the phrase "daycare waiting lists" at all. Yet the LLM can understand the meaning of the statement and identify the underlying social issue. This is essentially the same kind of ability used to solve reading-comprehension questions.

The important point here is that ChatGPT is not merely "a service for chatting with AI." The essence of an LLM is that when you input text written by a human, it can understand the content, summarize it, classify it, evaluate it, and return the result as a **component that can be called from a program (an API)**.

Traditionally, programs could handle strings of characters, but they could not understand their meaning. Reading free-text responses and judging what they said was work that only humans could do. But once LLMs could be incorporated as components in software, that barrier of semantic understanding could be crossed. Programs could now answer questions such as "What is this opinion about?", "Is it supportive or opposed?", and "What kind of emotion does it contain?"

What this means is that instead of a human manually asking ChatGPT one question at a time, a program can now repeat the process automatically thousands or tens of thousands of times. The burden on analysts has dropped dramatically. It is also possible to compare multiple opinions to find commonalities, classify opinions by category, and quantify the strength of sentiment. The technical details are explained in Chapter 12, and implementation methods in Chapter 13.

The Public Listening AI discussed throughout this book, as well as its prototype Talk to the City, combine these LLM capabilities across multiple steps. First, themes and issues are extracted from each of thousands of free-text responses. Next, similar opinions are grouped together through clustering. Finally, summaries are generated for each group. By automating this sequence of processes in software, work equivalent to reading and organizing each response one by one can be completed in a short time.

In this way, the arrival of LLMs transformed open questions from something that was "easy to collect but hard to analyze" into something that, "once collected, can also be analyzed." This opened a technological path to overcoming the "trade-off between scale and depth" described in Chapter 1.

## The Trade-off Between Scale and Cost

In addition to the trade-off between scale and depth, **cost** is another important axis. In Japan, the national census is conducted only once every five years. A full-population survey with complete representativeness requires a budget on the order of 70 billion yen (roughly US$470 million in 2025 terms). Even at that cost, the questionnaire is limited to just 19 items and does not include questions asking for opinions. Ordinary opinion polls reduce costs through sampling, but once open questions are included, someone must read and classify each free-text response one by one, causing costs to rise sharply.

In other words, traditional research methods have been constrained by a three-way trade-off among **scale, depth, and cost**. If you want scale and representativeness, the cost becomes enormous. If you want to keep costs down, you have to sacrifice either scale or depth. Survey design centered on closed questions has effectively chosen to "sacrifice depth in order to reduce cost."

The arrival of LLMs changed this three-way structure. As Chapter 13 will estimate in detail, the cost of analyzing 10,000 free-text responses with Public Listening AI is about 500 yen (roughly US$3 in 2025 terms). Work that previously would have required dozens of person-days can now be completed for about the price of a coin-operated snack. Compared with the 70 billion yen (roughly US$470 million in 2025 terms) cost of the census or the millions of yen (roughly tens of thousands of US dollars in 2025 terms) required for ordinary opinion polling, the difference is not just a matter of scale—it is on an entirely different order of magnitude. Once the cost of analyzing open questions effectively approaches zero, there is no longer any need to "sacrifice depth in order to reduce cost."

## "Polis": Broad Listening Before LLMs

Even before the arrival of LLMs, there were systems capable of aggregating opinions at scale. One such system was Polis, which was used in Taiwan's vTaiwan process.

MEMO: Insert a screenshot of Polis around here

When you open the Polis interface, the first thing you see is a single opinion posted by another participant. For example, suppose the statement shown is, "Ride-sharing services like Uber should be legalized." Participants respond by choosing one of three options: "agree," "disagree," or "pass." Once they answer, the next opinion appears, and they respond again with the same three choices. This process repeats.

At the same time, participants can also post their own opinions freely. They can raise new issues such as, "The taxi industry's vested interests should not be protected," or "Services with unclear safety standards should not be allowed." These submitted opinions are then shown on other participants' screens, where they receive the same agree/disagree/pass votes.

What made Polis effective was this hybrid design combining open questions (free opinion submission) and closed questions (three-choice voting). Rather than directly analyzing the content of free-text responses, it analyzes the **voting patterns** on those responses.

More specifically, participants with similar voting tendencies are grouped together and visualized on a two-dimensional scatter plot. This makes it possible to see at a glance things like, "This opinion sharply divides supporters and opponents," or "This opinion is unexpectedly supported by both groups." It reveals not only points of conflict, but also areas of agreement.

One distinctive feature of Polis is that as participants continue voting, the icon representing them moves around on the scatter plot. This provides a real-time visualization of what opinion group they belong to and makes it visibly clear that "there are people who think differently from me." As a result, even people who are usually exposed only to like-minded views within social media filter bubbles can directly sense that society contains a diversity of opinions.

In Taiwan, Polis was used in 2015 in discussions over Uber regulation, where it revealed that two opposing groups—the taxi industry and ride-share supporters—could actually agree on one point: mandatory insurance coverage. Policy then began to take shape starting from that point of agreement.

Within the technological constraints of the time, Polis cleverly preserved the advantages of open questions—allowing citizens to raise issues freely—while reducing the burden of analysis by converting responses into numerical voting data. With the arrival of LLMs, it became possible to analyze free-text responses directly without such workarounds, but the design philosophy of Polis was still an important step in the history of broad listening.


## Misunderstandings Caused by the "Science-Like" Appearance of Broad Listening

The results of broad listening are often presented with a "science-like" appearance. Opinions are automatically clustered, visualized on a scatter plot, and each cluster is labeled with a count. This visual style can create a kind of mysterious persuasive power.

TODO: Example of a clustering scatter plot  
TODO: Examples of tools other than Public Listening AI, such as Polis?

When people look at a scatter plot, they unconsciously assume that the vertical and horizontal axes must have some meaning. In an ordinary graph, the axes do have clear meanings—for example, time on the horizontal axis and sales on the vertical axis. But in a broad-listening scatter plot, the positions are nothing more than the result of compressing high-dimensional opinion vectors into two dimensions so that humans can view them more easily. Interpretations such as "opinions in the upper right are good" or "opinions in the lower left are bad" are completely mistaken. The only thing the positions indicate is the **relative similarity between opinions**; the vertical and horizontal axes themselves have no meaning at all.

When numbers and graphs are present, people are prone to assume they are looking at quantitative data. If they are told, "Cluster A contains 500 opinions and Cluster B contains 200," they naturally want to interpret that as "Opinion A is 2.5 times more common than Opinion B." But that is not what it means. These are merely the internal breakdown of already biased participation data. The respondents are people who were highly motivated to voice opinions in A or B. That is entirely different from the question of what proportion of society as a whole holds those views.

In fact, people who are most accustomed to reading quantitative analysis reports are often the most likely to fall into this trap, because they try to interpret unfamiliar outputs using familiar frameworks. But the results of broad listening must not be read in the same way as traditional opinion polls. What broad listening shows is that "there are people who hold these kinds of views," not "this is how much support these views have across society as a whole." The size of a cluster represents the **number of submitted opinions**, not the **share of people in society who hold that opinion**.


## Summary

This chapter explained the difference between broad listening and surveys.

Whereas traditional surveys have been designed primarily around closed questions (multiple choice), broad listening is distinguished by its ability to handle open questions (free-text responses) at scale. The essential significance of handling open questions is that it opens up agenda-setting power to citizens and makes it possible to discover issues that survey designers had not anticipated.

Broad listening is a form of qualitative analysis, not quantitative analysis that measures "what percentage of people think this way." Its value lies in discoveries such as, "I hadn't considered that perspective." Because the opinions collected are biased toward "people who chose to speak up," broad listening should be positioned as a tool for **hypothesis generation**, with the issues it uncovers then tested through quantitative analysis when necessary.

What made this kind of analysis possible was the development of LLMs after 2022. Earlier tools such as word clouds could only visualize word frequency, but the arrival of LLMs made it possible to automatically extract issues from thousands of free-text responses.

---

[^1]: Pew Research Center, "National Politics on Twitter: Small Share of U.S. Adults Produce Majority of Tweets," October 2019 https://www.pewresearch.org/data-labs/2019/10/23/national-politics-on-twitter-small-share-of-u-s-adults-produce-majority-of-tweets/
