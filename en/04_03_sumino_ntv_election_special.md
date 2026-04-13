## 4.3 Nippon TV Election Special: Using Broad Listening in Television News Coverage

Written by: @nasuka

English translation by Luke Closs

The project introduced in this section was carried out through collaboration between Nippon TV and Anno's team. Anno's team consisted of Anno himself and members who had supported his Tokyo gubernatorial election campaign, and handled the design and execution of the TTTC-based analysis. The author participated in this project as an engineer on Anno's team. This section is also based on an interview conducted with the Nippon TV staff member responsible for the project at the time.

This case showed two things: that broad listening succeeds or fails largely on the design of data collection, and that bringing social media analysis to viewers also requires careful thought about how the results are presented. The sections below explain these lessons through the details of the project.

### 4.3.1 Project Background

In the July 2024 Tokyo gubernatorial election, candidates without traditional support bases, such as Shinji Ishimaru, made major gains. Nippon TV took from that result the lesson that online voices could influence actual voting behavior, and began considering ways to analyze online discussion and incorporate it into election coverage.

In that context, Nippon TV focused on Anno's use of broad listening in the Tokyo gubernatorial election, approached Anno's team about using it in election coverage, and the project began.

### 4.3.2 Use in Election Coverage

In the October 2024 House of Representatives election, Nippon TV carried out election coverage using broad listening. It was a pioneering case of incorporating broad listening in earnest into television election coverage.

During the election period, a YouTube program titled *Who Should We Vote For? A Meeting to Create the 2024 House of Representatives Election Through Everyone's Voices* was streamed in six episodes. The format involved collecting opinions about each political party from X (formerly Twitter), analyzing them with TTTC, and then inviting lawmakers from each party to discuss the findings. The reports visualized which topics were attracting attention on X, and Takahiro Anno also participated as a commentator.

Then, on October 27, election day, a terrestrial TV election special titled *zero Election 2024: How Japan Will Change* was broadcast. Starting at 8:00 p.m., when voting closed, the team collected posts on X related to the House of Representatives election for about 25 minutes. Those posts were analyzed with TTTC to produce a report. During the program, Anno explained the report and delivered voters' reactions immediately after the polls closed in a near-real-time format.

Each report that was produced was not only used on the program, but was also made publicly available online.[^1]

### 4.3.3 Analyses Conducted

In this initiative, posts on X related to the House of Representatives election were analyzed with TTTC, and reports were created for use in the program. However, simply collecting election-related posts indiscriminately would have made it hard to see what voters regarded as the issues, because political maneuvering and everyday posts would have buried the signal. To focus on what kinds of opinions were being expressed about policies and political parties, the following four types of analysis were designed.

1. **Analysis of opinions on the House of Representatives election as a whole**: to provide an overview of voter interests and topics related to the election overall
2. **Analysis of opinions on policy**: to understand views on specific policy areas such as the economy, social security, and foreign affairs
3. **Analysis of opinions on each political party**: to understand how each party was being discussed
4. **Analysis of opinions immediately after the polls closed (immediately after vote counting began)**: to capture voters' reactions immediately after counting began

The data analyzed was collected from X, and a separate query was prepared for each type of analysis. The analysis was conducted using a custom-modified version of TTTC. Analyses 1 and 2 were carried out in three periods: before the official campaign announcement, the first half of the campaign, and the second half of the campaign. Analysis 3 was conducted when lawmakers from each party were invited onto the YouTube program. Analysis 4 focused on collecting and analyzing data intensively during the short period immediately after the polls closed.

The workflow from data collection to report publication was as follows. It was a collaborative process in which Anno's team and Nippon TV checked report quality as the work proceeded.

![Workflow from data collection to report publication](images/04_03_ntv_workflow.png)

The following is an example of the analysis results. TTTC places posts from X on a scatter plot based on semantic similarity and displays them color-coded by cluster. Each point corresponds to one opinion, and similar opinions are placed closer together. The vertical and horizontal axes themselves do not have any specific meaning; they are the result of compressing high-dimensional data into two dimensions.

![Scatter plot of analysis results produced by TTTC. Opinions are displayed in different colors by cluster](images/ntv_scatter.png)

Analysis with TTTC does not automatically produce useful results just because data has been fed into it. Design decisions such as what queries to use for collecting data and how many clusters to create have a major effect on the quality of the analysis. The following sections introduce the design choices encountered in practice and the lessons drawn from them.

#### Adjusting the Number of Clusters

Because the number of clusters changes the granularity of the analysis, this required trial and error. The team compared outputs with cluster counts between 5 and 20, and adjusted the number for each report according to the nature and volume of the data. For each report, it was necessary to find a level of granularity that was neither so coarse that clustering collapsed together, nor so fine that meaningless small groups proliferated.

#### Differences in Results Depending on the Data Collected

Even when analyzing the same period, the picture that emerges can change greatly depending on the query used to collect the data. Let us compare the results for "the House of Representatives election as a whole" and "policy" during the pre-announcement period (October 6-12).

In the **analysis of the House of Representatives election as a whole**[^2], posts related broadly to the election were collected.

Clusters emerged around topics such as "interest in the consumption tax and the slush-fund scandal," "opposition cooperation and election strategy," and "personal election-related experiences and effects on daily life." The discussion centered on "the election as an event in itself," including criticism of the slush-fund scandal, debate over opposition candidate coordination, declarations of intent to vote, and complaints about the noise from campaign vehicles. The scatter plot shown above (images/ntv_scatter.png) is the result of this analysis of the House of Representatives election as a whole.

In the **policy analysis**[^3], posts were collected using policy-related keywords.

![Scatter plot of policy analysis](images/04_03_ntv_policy_scatter.png)

Clusters emerged around topics such as "dissatisfaction with and criticism of the ruling parties," "concerns about tax increases and dissatisfaction with economic policy," "concerns about national security and immigration policy," and "debate over reducing or abolishing the consumption tax." Opinions were grouped by specific policy themes such as cutting or abolishing the consumption tax, concerns about immigration policy, raising the minimum wage, allowing married couples to keep separate surnames, and the My Number health insurance card (My Number, Japan's national ID system). This made it much clearer what voters saw as the key issues.

**Principle: Broad Listening Is Determined by the Question and the Data Design**

The results of an analysis, and the insights that can be drawn from it, depend on the data collected. For example, if the goal is to analyze opinions on policy, collecting election-related data broadly may not produce the desired results. That is because data on the election as a whole contains many topics related to political maneuvering and campaign activity, which can bury discussion of policy.

The lesson from this experience is the importance of first clearly defining the question, "What do we want to find out?", and then designing data collection so that it gathers data capable of answering that question. As the phrase "garbage in, garbage out" suggests, the quality of the analysis is determined not by the algorithm, but by the design of data collection. The design of the query becomes the way the world is sliced.

### 4.3.4 Analysis Immediately After the Polls Closed: A Three-and-a-Half-Hour Battle

Unlike analyses that could be prepared in advance (three periods across the pre-campaign period, the first half, and the second half), the analysis immediately after the polls closed relied on data that did not exist until election day itself.

After voting closed at 8:00 p.m., posts on X were collected, the analysis was run, the results were reviewed, and the findings were finalized into a report. From the start of data collection to its use on air, all of this had to be completed in about three and a half hours.

In practice, the schedule was tightly choreographed. After voting closed at 8:00 p.m., the team collected 25 minutes' worth of posts from X. This was the point judged to provide enough data while still leaving barely enough time for the downstream work. After collection, the report was output within one hour. Then the team checked the clustering and fact-checked the summary text. Next, the results were shared with the program's production staff, decisions were made about which analysis results to use on air, a script was written, and the presenters were briefed. By the time the broad listening segment aired around 11:30 p.m., the team had analyzed a dataset of 10,091 items.

From data collection through fact-checking, production decisions, scriptwriting, and presenter briefing, everything had to happen in three and a half hours. It was by no means a relaxed schedule, but producing an equivalent analytical report from scratch by human labor alone would not have been realistic. The operation was possible because broad listening technology existed.

### 4.3.5 Improvements to TTTC for Broadcast Use

To use TTTC in news reporting, major improvements were required in both functionality and UI. The main enhancements that were implemented are introduced below.

First, practical features needed for live broadcasting were added. These included a favorites feature that allowed noteworthy comments to be bookmarked and called up quickly during the program, and a redesigned mobile UI for the many viewers expected to access the reports from smartphones.

![Detailed cluster view. You can see the cluster analysis results, representative comments, and the favorites feature](images/04_03_ntv_cluster_example_en.png)

**Improved Accuracy in Extracting Representative Comments**

TTTC has a feature that displays comments representing each cluster. Previously, however, comments were selected at random from within a cluster, which sometimes resulted in opinions being highlighted that had little connection to the cluster title.

To address this problem, the system was improved to use an LLM to extract comments with a high degree of relevance to the cluster title. This made it possible to convey the content of each cluster more accurately.

That said, this is fundamentally also a problem that should be addressed by improving the clustering itself. Preventing loosely related opinions from being grouped into the same cluster in the first place would be a more fundamental solution. In the 広聴AI (Kouchou AI, Broad Listening AI) developed later, improvements to the algorithm were attempted in order to address this issue. Details are explained in Chapter 13.

### 4.3.6 Challenges and Limitations

In operating broad listening, this initiative encountered several challenges.

**The Problem of Hallucinations and Misinformation**

The analysis results can sometimes include content that is not factually correct. There are two causes of this problem.

The first is LLM hallucination. When AI extracts or summarizes opinions, it may generate information that is not present in the original data.

The second is that the original posts themselves may contain false information. Posts on X are a mixed bag, including misinformation and inaccurate claims. Because TTTC aggregates them without distinguishing between them, factually incorrect content can end up appearing in the analysis results.

In this initiative, the team fact-checked cluster titles and summaries, manually correcting statements that were factually wrong or potentially misleading. In many cases, Nippon TV reporters working in political news were better positioned than Anno's team to judge factual accuracy, so the work proceeded through back-and-forth between the two sides. This took significant time and cost.

**Bias Arising from the Data Source**

Because the analysis targets data on X, the results are inherently biased. X users represent only a portion of Japan's population, and the number of people who post about elections on X is smaller still. Posts by children or foreign users may also be included. There is also a risk that the opinions of especially vocal users will be overrepresented.

In addition, figures such as cluster proportions and post counts can easily appear "objective" to viewers, almost like public opinion polling. For that reason, it is necessary to clearly communicate that the analysis is based on the limited data source of X.

**Explanation to Viewers and Care in Wording**

Given these constraints, the program explained to viewers that the analysis covered voices posted on X, that it was not collecting a broad range of voices in the way an opinion poll does, and that care was needed because it reflected online discussion.

Based on this experience, the following wording rules were established for the broadcast:

1. Do not use words such as "the public will" or "public opinion"; instead, say "voices on X" or "voices on social media"
2. Avoid subjects such as "voters'..." because the people posting are not necessarily eligible voters
3. Clearly state that the data source is the limited platform X

Broad listening results are an analysis of a limited subset. Communicating that premise clearly to viewers was just as important as the analysis results themselves.

### 4.3.7 Response and Sense of Impact

The YouTube program received comments such as "This kind of data provision will be important for journalism from now on" and "I hope each political party will use AI to analyze opinion in online spaces." Some broadcasts reached 500,000 views on YouTube.

A Nippon TV staff member reflected on the significance of the effort as follows: "Until now, if we wanted to incorporate online voices into election coverage, we had no choice but to arbitrarily select and introduce individual posts. Broad listening made it possible to visualize what kinds of opinions were gathering in online spaces as a whole, which was a breakthrough." The ability to view overall tendencies, including minority opinions, from 10,091 data points was something difficult to achieve with conventional methods.

Broad listening also became an effective means in pre-election coverage for visualizing the gap between the issues politicians were emphasizing and the issues the public was thinking about, and for putting that gap to politicians. Some politicians who appeared on the program were overwhelmed by the volume of information, which could not be fully digested in a one-hour broadcast, but later looked back at the analysis results after the program and reportedly remarked, "So this is what it looks like."

### 4.3.8 Remaining Questions

The 2024 House of Representatives election project was the first case of bringing broad listening into terrestrial television election coverage. At the same time, it also surfaced unresolved issues.

**What Are "Online Voices"?**

In this initiative, the data source was again limited to X. In the 2024 Tokyo gubernatorial election, it became widely recognized that voices on social media could influence election dynamics. Since then, however, gaps have also become visible between online excitement and actual voting behavior, as candidates who appeared to have strong online support failed to perform well in actual elections.

As concerns grow that discourse on X tends to become more extreme, there is also a risk that extracting only voices from that space will present a biased image. On the other hand, if analysis is expanded to include other social media platforms, technical constraints and human costs increase significantly. There is also the structural problem that only some social media platforms are well suited to text-based analysis.

Ultimately, this leads to the question of how "online voices" can be handled as one valid aspect of public opinion. This is not a problem unique to broad listening; it is also shared by conventional methods such as opinion polls and street interviews.

### Summary of This Section

The Nippon TV election special case shows two principles for putting broad listening into practice.

First, the quality of the analysis is determined not by the algorithm, but by the design of data collection. How the question "What do we want to know?" is framed determines the results.

Second, when communicating the results of social media analysis in a public setting, careful design of expression is essential. Do not call it "public opinion"; state the data source; do not imply representativeness. These kinds of rules form the foundation that supports the credibility of broad listening.

---

### Notes

[^1]: Nippon TV report publication page: https://news.ntv.co.jp/category/society/1594a26c1d794967a9245ed34e70d681

[^2]: Analysis results for the House of Representatives election as a whole during the pre-announcement period: https://news.ntv.co.jp/static/shugiinsenkyo2024/whole-1015/index.html

[^3]: Analysis results for policy during the pre-announcement period: https://news.ntv.co.jp/static/shugiinsenkyo2024/policy-1015/index.html
