## 4.3 Nippon TV Election Special: The World’s First Use in Television News Reporting

Written by: @nasuka


### 4.3.1 Use in Election Coverage

In the October 2024 House of Representatives election, Nippon TV carried out election coverage using broad listening. This was likely the **world’s first example of broad listening being used in television news reporting**.

During the election period, a YouTube program titled *Who Should We Vote For? A Meeting to Create the 2024 House of Representatives Election Through Everyone’s Voices* was streamed in six episodes. The format involved collecting opinions about each political party from X (formerly Twitter), analyzing them with TTTC, and then inviting lawmakers from each party to discuss the findings. The reports visualized which topics were attracting attention on X, and Takahiro Anno also participated as a commentator.

Then, on October 27, election day, a terrestrial TV election special titled *zero Election 2024: How Japan Will Change* was broadcast. Posts on X related to the House of Representatives election were collected for about 25 minutes starting from 8:00 p.m., when voting closed, and analyzed with TTTC to produce a report. During the program, Anno explained the scatter plot and delivered voters’ reactions immediately after the polls closed in a near-real-time format.

Each report that was produced was not only used on the program, but was also made publicly available online.

### 4.3.2 Analyses Conducted

To focus the analysis on what kinds of opinions were being expressed about policies and political parties, the following four types of analysis were conducted.

1. **Analysis of opinions on the House of Representatives election as a whole**: to provide an overview of voter interests and topics related to the election overall
2. **Analysis of opinions on policy**: to understand views on specific policy areas such as the economy, social security, and foreign affairs
3. **Analysis of opinions on each political party**: to understand how each party was being discussed
4. **Analysis of opinions immediately after the polls closed**: to capture voter reactions at the moment vote counting began

The data analyzed was collected from X, and a separate query was prepared for each type of analysis.  
The analysis was conducted using a modified version of TTTC.  
Analyses 1 through 3 were carried out in three phases: before the official campaign announcement, the first half of the campaign, and the second half of the campaign. Analysis 4 focused on collecting and analyzing data intensively during the short period immediately after the polls closed.

![Scatter plot of analysis results produced by TTTC. Opinions are displayed in different colors by cluster](images/ntv_scatter.png)

In the scatter plot, each point represents one opinion. Opinions that are semantically similar are placed closer together, and opinions belonging to the same cluster are shown in the same color. The vertical and horizontal axes themselves do not have any specific meaning; rather, they show high-dimensional data compressed into two dimensions.

#### Differences in Results Depending on the Data Collected

Even for data from the same election period, the picture that emerges can differ greatly depending on the query used to collect it. Let us compare the results for “the House of Representatives election as a whole” and “policy” during the pre-announcement period (October 6–12).

In the **analysis of the House of Representatives election as a whole**[^1], posts related broadly to the election were collected.

Clusters emerged around topics such as “interest in the consumption tax and the slush-fund scandal,” “opposition cooperation and election strategy,” and “personal election-related experiences and effects on daily life.” The discussion centered on “the election as an event in itself,” including criticism of the slush-fund scandal, debate over opposition candidate coordination, declarations of intent to vote, and complaints about the noise from campaign vehicles.

In the **policy analysis**[^2], posts expressing opinions about policy were collected.

Clusters emerged around topics such as “dissatisfaction with and criticism of the ruling parties,” “concerns about tax increases and dissatisfaction with economic policy,” “concerns about national security and immigration policy,” and “debate over reducing or abolishing the consumption tax.” Opinions were grouped by specific policy themes such as cutting or abolishing the consumption tax, concerns about immigration policy, raising the minimum wage, allowing married couples to keep separate surnames, and the My Number health insurance card (My Number, Japan’s national ID system). This made it much clearer what voters saw as the key issues.

**What this comparison shows**

This may go without saying, but the results of an analysis—and the insights that can be drawn from it—depend on the data collected. For example, if the goal is to analyze opinions on policy, collecting election-related data broadly may not produce the desired results. That is because data on the election as a whole contains many topics related to political maneuvering and campaign activity, which can bury discussion of policy.

That is why it is important first to clearly define the question, “What do we want to find out?” and then design the data collection so that it gathers data capable of answering that question.

There is a saying: “Garbage in, garbage out.” If the quality of the input data is poor, even the best analytical methods will not produce useful results. In broad listening as well, collecting appropriate data suited to the objective is the key factor that determines whether the analysis succeeds.

### 4.3.3 Analysis Immediately After the Polls Closed

Unlike analyses that could be prepared in advance (three periods × three types: before the election period, the first half of the campaign, and the second half), the analysis immediately after the polls closed relied on data that did not exist until election day itself.

After voting closed at 8:00 p.m., posts on X were collected, the analysis was run, the results were reviewed, and the findings were finalized into a report. From the start of data collection to its use on air, all of this had to be completed in about three hours.

It was an extremely demanding schedule, but it would have been difficult for humans to produce an equivalent analytical report from scratch. In that sense, it was only possible because broad listening technology was used. This case demonstrated the usefulness of the technology in situations like election reporting, where real-time responsiveness is required.

### 4.3.4 Improvements to TTTC for Broadcast Use

To use TTTC in news reporting, major improvements were required in both functionality and UI. The main enhancements that were implemented are introduced below.

**Implementation of a favorites feature**

In the previous version of TTTC, there was no way to save specific comments found among thousands of data points. This created the problem of having to search around asking, “Where was that comment again...?” Even if producers had identified opinions they wanted to introduce during a live broadcast, they could not reliably find them smoothly when the program was actually on air.

To solve this issue, a favorites feature was implemented. By bookmarking comments of interest, they could be called up with a single click during the program, and people viewing the published reports could also revisit comments they found noteworthy.

![Detailed cluster view. You can see the cluster analysis results, representative comments, and the favorites feature](images/04_03_ntv_cluster_example_en.png)

**UI improvements for smartphones and tablets**

The scatter plot in the previous version of TTTC was designed mainly with a PC-oriented UI, which made it difficult for smartphone users to view analytical reports. For example, in the smartphone version of the report, there were issues such as being unable to zoom in and out on the full-screen map.

In this use case, it was expected that many viewers would access the reports on smartphones. For that reason, the UI was redesigned to make it easier to use on mobile devices.

**Improved accuracy in extracting representative comments**

TTTC has a feature that displays comments representing each cluster. Previously, however, comments were selected at random from within a cluster, which sometimes resulted in opinions being highlighted that had little connection to the cluster title.

To address this problem, the system was improved to use an LLM to extract comments with a high degree of relevance to the cluster title. This made it possible to convey the content of each cluster more accurately.

That said, this is fundamentally also a problem that should be addressed by improving the clustering itself. Preventing loosely related opinions from being grouped into the same cluster in the first place would be a more fundamental solution. In the Public Listening AI developed later, improvements to the algorithm were attempted in order to address this issue. Details are explained in Chapter 13.

### 4.3.5 Challenges and Limitations

In operating broad listening, this initiative encountered two major challenges.

**The problem of hallucinations and misinformation**

The analysis results can sometimes include content that is not factually correct. There are two causes of this problem.

The first is LLM hallucination. When AI extracts or summarizes opinions, it may generate information that is not present in the original data.

The second is that the original posts themselves may contain false information. Posts on X are a mixed bag, including misinformation and inaccurate claims. Because TTTC aggregates them without distinguishing between them, factually incorrect content can end up appearing in the analysis results.

In this initiative, the generated reports were reviewed, and any incorrect descriptions were corrected manually.

**Bias arising from the data source**

Because the analysis targets data on X, the results are inherently biased. X users represent only a portion of Japan’s population, and the number of people who post about elections on X is smaller still. There is also a risk that the opinions of especially vocal users will be overrepresented.

In addition, figures such as cluster proportions and post counts can easily appear “objective” to viewers, almost like public opinion polling. For that reason, it is necessary to clearly communicate that the analysis is based on the limited data source of X.

**Explanation to viewers**

These constraints were also explained to viewers during the program. The broadcast made clear that the analysis did not represent public opinion itself, but was an analysis of posts on X.

When using the results of broad listening analysis, it is important to properly communicate these assumptions and limitations to the people viewing the results.

---

### Notes

[^1]: Analysis results for the House of Representatives election as a whole during the pre-announcement period: https://news.ntv.co.jp/static/shugiinsenkyo2024/whole-1015/index.html

[^2]: Analysis results for policy during the pre-announcement period: https://news.ntv.co.jp/static/shugiinsenkyo2024/policy-1015/index.html
