# Chapter 4: The Spread of Broad Listening in Japan

<!--MEMO: Permission to use the images has not been obtained, so the images have not been committed and are only in Nakayama's possession. -->

## The Cutting Edge of Data Journalism at The Asahi Shimbun

The Asahi Shimbun, one of Japan's major national newspapers, through its Media Research & Development Center (commonly known as M-Lab)[^mken], began with a May 2025 article analyzing social media posts about cutting the consumption tax using Talk to the City[^consumption_tax], and has since applied broad listening to reporting on a range of topics, including the House of Councillors election and the Expo. We interviewed Shoichi Otsu of M-Lab and Keisuke Yamazaki of the Data Journalism Team in the Digital Investigative Reporting Department about the background to these efforts and where they may lead next. (Interview conducted in January 2026)

### What Is the Media Research & Development Center?

**Nakayama** To start, could you tell us what kind of organization The Asahi Shimbun's Media Research & Development Center is?

**Otsu** The Media Research & Development Center, or M-Lab, was established in April 2021. It is a research and development organization aimed at solving a variety of internal and external challenges by making use of the kinds of content resources a newspaper company is uniquely strong in—abundant text, photographs, audio, and video. In particular, it seeks to create new value in journalism and media through technology.

M-Lab mainly conducts research and development centered on artificial intelligence (AI), natural language processing, and image processing. Its research scope has expanded beyond text into audio and video as well. In data journalism, we also analyze trends in online discourse through social media analysis and use those findings in articles, while also visualizing and analyzing social phenomena using geographic and statistical data.

I only transferred to M-Lab last April, so it has been about nine months. I joined the company on the technical side, but after working in a technical role for just two years, I spent the rest of my career in editorial work. I spent about ten years in the layout desk, working internally on page design and headlines, and about another ten years as a field reporter. After that, I worked in the marketing division doing analytical work, and that led naturally to the kind of analysis I do now.

### What Is Data Journalism?

**Nakayama** Could you explain data journalism in a way that is easy for readers to understand?

**Yamazaki** I joined the company as an engineer, but after working in regional postings and then as a reporter at headquarters, I now work as a reporter on the Data Journalism Team in the Digital Investigative Reporting Department. We are a team of about five people, and our job is to use data to find leads for reporting and then write articles.

In the past, investigative reporting was basically paper-based. At our company we call it *mekuri*—literally flipping through documents thoroughly to find leads. That was a typical reporting method. But as IT developed, data accumulated, and open data increased, it became possible to analyze data while looking for leads. We can now analyze large volumes of data in ways that were not possible before. That is data journalism.

### What Led to the Introduction of Talk to the City?

**Nakayama** You used Talk to the City for analysis. What led you to decide to try it?

**Otsu** By the time I came to M-Lab in April last year (2025), there were already people there experimenting with Talk to the City. But it had not yet reached the point of being used in reporting.

The trigger was a request from the political desk. There was a case of broad listening being used to inform questions in the Japanese Parliament, and we were going to cover it in a series called "AI and Democracy."[^ai_democracy]

In that context, someone asked whether The Asahi Shimbun could use a similar method to show something of its own. At the time, cuts to the consumption tax were a hot topic, so we decided to try using consumption tax policy as the theme.[^consumption_tax] That became our test case.

![Scatter plot of social media views on cutting the consumption tax](images/04_04_消費税減税SNS散布図.png)

### What Emerged from the 2025 House of Councillors Election Analysis

**Nakayama** Was there anything newly revealed in the analysis results, or anything you found particularly interesting?

**Otsu** What stayed with me most was that when we tried broad listening before the 2025 House of Councillors election, anti-foreigner views appeared to have increased compared with the 2024 House of Representatives election—at least within the range of social media posts we collected. That cluster had a lot of volume, or at least it stood out.[^sangiin1]

![Distribution of opinions exchanged on social media about the House of Councillors election. The "foreign nationals policy" cluster sits adjacent to "expectations for political parties and leadership," "the performance of politics and politicians," and "population decline," suggesting a link between political distrust and anti-foreigner views.](images/04_04_参院選SNS意見分布.png)

We had assumed that "on the internet, anti-foreigner opinions are always more likely to attract noise and attention." But once the election campaign actually began, anti-foreigner sentiment and issues involving foreign nationals really did become major points of focus. It made me feel that what broad listening had shown was something that then unfolded in reality. I found that very interesting.

**Yamazaki** The scatter plot—the final output of broad listening—is really one of its strengths. AI aggregates the many different opinions on social media and sorts them by theme. Looking at the plot, you can also get a sense of the distance between themes—how close or far apart one theme is from another.

When we analyzed the entire election period, the most common theme was something like "restoring trust in politicians," but nearby there was also a theme like "foreign nationals policy."[^sangiin2] From the results, you could read an image in which distrust of established parties was giving rise to anti-foreigner views. I think that helped make the benefits of using Talk to the City visible.

![Distribution of opinions posted on social media just before voting day in the House of Councillors election (July 1–10, 2025; about 30,000 posts analyzed with Talk to the City)](images/04_04_参院選投票日直前SNS意見分布.png)

**Otsu** Exactly. Looking at the cluster titles on the scatter plot, there were about three clusters gathered near the center. When you looked closely, all of them were really talking about political distrust. They were separated as categories, but there was one large core of political distrust in the middle. That made me feel that the positions themselves carry meaning.

### Expanding Use Cases

**Nakayama** Do you plan to continue this kind of work going forward? My impression is that since the second half of 2025, there has been less visible use of Talk to the City by media organizations.

**Yamazaki** Rather than saying people stopped using Talk to the City, I think it is more that over the past six months there have been fewer topics suited to using this system to visualize public opinion. Before that, there had been events like the House of Representatives election (October 2024), the House of Councillors election (July 2025), and the Tokyo governor election (July 2024), where opinion on social media was sharply divided. Without events like that, it can be harder to use.

That said, our company has continued working with it. In October 2025, we used Talk to the City for the Miyagi governor election[^miyagi], and before that we also analyzed social media discourse around the Expo.[^expo]

![Results of analysis of X posts about the Osaka Expo (first period: April–July 2025). There are many critical clusters, including dissatisfaction with operations and criticism of construction costs.](images/04_04_万博SNS分析_前期.png)
![Results of analysis of X posts about the Osaka Expo (second period: July–October 2025). More experience-based opinions appear, including positive evaluations of visiting and comments expressing a desire to return.](images/04_04_万博SNS分析_後期.png)

**Otsu** We also used it in a survey of atomic bomb survivors. That was for a joint feature with the *Chugoku Shimbun* in Hiroshima and the *Nagasaki Shimbun*, titled "Tsumugu."[^tsumugu] We used Talk to the City to analyze the free-response answers in the survey.

**Yamazaki** There are all kinds of ways to use it. It is not limited to social media analysis. In fact, our company uses it in more ways than people might expect.

### Technical Ingenuity

**Nakayama** Was it difficult to embed the HTML output from Talk to the City into The Asahi Shimbun's website?

**Yamazaki** The scatter plots embedded in the articles are not actually using Talk to the City's HTML as-is. We use a visualization tool called "Flourish."[^flourish]

Mr. Otsu does the analysis, and then we take the coordinate data and embed it into Flourish, which we had already been using, to display it. Talk to the City outputs data as XY coordinates, so if you input those, it can be done. We tried to make it as easy to read as possible for general readers.

### The Challenge of Labeling Clusters

**Nakayama** Were there any issues with the labels generated by Talk to the City?

**Otsu** The first thing we struggled with was that the labels generated by Talk to the City could not be used as they were. They were too abstract and hard to understand.

One thing I did was instruct it in the prompt: "Please provide five headlines." If you ask for just one, it tends to be abstract, but if you ask for five, it is forced to become more concrete. I think Mr. Yamazaki then looked at those and created the labels.

**Yamazaki** We had a very hard time with that during the House of Councillors election. There was no way we could publish the original labels as they were, so in the end I rewrote the labels for the article myself. Before the real thing, we used data from several previous elections and experimented repeatedly with prompts to figure out what kind of output would work best.

**Otsu** One concern, of course, was what we would do if we ran the analysis right before the election and got results that did not work well. So we wanted to test it in advance. We used data from the House of Representatives election a year earlier for that purpose.

Another thing we had in mind was comparing how people's voices changed from the House of Representatives election to the House of Councillors election. If the difference had been clear, we would have liked to use that as well, but it was not clear enough, so we did not use the earlier data.

**Yamazaki** We also tried different numbers of clusters several times to see what worked best.

**Otsu** Yes, the number of clusters was another challenge. Fewer clusters are easier for readers to understand, but they become too abstract. If you increase the number to 40 or 50, you get clusters that are clearer and more interesting, but at that point it becomes impossible to communicate them all to readers.

So we try to keep the clusters relatively large while making the headlines more effective.

Also, among the many opinions posted on X (formerly Twitter), I think it might be useful to have a function that excludes opinions that are essentially "off-topic." Because every opinion is placed on the scatter plot, everything outside the neatly grouped areas ends up forming a large mass. For example, if you want to present ten clusters in an article, the largest cluster may end up being one whose meaning is unclear—a cluster without any real coherence of opinion. That makes it very difficult to handle journalistically. It would be better if the clustering focused only on content relevant to the theme at hand.

### The Difference Between Social Media Analysis and Survey Analysis

**Nakayama** Is there any difference between social media analysis and survey analysis?

**Otsu** My impression is that surveys cluster much more cleanly than social media. The opinions are aggregated in a much clearer way.

In the case of social media, we collect posts through keyword searches, but the opinions point in many different directions. In a survey, by contrast, respondents are answering within a somewhat defined frame, so when you cluster the responses, the opinions separate much more cleanly.

The survey of atomic bomb survivors I mentioned earlier, "Tsumugu," was analyzed very cleanly.

**Nakayama** Were there any other difficulties in carrying out the analysis?

**Otsu** The clustering itself in Talk to the City is difficult enough, but another challenge was figuring out what keywords to search for in order to collect good posts. How do you gather posts that contain a high density of content related to the theme—whether that is "House of Councillors election" or "consumption tax"? That is probably the hardest part. We went through a lot of trial and error.

### Reader Response

**Nakayama** What kind of response did you get from readers or on social media?

**Yamazaki** To be honest, if you ask whether there was a large and vocal response from readers, the reality is that there was not much. I cannot share the pageview numbers, but they were decent.

That said, looking at social media, there were comments like, "It's nice to see The Asahi Shimbun quietly using Talk to the City."

Also, *Asahi Shimbun Digital* has a comment section where experts can post comments, and one of them said that it is important to look at public opinion by combining traditional opinion polling with Talk to the City, and that they hoped we would continue putting effort into that.

We also heard from the public editor—an outside figure who serves something like an auditor for the editorial division. They commented on the significance of a newspaper doing this. They said it was good to see an approach that combines data-driven efforts like Talk to the City with traditional newspaper reporting based on talking to people and conducting opinion polls.

### Why Publish the Source Code?

**Nakayama** You have published the code used for the analysis on GitHub. Why did you decide to make the source code public?

**Otsu** Because AI is being used for classification and labeling, I think there is a challenge in that the decision-making process is harder to see than with statistical methods and can become a black box.

If we simply present the results, it is hard for readers to understand what standards were used and how the conclusions were reached. That could invite suspicion that the analysis was arbitrary. We believe it is important to ensure verifiability as much as possible, which is why we publish the source code.[^github_consumption][^github_expo] If the code is public, then specialists should be able to reproduce something similar and check it for themselves.

**Yamazaki** In the Digital Investigative Reporting Department, I and several others who can write code have been publishing things on GitHub on our own initiative. Our boss has said, "Sounds good," so we have just kept publishing more and more.

**Nakayama** Incidentally, when developing 広聴AI (Kouchou AI - meaning Public Relations AI), we also referred to code published by The Asahi Shimbun. Thank you very much.

### Looking Ahead

**Nakayama** Please tell us about your outlook going forward.

**Otsu** I have heard that development of Talk to the City has stopped,[^tttc_archived] but I do not think the broader direction will change: we will still want to keep using tools that can aggregate this kind of opinion to some extent.

When I show the results to various editors and say, "Would you like to try this kind of analysis with Talk to the City?" they often respond, "This is great." What that makes me think is that, while I am not really much of a programmer, I want something that can be operated through a user interface.

At the same time, if it runs on another company's servers, then we cannot upload the data there, so ideally we would want something that can be done locally. If we have a contract with OpenAI that guarantees the security of the input data, then using ChatGPT or its API is not a problem. But if it means uploading data to another company's servers, that still feels like a hurdle.

**Yamazaki** In future elections and similar events, I think we will need to combine analysis not just of X but also of video and other social media platforms. Video is harder to analyze because it is difficult to tell what exactly a post is positive or negative about, but given the current pace of AI progress, it may become possible to understand to some extent what kinds of comments were attached to what kinds of videos.

Not just with Talk to the City, but by making good use of new technologies and AI, I want to keep thinking up new forms of investigative reporting. That is something I think about every day in my work.

---

[^mken]: The Asahi Shimbun Media Research & Development Center https://cl.asahi.com/
[^miyagi]: The Asahi Shimbun article analyzing the Miyagi governor election https://www.asahi.com/articles/DA3S16332343.html
[^ai_democracy]: The Asahi Shimbun series "AI and Democracy" https://www.asahi.com/rensai/list.html?id=3039
[^consumption_tax]: The Asahi Shimbun article analyzing cuts to the consumption tax https://www.asahi.com/articles/AST5Q3DZ4T5QUTFK001M.html
[^expo]: withnews article analyzing social media discourse on the Osaka Expo https://withnews.jp/article/f0251026001qq000000000000000W0jf10201qq000028332A
[^expo_github]: GitHub Osaka Expo analysis https://github.com/asahi-research/Asahi_TTTC_osaka_expo
[^tsumugu]: The Asahi Shimbun survey of atomic bomb survivors, "Tsumugu" https://www.asahi.com/special/tsumugu-asahi/
[^sangiin1]: The Asahi Shimbun article analyzing the House of Councillors election (during the campaign) https://www.asahi.com/articles/AST7441D7T74ULLI00HM.html
[^sangiin2]: The Asahi Shimbun article analyzing the House of Councillors election (after the election) https://www.asahi.com/articles/AST7Z2BVZT7ZULLI008M.html
[^github_consumption]: GitHub consumption tax analysis https://github.com/asahi-research/TTTC_consumption_tax_20250525
[^github_expo]: GitHub Osaka Expo analysis https://github.com/asahi-research/Asahi_TTTC_osaka_expo
[^flourish]: Flourish https://flourish.studio/
[^tttc_archived]: The initial Talk to the City repository, `talk-to-the-city-reports`, was archived on July 31, 2025, but development continues in the successor lightweight repository, `tttc-light-js`. https://github.com/AIObjectives/talk-to-the-city-reports / https://github.com/AIObjectives/tttc-light-js/
