## 10.3 Bowling Green: “America’s Largest Town Hall,” with 7,890 Participants

### 10.3.1 Bowling Green’s Challenge

Taiwan’s vTaiwan was a pioneering example of consensus-building using Polis. The second case introduced here is a large-scale civic dialogue project conducted in a local American city.

Bowling Green, located in southern Kentucky, is a city of about 76,000 people. If you search for “Bowling Green, KY” on a map app, you can see new residential developments spreading one after another across gently rolling farmland. Thanks to the spillover effects of Nashville’s rapid economic growth, located about an hour away by car, the acceptance of immigrants from around the world through a refugee resettlement center, and a strong employment base centered on Western Kentucky University and General Motors’ Corvette plant, the population grew by about 1.5 times between 2000 and 2020. According to projections by the University of Louisville’s Kentucky State Data Center, Warren County’s population is expected to reach about 230,000 by 2050.

To respond to this rapid growth, a 25-year strategic plan called “BG 2050” was launched. The initiative came from Warren County Judge Doug Gorman, prompted by his learning in 2017 of projections that the population could double. As he put it: “Are we going to let this change happen *to* us, or make it happen *for* us? If it’s the latter, we need a plan.”

In the conventional planning process, it was typical for about 100 local leaders to be involved. But BG 2050 did not stop there. To hear the voices of the community as a whole, it embarked on a large-scale online dialogue.

### 10.3.2 Polis + Sensemaker: From Collecting Opinions to AI Analysis

The technology supporting this large-scale dialogue was [Sensemaker](https://github.com/Jigsaw-Code/sensemaking-tools), an open-source tool developed by [Jigsaw](https://jigsaw.google/), a technology unit under Alphabet (Google).

Sensemaker is a toolkit for structuring and summarizing large-scale online dialogue. Using LLMs, it extracts topics, classifies opinions, and generates summary reports that make areas of agreement and disagreement explicit. Functionally, it is almost equivalent to Talk to the City (TTTC), introduced in Chapter 11, and the idea of using AI to interpret Polis output was likely influenced by how TTTC was used in Taiwan. The main differences are that, whereas TTTC assumes ChatGPT, Sensemaker uses Google Gemini as its AI model, and that instead of TTTC’s scatterplot-style visualization, it presents results as a hierarchical list of topics and subtopics.

One especially notable aspect of Sensemaker’s design is that it is built with Polis integration in mind. By exporting the opinions and voting data collected in Polis as CSV files and feeding them into Sensemaker, thousands of comments can be transformed into a structured analytical report. Chris Small, co-founder of Polis, is also a software engineer at Jigsaw, and the two projects are closely connected both personally and technically.

In other words, an open-source pipeline was created that connects Polis’s strengths in “collecting and visualizing opinions,” as demonstrated by vTaiwan, with Sensemaker’s strengths in “large-scale AI analysis and structuring.”

### 10.3.3 “What Could BG Be?”: A 33-Day Experiment

Using this pipeline, Bowling Green carried out the “[What Could BG Be?](https://report.whatcouldbgbe.com/)” project in 2025.

For 33 days, from February 14 to March 17, 2025, residents could visit [whatcouldbgbe.com](https://www.whatcouldbgbe.com/), freely submit their own ideas, and vote on other residents’ ideas with “agree,” “disagree,” or “pass.” The system was designed not to collect any personal information or demographic data.

![BG 2050: Over 33 days, 7,890 residents participated and cast more than 1 million votes](images/column_bg2050_stats.png)

The results were striking. **7,890** residents participated, **3,940** original ideas were submitted, and **1,034,868 votes** were cast. People involved with the Computational Democracy Project, the organization behind Polis, described it as “one of the most active conversations in the history of Polis.” Judge Gorman called it “America’s largest town hall.”

### 10.3.4 The Map of Consensus Revealed by AI

Sensemaker analyzed the 3,940 ideas collected over those 33 days. A task that would have taken hundreds of hours to organize manually was processed by AI in just minutes.

As a result, the ideas were automatically classified into 12 topics and 70 subtopics. The overall landscape of residents’ concerns became visible, including arts and culture (644 items), infrastructure and transportation (574), economic development (499), and community identity (479).

![Topic classification by Sensemaker: 3,940 ideas were automatically sorted into 12 topics and 70 subtopics](images/column_bg2050_topics.png)

When each topic is selected, individual ideas are visualized in a dot chart ordered by agreement rate, alongside an AI-generated summary.

![Topic detail: A single screen shows both the distribution of agreement rates in a dot chart and an AI-generated summary](images/column_bg2050_topic_detail.png)

Most noteworthy is the distribution of agreement rates. According to Judge Gorman, **about 60% of all ideas—2,370 in total—recorded agreement rates above 80%**.

The seven themes that drew the strongest agreement among residents were as follows:

1. More family-friendly activities
2. Balanced development across the county
3. Access to grocery stores
4. Connecting downtown and the riverfront
5. Better sidewalks and walkability
6. Reducing congestion on Scottsville Road (a major arterial road)
7. Flood and disaster preparedness

![Comparison of proposals with high agreement rates and proposals that divided opinion](images/column_bg2050_consensus.png)

Particularly notable is that some areas achieved high agreement even among politically opposed groups of residents: parks and green space, housing affordability, access to mental health resources, sustainable development, preservation of historic buildings, and making use of existing commercial space rather than building new structures. In a local American city, the significance of making visible this common ground across political divides is considerable.

Scott Carpenter, head of Jigsaw, described it as “like Google Maps for conversation.” Just as a map shows the shape of the land, this tool shows the terrain of opinion.

### 10.3.5 From Voice to Policy: Achievements and Limits

Up to this point, the process is still at the stage of “collecting voices and making them visible.” As seen in the vTaiwan case, if there is no design for connecting these results to policy, the project risks ending as little more than a safety valve.

In Bowling Green, after the report was published, volunteer implementation teams of 10 to 12 people for each theme—called pillar groups—were formed, and the process moved into drafting policy recommendations for Warren County leadership. By the summer of 2025, some measures had already been put into action by the Warren County Fiscal Court. A budget of $5 million was approved to establish a unified fire department by consolidating nine volunteer fire departments.

In follow-up surveys of participants, **70% said they felt more confident that their voice could influence important issues**. Another notable point is that the total budget for the entire project was under $10,000. Polis is open source, and Sensemaker is also available free of charge. As the cost of technology has fallen, the barrier to “listening to citizens’ voices” has dropped dramatically.

On April 15, 2025, a time capsule containing the project’s results was buried. It is scheduled to be opened 25 years later.

That said, this success comes with an important qualification. More than 80% may agree on “more family-friendly activities.” But the moment the discussion turns to specifics—“what should be built, where, and who should pay for it?”—consensus becomes much harder. The consensus extracted by AI becomes more fragile as the level of specificity rises. This is not a flaw in the technology. It is an inherent feature of human consensus-building itself. That is precisely why a stage like the pillar groups, in which people take the AI’s output and turn it into concrete proposals, is indispensable. AI can draw a “map of consensus.” Deciding where to go on that map remains human work.

### 10.3.6 “We The People”: Expansion Across the United States

The success of Bowling Green did not remain just an experiment in one local city.

July 4, 2026, marks the 250th anniversary of the founding of the United States. Ahead of that milestone, in November 2025, the think tank [Napolitan Institute](https://napolitannews.org/) announced a nationwide initiative called “[We The People](https://wethepeople-250.org/)” in collaboration with Jigsaw. The project aims to gather citizens’ voices from all 435 congressional districts around the theme: “As the nation reaches its 250th year, what do freedom and equality mean to Americans?”

The project proceeds in two stages. In the first stage, participants converse with an AI chatbot. By probing further with personalized follow-up questions, the AI draws out responses averaging **140 words**, compared with the average of just eight words typically obtained in ordinary surveys. In the second stage, the AI generates summary statements from the large volume of conversations, and participants vote on them by agreeing or disagreeing. In Bowling Green’s Polis process, participants themselves submitted ideas, but in We The People that role is taken on by the AI chatbot, representing an evolution in design aimed at eliciting deeper views.

More than 2,400 people took part in the first dialogue, sharing a total of over 1.6 million words. Of the 26 summary statements generated by AI, **22 recorded agreement rates above 80%**. Meanwhile, 94% of participants said they felt their views were reflected.

The [results of the dialogue on freedom and equality](https://freedom.wethepeople-250.org/) are published on the website, making visible cross-partisan areas of agreement such as: “Freedom means guaranteeing basic human rights and opportunity regardless of nationality or background,” and “Equality means that everyone is treated equally under the law and that their voice is respected.” A final report is scheduled for release in conjunction with the 250th anniversary of the nation’s founding, and the pipeline demonstrated in Bowling Green—“opinion collection → AI analysis → consensus-building”—now appears poised to scale to the national level.
