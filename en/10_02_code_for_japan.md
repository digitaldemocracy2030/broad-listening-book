# Code for Japan

Written by: Kenjiro Higashi (Code for Japan)

## Introduction

Code for Japan (CfJ) supports richer and more advanced participation processes by using 広聴AI (Kouchou AI, Broad Listening AI) to visualize citizens' opinions. This article introduces cases where CfJ has worked with local governments to use Kouchou AI, then outlines the practical issues local governments tend to face during implementation and operation, as well as the broader questions that lie beyond them.

### About Code for Japan

CfJ is a nonprofit general incorporated association that promotes citizen participation and collaboration with government through civic tech practice, under the vision of "a society we think through and build together." Its main areas of work are: support for digital transformation in local governments; support for introducing and operating citizen-participation tools that use open source, and for building data-linkage platforms based on open standards such as FIWARE; and formation and development of civic tech communities. In the context of citizen participation in particular, CfJ supports the introduction and operation of Decidim, the open-source citizen engagement tool that originated in Barcelona, Spain, as its only official partner in Japan.

Reference: Code for Japan website
[https://www.code4japan.org/](https://www.code4japan.org/)

## Case Studies

CfJ has also worked with local governments on Kouchou AI from the perspective of how citizens' voices can be connected to policymaking and participation mechanisms.

### Using Existing Data: Analysis of a Resident Awareness Survey in Kakogawa City, Hyogo Prefecture

Kakogawa City conducts surveys of residents' awareness, including satisfaction and importance ratings, for use in monitoring progress on its Comprehensive Plan for fiscal 2021 through fiscal 2026 and in considering future policy development. Of this data, CfJ used the open-ended responses that had been preprocessed by the city hall and publicly released, so that staff could understand the characteristics of Kouchou AI. The data came from a fiscal 2023 survey of 6,000 residents aged 18 or older, randomly sampled from the Basic Resident Register, with a valid response rate of 33.9%.

Reference: Kakogawa City Fiscal 2023 Resident Awareness Survey
[https://www.city.kakogawa.lg.jp/soshikikarasagasu/kikakubu/kikakubukohoka/kakogawashinoseisakuzaisei/shiminishikichosa/43030.html](https://www.city.kakogawa.lg.jp/soshikikarasagasu/kikakubu/kikakubukohoka/kakogawashinoseisakuzaisei/shiminishikichosa/43030.html)

![Figure: Kouchou AI visualization of Kakogawa City's resident awareness survey, showing clustered free-text responses in the main view.](images/10_02_01_kakogawa-survey.png)

![Figure: Filter settings for the Kakogawa City resident awareness survey visualization, including attributes such as age and residential area.](images/10_02_02_kakogawa-survey-filter.png)

(Figure 1) Visualization of Kakogawa City's resident awareness survey and filter settings

Because the original data included age and residential-area attributes, these were set as filters, making it possible to visualize differences in opinion distribution by age group and by area. City staff found some aspects of the categorization difficult to understand, such as cases where comments about the same childcare-related topic were placed in different clusters. At the same time, they also seem to have felt that because the number of groups and the results change with each generation, giving feedback on each result can improve the accuracy and usefulness of the visualization.

When already-public data like this exists, it can be useful not only for citizens who want to analyze it using tools such as Kouchou AI, but also as an easy first case for local government staff to test the tool's effects.

That said, although some local governments publish the underlying data from existing surveys, this is still not very common, and Kakogawa City is not the norm. Free-text fields written by residents may include personal information or other data unsuitable for publication, as well as responses such as "nothing in particular" or meaningless answers. Whether to include "nothing in particular" in an analysis will depend on the case. As a result, when this kind of data is released, some preprocessing is normally required. In many local governments, publication stops at analysis reports based on statistical processing or text mining, and it is hard to say that detailed data is being shared with citizens.

### Integration with Decidim

There are use cases around the world where AI is used to make efficient and effective use of opinions gathered on online platforms such as Decidim. In Brazil, for example, where Decidim is being used at the national level, a machine-learning model has been implemented to classify ideas submitted by millions of citizens and recommend similar ones.

Reference: Brasil Participativo
[https://brasilparticipativo.presidencia.gov.br/](https://brasilparticipativo.presidencia.gov.br/)

This suggests one possible direction: integrating Kouchou AI itself into the functions of a platform such as Decidim. Here, I introduce cases CfJ has been working on.

#### Deepening Opinions Submitted Through Decidim: Formulating and Revising Municipal Plans

![Figure: Kouchou AI visualization of comments submitted on Kakogawa City's Smart City Plan, showing opinion clusters across the submitted comments.](images/10_02_03_kakogawa-smartcity-plan.png)

![Figure: Second-level groups generated from comments on Kakogawa City's Smart City Plan, showing how the clustered opinions were organized into broader themes.](images/10_02_04_kakogawa-smartcity-plan-cluster.png)

(Figure 2) Visualization of opinions on Kakogawa City's Smart City Plan and second-level groups

In Kakogawa City, Hyogo Prefecture, the city has used opinions submitted to its Smart City Plan since Decidim was first launched there in 2020, and various measures have been realized as a result, including shared bicycles, bus timetable signage, and online facility reservations. These were realized step by step through opinions submitted at each stage of the plan-formulation process, as well as through sharing implementation status for measures based on the plan and receiving feedback on them.

From there, CfJ tried using Kouchou AI from the perspective that there might be other ideas worth picking up, or different ways of understanding the ideas that had already been submitted. The comments up to that point had been submitted in comment fields aligned with the themes set out in the Smart City Plan. The city hall side had also treated them as opinions on each of those themes. The thought was that integrating them might produce new insights. In a sense, this was an attempt to make active use of the kind of clustering variation observed in the resident awareness survey.

During the analysis process, the opinion groups were adjusted. In particular, there was a comment that using the second-level groups generated by Kouchou AI might make it possible to think about combinations of ideas that had not previously emerged.

Kakogawa City is currently expanding opportunities for dialogue with residents in its community development work around JR Kakogawa Station by combining online opinion collection with in-person workshops. Based on its experience using Kouchou AI, the city is now trying to analyze survey responses about what kinds of facilities and functions are needed with Kouchou AI and reflect them in the community development plan.

Reference: Kakogawa City Decidim
[https://kakogawa.diycities.jp/](https://kakogawa.diycities.jp/)

#### Evolving the Participation Process Through Decidim: Considering New Policies

Shinagawa City in Tokyo positions the reflection of citizen opinions as part of evidence-based policymaking in its city policy direction, and in fiscal 2025 it began using Decidim as a form of policy design carried out together with residents.

As part of this work, the city decided to solicit a wide range of opinions through Decidim in order to create what would be the fourth disaster-prevention charter in Japan. It also designed a participation process that combined online and in-person methods, including surveys of local residents who took part in comprehensive disaster drills and other events, and in-person workshops with foreign residents and junior and senior high school students. The themes for input were also standardized, and ultimately all opinions were posted on Decidim.

This made it possible to integrate and analyze opinions submitted online and in person, while also analyzing each opinion in light of insights gained through face-to-face settings. In using Kouchou AI, the city was able to visualize opinions while preserving the fact that even opinions common across diverse participants still contained internal diversity. For example, even for the same opinion, "Let's greet one another," the clustering was adjusted after concretely understanding that the meaning differed between older residents living in the area and junior high school students.

In this way, the city organized not only the comments themselves but also the underlying feelings and contexts that are difficult to convey on the surface, and ultimately extracted keywords that should be incorporated into the draft disaster-prevention residents' charter.

![Figure: Loop diagram showing the process for creating the Shinagawa Disaster-Prevention Residents' Charter, from input collection through analysis and drafting.](images/10_02_05_shinagawa-process.png)

(Figure 3) Process for creating the Shinagawa Disaster-Prevention Residents' Charter

In addition to the text of the completed Shinagawa Disaster-Prevention Residents' Charter, the city also created "action guidelines" to connect the charter to more concrete behavior. The opinions consolidated through Kouchou AI were also used as a reference in drafting those guidelines.

Reference: Shinagawa Open Talk "Shina-Talk"
[https://shinagawa.makeour.city/](https://shinagawa.makeour.city/)

## Possibilities and Challenges for Use by Local Governments

The cases introduced above show that the value of Kouchou AI is not limited to summarizing free-text responses. When it is properly incorporated into a participation process, it can connect information with different characteristics, such as existing survey results, online posts, and insights from in-person settings, and capture the issues and contexts behind citizens' voices.

Here I would like to point out both the possibilities and the challenges involved in local government use.

### Various Possibilities for Use

From the perspective of local government practitioners, Kouchou AI appears to be valued for qualities not seen in previous analysis methods, such as the "comprehensiveness" with which it gives a broad overview of opinion distribution and the "summarization" of categorized opinions.

For example, reports from various surveys conducted by local governments could become targets for analysis. Such reports usually analyze free-text responses through descriptive statistics, such as "there were X comments of this kind," or through text-mining methods such as word clouds. These approaches are mostly ways to check which words or categories appear in large numbers. They may be accurate, but it is not always easy for readers, whether citizens or staff, to understand what they should take away from the analysis results. By contrast, an analysis like Kouchou AI, which uses generative AI to capture the overall shape of opinions while preserving context from the original free-text responses, may be understood as an extension of previous methods.

Kouchou AI can also be seen as a strong use case for generative AI. As generative AI use has made some progress in local governments, there is growing interest in using it more directly inside actual work, and governments are still exploring what kinds of use cases are possible. For example, there are already cases where local governments outsource analysis work under specifications that require generative AI summaries for several thousand survey responses. But many local governments also handle survey analysis internally rather than outsourcing it, and using Kouchou AI could reduce that workload.

In that context, local governments often have accumulated datasets such as recurring surveys that ask similar questions over time, or large volumes of council meeting minutes. It would therefore likely be useful for Kouchou AI interfaces to make it easy to analyze, for example, how the distribution of written content changes over time.

### What Needs to Be Done to Become Kouchou-AI-Ready

However, when local governments try to use Kouchou AI, they face several practical hurdles. Unless the question of "how can we make this usable?" is resolved before the question of "what should we analyze?", the range of use cases is unlikely to expand.

The current version of Kouchou AI uses external services and external generative AI APIs that assume internet connectivity. For a local government to build the environment quickly on its own, it has to organize many things at once: the specifications of work devices, restrictions on installable software, whether online accounts can be used, connection restrictions caused by three-tier network separation, classification of information assets, and even payment methods.

Among these, the two especially large issues are the hurdle of using cloud services and the hurdle of payment methods. In practice, however, these are not separate questions so much as a combined set of conditions for introduction.

#### Practical Hurdle 1: Cloud Use

Local governments have made progress in organizing their approach to cloud use itself, but in many governments the design of work devices and networks still does not necessarily fit usage patterns that assume external services on the internet. The network configuration currently assumed by local governments, known as "three-tier separation," is a security model that separates the LGWAN-connected environment from the internet-connected environment. Even after revisions to this model, the use of cloud services on the internet still assumes that additional security measures will be taken. That is reasonable, but it adds work.

In addition, Ministry of Internal Affairs and Communications guidelines require approval for use of cloud services, setting security requirements according to the status and handling restrictions of the information involved, and checking the country or region where data is stored. As a result, "just trying Kouchou AI inside city hall" is more difficult than it appears.

One point that requires particular caution is that the input data often contains many free-text responses. Free text can contain personal information or sensitive information, so it cannot always be submitted to an external service as-is. Based on the guidelines, depending on information-asset classifications and internal rules, data may need to be preprocessed through anonymization, masking, or summarization before it can be used online.

Some local governments have begun preparing closed environments or shared-use environments with generative AI use in mind, but at least at present, that cannot be called the general starting point. In practice, a more realistic path is likely to introduce the tool step by step while receiving external support, under an appropriate contract, for providing the usage environment itself and accompanying the operation.

#### Practical Hurdle 2: Credit Card Payment

Many cloud services are used by creating an online account and registering a credit card or similar payment method. So alongside the question of whether the service can be used technically, there is also the question of how to secure a payment method.

On this point, a Ministry of Internal Affairs and Communications notice titled "Points to Note When Having Staff Use Credit Cards for Local Government Expenditures" (February 24, 2021) states that having local government staff use credit cards is not itself in conflict with laws and regulations, while also setting out points to note in operation.

Specifically, issues such as how cloud services can handle estimates and invoices required by local governments, and how authority should be delegated for the payment act itself, are organized as matters for each local government to decide, rather than as legal restrictions. In other words, the notice simply clarified again that credit card payment is not prohibited by law and can originally be made possible through each local government's financial accounting rules, internal-control design, and procedures for card use.

The practical problem is that even for an initial Kouchou AI proof of concept that often costs only a small amount, it may still be difficult to carry out the corporate-card procedures that require a certain amount of administrative cost. This is not something the department that wants to use Kouchou AI and the department responsible for accounting can decide on their own. Given that examples of local government credit card payment are still not very common, many local governments may be interested in Kouchou AI but hesitant to move forward because of these practical barriers.

### How to Design the Analysis Process

On the other hand, because tools such as Kouchou AI face significant practical constraints, some people look for shortcuts to try them quickly, such as having staff pay personally or use personal devices to handle work data. Naturally, I do not recommend this.

In the end, even when using this kind of tool, it is necessary to authorize the constraints within work rules and think through the total design: which department will be responsible, what data will or will not be handled, and which policymaking process the analysis results will be placed into. Many local governments may be struggling with how far they can design this kind of analysis process.

#### Understanding the Analysis Method and Framing the Question

It is also important to remember that natural-language analysis is by no means easy. As the detailed technical issues are introduced in another chapter, simply analyzing a resident survey with Kouchou AI does not mean the analysis has truly been completed.

One common misunderstanding is that data cannot be used if the number of responses is small. But in the cases so far, the question is not simply the number of responses. It is what the government wants to explore in depth, what kind of question the administration has framed, and how much shared understanding exists between residents and government. Receiving diverse opinions is not something that can be satisfied by numbers alone.

Let us step back and consider this from a slightly broader perspective. When the relationship between residents and government is understood through the concept of participation, OECD guidelines divide it into three stages: information, consultation, and engagement. They emphasize that engagement requires a two-way relationship in which necessary information is provided and residents submit opinions on that basis.

Reference: OECD Guidelines for Citizen Participation Processes
[https://www.oecd.org/en/publications/oecd-guidelines-for-citizen-participation-processes_f765caf6-en.html](https://www.oecd.org/en/publications/oecd-guidelines-for-citizen-participation-processes_f765caf6-en.html)

Seen this way, even when applying Kouchou AI to public comment procedures, which local governments often say they need, simply asking "What do you think about X?" will not allow AI to bridge the gap if the administration and residents do not share an understanding of the issue.

Before broad listening, there are often problems with the prerequisite broadcasting: presenting the necessary information and framing the issues. If AI is treated as a silver bullet and applied without reforming the existing policymaking process, the effort may end as a one-off event, or leave behind only the prejudice that "AI is not useful."

#### Participation Architecture

There are probably many cases beneath the surface where visualization and analysis were performed, but the results could not be shared within city hall, did not lead to policymaking or feedback to citizens, and ended with "we tried it once." Even local governments that already analyze surveys internally may hesitate to build the organizational structure needed for staffing, role division, and responsibility for ongoing operation if various institutional constraints prevent them from fully internalizing Kouchou AI use. Compared with the entry point of "we want to try it for now," the number and cost of coordination tasks required to embed the tool in actual work are heavy. This may be common in the early stages of introducing new technology. But it can also be understood as room for growth.

In this regard, there are examples where staff obtained cooperation from colleagues in other departments to build an internal Kouchou AI environment. Initiatives that can be understood as having made use of an internal talent bank for a leading project are highly compatible with digitalization. If this kind of cross-organizational talent development progresses, ripple effects can be expected not only for Kouchou AI but for other digital initiatives as well. From that perspective, developing rules and organizational structures should be treated as matters for decision-making and investment.

For this reason, it is important to involve stakeholders from an early stage. These include the information policy department, staff responsible for information disclosure and personal information protection, the financial accounting department, source departments that hold survey or public input data, and planning departments responsible for policymaking. Both top-down and bottom-up activity matter.

When considered from the perspective of the participatory society that Kouchou AI aims for, where many citizens' opinions are reflected, these points also overlap with the discussion of "participation architecture" that I have presented elsewhere. There, I organized three elements necessary for participation processes: making good use of existing mechanisms, focusing on workflows as processes for handling and processing information, and digital facilitation.

Reference: Kenjiro Higashi, "Resident Self-Governance in a Digital Society," in *Digital Society and Local Governments: The Future of Local Autonomy and Urban Management* (co-authored, Japan Municipal Research Center, 2024)
[https://www.toshi.or.jp/publication/19140/](https://www.toshi.or.jp/publication/19140/)

What is especially important is to review participation systems and rules from the user's perspective, especially by sharing results with citizens in an easy-to-understand way, and to recognize that digital services require redesigning not only the interface but the full workflow, including the back office.

If the local government cases introduced at the beginning are organized using this framework, it becomes clear that whether the participation process has been designed affects how submitted opinions are positioned within internal deliberation and whether the results can be fed back.

|  | Use of existing mechanisms | Workflow as an information-handling and processing process | Facilitation of the participation process using digital tools |
| :---- | :---- | :---- | :---- |
| Use with existing surveys | Analysis using an existing mechanism | Because survey design determines later preprocessing, workload must be taken into account | Insights for internal use; future sharing of results with citizens is needed |
| Deepening opinions submitted through Decidim | Same as above | Insights are gained by combining in-person dialogue and reflecting what was learned there in the analysis | Same as above |
| Evolution of the participation process through Decidim | A new participation design combining surveys, in-person workshops, and online opinion collection | Same as above | Sharing the insights gained through analysis with citizens can build empathy for the measures that are developed |

(Table 1) Relationship between elements of participation architecture and Kouchou AI use cases

These initiatives have not yet produced revolutionary results, but I believe they can steadily build results from here. I hope many local governments will take part in this challenge.
