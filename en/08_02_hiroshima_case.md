## Hiroshima Prefecture: A Prefecture of “Learning from Failure” Takes On the Challenge of Building Systems to Hear People’s Voices

Written by: @tokoroten

English translation by Luke Closs

This section is based on the Hiroshima AI Lab annual report, records of responses in the Hiroshima Prefectural Assembly, materials published on the prefecture’s official website, and interviews with Hiroshima AI Lab staff.

### The Cost of “Reading” Was Too High to Put Diverse Voices to Use in Policymaking

Government receives a wide range of public input: public comments, free-response survey answers, and social media posts. But the work of reading, classifying, and summarizing those voices into material that can inform policy is enormous. According to comments from staff recorded in Hiroshima Prefecture’s annual report, “in the past, we spent several days classifying responses by hand and preparing briefing materials”[^hiroshima_report].

In September 2024, Hiroshima Prefecture announced the “Hiroshima Declaration: Opening the Future with AI” and established the Hiroshima AI Lab as an in-house hub for AI exploration. Its slogan was “HIROSHIMA AI TRIAL — Learn from Failure.” As one of the lab’s exploratory activities, the prefecture piloted broad listening using DD2030’s Public Voice AI, driven by the goal of understanding diverse resident needs and reflecting them in policy development.

Before launching a public call for opinions, the Hiroshima AI Lab had already conducted around 20 internal analyses. It fed in inquiry logs from the “Migration AI Navi” chatbot for people interested in relocating to identify patterns. It also tried running participant comments from youth-oriented local events through Public Voice AI without even knowing “how they should be categorized,” and discovered “this is another way to group them.” The lab also analyzed 4,924 posts on X related to the Hiroshima governor election (voting and counting in November 2025), gaining practical knowledge about prompt design to filter out social-media-specific noise such as promotional posts and reposted news. Through this process of trial and error, the Hiroshima AI Lab moved on to broad listening aimed at prefectural residents.

### “What Kind of Future Hiroshima Can We Create Through Digitalization?”

In August 2025, Hiroshima Prefecture launched a public call for opinions titled “What Kind of Future Hiroshima Can We Create Through Digitalization?”[^hiroshima_mirai] The question posed was: “How would you like work and daily life to change through digital technology?” The idea was to set aside feasibility for the moment and invite exciting, free-form ideas about the future. Five channels were provided—web form, X (using the hashtag #広島デジタル), email, postal mail, and telephone—and free-response submissions were accepted over roughly one month.

In the end, 2,953 submissions were collected. Public Voice AI analyzed them and extracted 3,179 opinions in about 15 minutes[^hiroshima_bl_site]. On September 3, during the collection period, the prefecture published interim results on its website—something that would have been impossible under the old manual process.

![Public Voice AI analysis results for “What Kind of Future Hiroshima Can We Create Through Digitalization?” A total of 2,953 opinions are classified and visualized into 15 clusters](images/08_02_hiroshima_broadlistening_result.png)
*Source: Broad Listening Hiroshima visualization site*

At the same time, challenges remained on the collection side. In interviews, prefectural staff reflected that “at first, we wanted to gather opinions from social media and younger people, but we weren’t able to get many.” Because the outreach was conducted as an extension of conventional survey methods, respondents appear to have skewed toward people already close to government, such as staff and their families, and did not reach social media users or younger demographics very effectively.

The results reflect this. The largest cluster was “Digital Coexistence for Seniors and Local Communities” (347 responses), but overall, clusters related to administrative and everyday procedures stood out: “More Flexible and Efficient Administrative Procedures” (183), “More Efficient Everyday Procedures Through My Number Utilization” (160; My Number, Japan’s national ID system), “Online Access to Everyday Procedures” (159), “Simplified Child-Rearing Procedures and Better Information Provision” (128), “Promoting Online Voting” (114), and “Digitalization for More Efficient Work and Daily Life” (95). Administrative and procedural topics alone accounted for more than a quarter of all responses.

The call was intended to gather visions for digitalization across a wide range of areas—work, daily life, communities, and government—but because respondents skewed toward people closer to government, that bias also appeared in the distribution of opinions. As for opinions submitted on X using the hashtag “#広島デジタル,” there were only two[^hiroshima_x_hashtag]. **Who you reach determines what comes back.** As discussed in Chapter 2, broad listening is not a tool for “collecting opinions” but for “structuring the opinions that have been collected.” Gathering opinions requires separate mechanisms, and the design of collection itself is what determines whether broad listening succeeds or fails.

### Internal Use and the Impact on Efficiency

Although there were challenges in collection, the 2,953 opinions gathered were steadily put to use within the prefectural government. The analysis results were not merely reported to the General Affairs Committee of the prefectural assembly. They were also shared at the “DX Promotion Headquarters Meeting,” attended by bureau directors and the governor, and each bureau was asked, cluster by cluster, “How are you currently responding?” and “How do you plan to address this going forward?” The prefecture also published the visualization site, showing a commendable commitment to transparency.

So how much efficiency was gained? The following is based on assembly responses[^hiroshima_gikai_cost] [^hiroshima_gikai_voice] and the annual report[^hiroshima_report].

According to staff, “we were able to classify 3,000 free-response entries in under 30 minutes, and we clearly felt the efficiency gains.” In this Public Voice AI workflow, staff spent about 10 hours total reviewing all opinions and adjusting prompts. Had the same work been done manually, it is estimated that it would have taken about 500 hours in total (around 62 person-days) and several months to half a year before publication.

What changed was not just speed. One staff member also noted that “there is much less mental burden in looking at mapped opinions after classification than in reading scattered opinions one line at a time in CSV or Excel format.” The experience of analysis itself had changed.

Among the submitted opinions was one from a university student: “If I could get a smartphone notification five minutes before garbage collection time, I could sleep a little longer.” The director of the DX Promotion Division commented, “We had never thought of combining garbage disposal and sleep, but from a different perspective, it connects to making better use of time, so we took it as an interesting idea that could benefit everyone, not just university students.” The value of broad listening lies precisely in surfacing these kinds of unexpected voices.

### The Biggest Gain Was Learning How to “Train” Prompts

The most important thing Hiroshima Prefecture gained was practical knowledge about how to “train” Public Voice AI outputs so they fit the realities of municipal work. Prefectural staff identified two key points for adjustment.

**First, the question of how far to go in “extracting” opinions.** When AI fills in context for short posts, it can generate plausible-sounding opinions. But those are not necessarily what the original poster meant. With the standard prompt, even vague posts were turned into “something that sounds like an opinion,” but the staff-created prompt was able to incorporate the judgment that “items that do not constitute an opinion should be excluded.” Below is an excerpt from comparison examples recorded in the annual report[^hiroshima_report].

| Original opinion | Result with standard prompt | Result with staff-created prompt |
|---|---|---|
| Applying for large-item garbage pickup | Information is needed regarding applying for large-item garbage pickup. | I would like the application process for large-item garbage pickup to be digitalized so that procedures can be completed easily. |
| I don’t know | Because the question or situation is unclear, more detail needs to be provided. | (Excluded) |
| I would like roads improved, bridges enhanced, and digitalization in municipalities promoted as soon as possible, and I also think more public tourist facilities are still needed. | 1) Road improvement, bridge enhancement, and promotion of digitalization in municipalities should be advanced as soon as possible.<br/>2) More public tourist facilities are still needed.<br/>→ Split into 2 items | 1) I would like road improvement and bridge enhancement to be advanced as soon as possible.<br/>2) I would like digitalization in municipalities to be promoted as soon as possible.<br/>3) I think more public tourist facilities are still needed.<br/>→ Split into 3 items |

*Source: Excerpted from Table 4-7 of the FY2025 Hiroshima AI Lab Annual Report*

The standard prompt forced meaning onto even a post like “I don’t know,” whereas the staff-created prompt could exclude items that did not amount to an opinion. It also handled posts containing multiple issues differently: while the standard prompt split one such post into two items, the staff-created prompt split it into three according to distinct areas of daily life, allowing for finer-grained issue separation.

**Second, the problem of translating cluster labels into language government can actually use.** The standard prompt generated more technical labels such as “Utilization of Digital Technology” and “Development of ICT Infrastructure.” But the language used in internal government meetings is framed around areas of daily life such as “transportation,” “education,” “tourism,” and “welfare.” By shifting labels away from technical jargon and toward everyday policy domains, the analysis results became easier to connect to the prefecture’s policy framework and easier to put on the table for discussion.

| Label from standard prompt | Label from staff-created prompt |
|---|---|
| Improving efficiency and convenience in child-rearing and educational environments through digital technology | Digitalization of education and child-rearing |
| Integrated proposals for improving transportation infrastructure and digital technology in Hiroshima Prefecture | Future-oriented transportation and mobility support in Hiroshima |
| Strategies for regional revitalization and digital technology utilization in Hiroshima Prefecture | Hiroshima’s appeal and regional revitalization |

*Source: Quoted from Table 4-9 of the FY2025 Hiroshima AI Lab Annual Report*

With the standard prompt, 16 of the 20 top-level cluster labels included the word “digital,” and the average label length was 25 characters. With the staff-created prompt, only 6 of 20 included “digital,” and the average length was shortened to 14 characters. By bringing everyday policy domains to the forefront, the results became easier to connect to internal policy discussions.

Most of the roughly 10 hours reportedly spent on prompt adjustment in the assembly response mentioned earlier went into this trial and error around “extraction” and “translation.” It may seem modest, but this kind of practical know-how is exactly what will be most useful to other local governments introducing Public Voice AI.

That said, analysis results can change depending on prompt design, and the user’s intentions can be reflected in the outcome. The annual report itself warns that “residents’ voices must not be arbitrarily selected or altered,” and recommends publishing the prompts and LLMs used in analysis so that third-party verification is possible[^hiroshima_report]. Hiroshima Prefecture’s decision this time to publish the visualization site and include prompts in the annual report can be seen as a step in that direction.

### It Matters Because They Run the Process Themselves

Adjusting prompts, checking outputs, and repeatedly asking, “Will this way of grouping make sense internally?” and “Does this extraction distort the original opinion?”—only staff themselves, who understand both the policy field and the internal context of government, can run that cycle. If analysis is outsourced and all that comes back is a report, this kind of accumulated adjustment does not remain within the organization.

By lowering the cost of reading through Public Voice AI, staff can spend more time on more essential questions: “How should we reinterpret this?” and “What should we explore further?” The important point is not simply that costs went down, but that **things that were previously impossible because the cost was too high have now become possible**.

### The Hiroshima AI Lab’s Next Move

Challenges remain on the collection side, especially the fact that outreach to social media users and younger people was not sufficient. At the same time, however, the prefecture has steadily advanced through around 20 internal trials, accumulated know-how in prompt adjustment, and connected the work to decision-making processes. True to the Hiroshima AI Lab slogan, “Learn from Failure,” the issues encountered this time are also lessons that can be applied in the next round of practice.

And Hiroshima Prefecture now has a strong ally. In January 2026, Hiroshima Prefecture signed a “Basic Agreement on Coordination and Cooperation for Promoting the Use of AI” with the Tokyo Metropolitan Government and GovTech Tokyo[^hiroshima_govtech]. GovTech Tokyo has already used broad listening to analyze about 28,000 opinions from Tokyo residents in the “New Tokyo 2050” project introduced in Chapter 5. On the two issues Hiroshima identified this time—“mechanisms for collection” and “connection to policy”—Tokyo has already accumulated practical knowledge.

At the signing of the agreement, Governor Yokota of Hiroshima said, “We will refine a new model for local government from the regions and share it nationwide.” The system for reflecting residents’ voices in policy is still under development. But by combining Tokyo’s experience as a point of reference with the practical knowledge Hiroshima has built through its own trial and error, the prefecture’s next public call for opinions is likely to look very different from this one. The prefecture’s slogan, “Learn from Failure,” is now about to be tested in the context of broad listening.

---

[^hiroshima_mirai]: Hiroshima Prefecture, “What Kind of Future Hiroshima Can We Create Through Digitalization?” (public comment page) https://www.pref.hiroshima.lg.jp/site/hiroshima-dx-torikumi/hiroshimanomirai.html
[^hiroshima_bl_site]: Broad Listening Hiroshima visualization site https://www.broadlistening-hiroshima.com/
[^hiroshima_gikai_cost]: Hiroshima Prefectural Assembly, General Affairs Committee (September 25, 2025), response by the Director of the DX Promotion Division (processing time and division of roles) https://www.pref.hiroshima.dbsr.jp/index.php/4605175?Template=document&VoiceType=all&VoiceID=58603#one
[^hiroshima_gikai_voice]: Hiroshima Prefectural Assembly, General Affairs Committee (September 25, 2025), response by the Director of the DX Promotion Division (specific examples of opinions) https://www.pref.hiroshima.dbsr.jp/index.php/4605175?Template=document&VoiceType=all&VoiceID=58599#one
[^hiroshima_report]: FY2025 Hiroshima AI Lab Annual Report https://www.pref.hiroshima.lg.jp/uploaded/attachment/658524.pdf
[^hiroshima_govtech]: Hiroshima Prefecture, Tokyo Metropolitan Government, and GovTech Tokyo, “Basic Agreement on Coordination and Cooperation for Promoting the Use of AI” (signed January 2026) https://www.pref.hiroshima.lg.jp/soshiki/265/aikyotei.html
[^hiroshima_x_hashtag]: Search results for the hashtag “#広島デジタル” on X (confirmed by the author as of March 2026) https://x.com/hashtag/%E5%BA%83%E5%B3%B6%E3%83%87%E3%82%B8%E3%82%BF%E3%83%AB?src=hashtag_click
