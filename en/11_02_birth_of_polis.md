# The Birth of Polis

In the previous section, we looked at how Polis came to be used in Taiwan. So how was Polis itself born?

## Lessons from the Arab Spring

From 2010 to 2011, the world witnessed digital political movements on an unprecedented scale. The Arab Spring, which began with Tunisia’s Jasmine Revolution, spread across the Middle East and North Africa; in New York, Occupy Wall Street emerged; and in Spain, the Indignados (“the outraged”) occupied public squares. What these movements shared was that social media acted as a catalyst for spreading information and calling for participation, enabling vast numbers of people to gather without relying on existing political parties or labor unions.

Yet many of these movements, despite their enormous impact, did not lead to lasting institutional change. In the Arab Spring, governments were toppled in Tunisia and Egypt, but the turmoil that followed and the return to authoritarianism make it hard to call the outcome an unqualified “success.”

At the root of the problem was that these loosely connected networks, assembled spontaneously through social media, lacked the capacity for control. Influencers could ignite a movement, but they could not extinguish it. With no leaders, there was no one to negotiate with the government. No bargain could be struck along the lines of “if these demands are met, the movement will end.” People who gathered in anger could force society to recognize a problem, but they did not have the power to solve it. This structural problem will be discussed in more detail in Section 11.5, “Turning Connective Action into Power.”

Colin Megill, who studied international relations, had long been interested in the coordination problems of collective action. During Occupy Wall Street, a web forum prepared by the core group NYCGA (Nycga.net) was used, but it did not function well for large-scale dialogue[^1]. Similar problems surfaced during the Arab Spring, deepening Megill’s conviction that “it is really hard for large numbers of people to communicate effectively”[^2]. Drawing on this experience, Megill developed Polis in 2012 together with Christopher Small and Michael Bjorkegren.

Megill has described the motivation for its development as “addressing the communication challenges in the large-scale distributed movements seen in Occupy Wall Street and the Arab Spring”[^3]. What they aimed to build was a real-time system for “collecting, analyzing, and understanding the opinions of large groups.” After being adopted in Taiwan’s vTaiwan process from 2014 onward, Polis was open-sourced under the AGPL v3 license in 2016 at the request of the Taiwanese government. Today it is operated by a nonprofit organization called The Computational Democracy Project (CompDem)[^4].

## How Polis Works

![Figure: Old Polis interface screenshot showing two opinion groups labeled A and B, participant icons on a scatterplot, and buttons for major opinions, groups, and opinion IDs.](images/11_02_polis_screenshot.png)

Figure: Example Polis screen.[^polis-screenshot]

Polis is built on a design philosophy fundamentally different from that of conventional online petitions such as Change.org. The goal of Change.org is to “gather supporters by the numbers.” Success is measured by maximizing signatures, and the structure tends to reward messages that intensify conflict.

The goal of Polis is to “make the distribution of opinions and points of agreement visible.” Suppose, for example, that Polis is used on the topic of regulating online abuse on social media. When participants open the Polis interface, they first see a single opinion posted by another participant. For example, suppose the statement shown is, “To protect freedom of expression, regulation should be kept to a minimum.” Participants respond by choosing one of three options: “agree,” “disagree,” or “pass.” Once they answer, the next opinion appears, and they respond again with the same three choices. This process repeats.

At the same time, participants can also post their own opinions freely. They can raise new points in short, one-sentence comments such as “To protect victims, platforms should be required to remove harmful content” or “This should be addressed through education rather than regulation.” These submitted opinions are then shown on other participants’ screens, where they receive the same agree/disagree/pass votes.

In this way, Polis uses a **hybrid design that combines open questions (free opinion submission) with closed questions (three-choice voting)**. The fundamental difference from Change.org is that, rather than “gathering supporters by the numbers,” Polis gathers “the reactions of diverse people to diverse opinions.”

This hybrid design reflects the technological constraints of the period in which Polis was created. In 2012, data-science methods such as PCA and clustering had become available as open-source implementations and could be used in real tools. But LLMs that could analyze the content of free-text responses at scale did not yet exist. Polis emerged in the gap of the 2010s, when data science had matured but LLMs had not yet arrived. Instead of analyzing free-text content directly, it treated **voting patterns on free-text comments** as numerical data and used that to make large-scale opinion aggregation possible with the technology of the time.

Participants with similar voting patterns are placed near one another on a scatterplot and automatically color-coded into groups. In this example, clusters such as a “freedom of expression” group and a “victim protection” group would emerge. Polis also identifies opinions supported by both sides of a conflict and highlights them. In the example of regulating online abuse, a statement such as “There should be some way to address malicious anonymous posts” might be supported by both groups. The ability to discover these “points of agreement hidden beneath conflict” is Polis’s essential strength. Audrey Tang calls this “uncommon ground”[^7]—that is, a shared foundation that people assumed they did not have with those on the other side, but in fact did.

Participants can see their own position on the scatterplot and the groups formed by other participants with similar voting tendencies. Both the group close to their own views and groups with different views are visualized as color-coded regions. On ordinary social media, it is easy not to notice that one is inside a “filter bubble,” where algorithms mostly show similar opinions. Through Polis visualization, by contrast, participants can recognize at a glance that people with views different from their own really do exist.

## How Polis Spread

Since its public release in 2012, Polis has been adopted by governments, local authorities, and international organizations around the world, with more than 10 million participants globally in total[^5]. Representative examples are shown below[^6].

| Case | Country | Year | Theme | Participants |
|------|----|----|--------|---------|
| vTaiwan | Taiwan | 2014–ongoing | Nationwide democratic process | 200,000+ |
| Klimarat (Citizens’ Climate Council) | Austria | 2022 | Climate action | 5,000+ |
| “Emergency Bill” referendum | Uruguay | 2020–2021 | Gathering public views on a referendum | 16,000+ |
| UNDP youth dialogue | Bhutan, Pakistan, East Timor | 2020–2021 | Youth and climate action | 30,000+ |
| HiveMind | New Zealand | 2016–2019 | Tax policy, sugar policy, basic income, etc. | 1,700+ |
| Mayors’ consultations | Philippines | 2020–ongoing | Municipal policy consultations | 3,000+ |
| Town hall | United States (Kentucky) | 2018 | Building consensus on local issues | 2,000+ |
| DEMOS survey | United Kingdom | 2020 | Attitudes toward data-driven political campaigns | 997 |
| Aufstehen | Germany | 2018 | Building a political base | 33,547 |
| Airbnb regulation consultations | Greece | 2023 | Solutions to Airbnb-related issues | 944 |
| What Could BG Be? (BG 2050) | United States (Bowling Green, Kentucky) | 2025 | Citizen proposals for a 25-year strategic plan | 7,890 |

The newest notable case is Bowling Green, shown in the last row of the table. Over 33 days, 7,890 residents participated and cast more than one million votes. People involved with CompDem described it as one of the most active dialogues in Polis’s history, making it one of the largest online civic dialogues by a U.S. local city. See Section 11.3 for details.

## How Polis Evolved Technically

From a technical perspective, the 2012 version of Polis was a statistical processing system centered on PCA (principal component analysis) and K-means clustering. It represented participants’ voting data as a matrix and generated a scatterplot of voting tendencies by compressing the data into two dimensions with PCA. K-means automatically detected groups on the scatterplot and statistically identified bridging opinions supported across multiple groups. In other words, it implemented the “numerical analysis of voting patterns” described in the previous section using the open-source data-science methods that were becoming widely available at the time. PCA and K-means are explained in detail in Chapter 12.

Polis has evolved both as a standalone tool and as part of an increasingly integrated ecosystem of other tools. In Taiwan’s vTaiwan, the process of combining Polis voting data with Talk to the City’s (TTTC’s) analysis of free-text responses has been attempted multiple times (see the previous section for details), and an ecosystem is beginning to take shape in which opinions can be understood from both voting data and open-ended responses.

Another important example is **Sensemaker**, the open-source tool developed by Jigsaw, a unit under Alphabet (Google), so that Polis agree/disagree data can be analyzed directly with LLMs. Sensemaker was used in the Bowling Green case discussed in Section 11.3. When a CSV exported from Polis is fed into Sensemaker, the LLM automatically classifies opinions by topic and generates a structured report including points of agreement and disagreement. In other words, an open-source pipeline is emerging that connects Polis’s strength in “opinion collection and voting-pattern visualization” with Sensemaker’s LLM-based “large-scale analysis and summarization of opinion content.”

**Polis 2.0**, whose design preview was published by CompDem in 2024, fundamentally updates the design of Polis on the assumption that LLMs now exist, building on lessons from this kind of integration. EVōC (embedding vector-oriented clustering) vectorizes comment text and groups semantically similar comments together, automatically generating a hierarchy of topics. In effect, this brings an approach similar to TTTC inside Polis itself. LLMs generate real-time summaries and reports of the deliberation as a whole, and the system also includes AI-assisted moderation and multilingual translation. Its architecture and infrastructure have also been redesigned to support simultaneous participation on the scale of millions[^5].

---

[^1]: Liz Barry, “vTaiwan: Public Participation Methods on the Cyberpunk Frontier of Democracy,” Civicist, 2016. https://civichall.org/civicist/vtaiwan-democracy-frontier/
[^2]: GeekWire, “Startup Spotlight: Pol.is uses machine learning, data visualization to help large groups spur conversation,” 2014. https://www.geekwire.com/2014/startup-spotlight-polis/
[^3]: Colin Megill personal website. https://colinmegill.com/
[^4]: Participedia, “Pol.is,” https://participedia.net/method/polis ; The Computational Democracy Project, https://compdemocracy.org/polis/
[^5]: The Computational Democracy Project, “Polis 2.0.” https://pol.is/home2
[^6]: The Computational Democracy Project, “Case studies.” https://compdemocracy.org/Case-studies/
[^7]: “Common ground” is a standard expression in negotiation and dialogue meaning “shared ground” or “a point of agreement.” Tang plays on this phrase by adding “uncommon” (“rare,” “unexpected”), giving it the nuance of “an unexpected shared foundation that cannot be found unless one actively looks for it.”
[^polis-screenshot]: This screenshot is from an older version of Polis. At that time, other participants were shown as individual dots. In the current version, other participants are visualized as group regions with similar voting tendencies rather than as individual points. This book uses the older screen because it conveys the experience of seeing that other people exist in the opinion space.
