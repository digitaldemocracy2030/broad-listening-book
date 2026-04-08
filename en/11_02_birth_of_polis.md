MEMO: Extracted from the column on collective action and connective action. Needs expansion TODO: screenshots, a screenshot of Polis, an explanation of how Polis works, and the addition of a schematic diagram

In the previous section, we looked at how Polis came to be used in Taiwan. So how was Polis itself born?

## The Birth of Polis: What the Arab Spring Raised

From 2010 to 2011, the world witnessed digital political movements on an unprecedented scale. The Arab Spring, which began with Tunisia’s Jasmine Revolution, spread across the Middle East and North Africa; in New York, Occupy Wall Street emerged; and in Spain, the Indignados (“the outraged”) occupied public squares. What these movements shared was that social media acted as a catalyst for spreading information and calling for participation, enabling vast numbers of people to gather without relying on existing political parties or labor unions.

Yet many of these movements, despite their enormous impact, did not lead to lasting institutional change. In the Arab Spring, governments were toppled in Tunisia and Egypt, but the turmoil that followed and the return to authoritarianism make it hard to call the outcome an unqualified “success.”

At the root of the problem was that these loosely connected networks, assembled spontaneously through social media, lacked the capacity for control. Influencers could ignite a movement, but they could not extinguish it. With no leaders, there was no one to negotiate with the government. No bargain could be struck along the lines of “if these demands are met, the movement will end.” People who gathered in anger could force society to recognize a problem, but they did not have the power to solve it. This structural problem will be discussed in more detail later in “Connective Action and Institutional Linkage.”

Colin Megill, who studied international relations, had long been interested in the coordination problems of collective action. During Occupy Wall Street, a web forum prepared by the core group NYCGA (Nycga.net) was used, but it did not function well for large-scale dialogue[^1]. Similar problems surfaced during the Arab Spring, deepening Megill’s conviction that “it is really hard for large numbers of people to communicate effectively”[^2]. Drawing on this experience, Megill developed Polis in 2012 together with Christopher Small and Michael Bjorkegren.

Megill has described the motivation for its development as “addressing the communication challenges in the large-scale distributed movements seen in Occupy Wall Street and the Arab Spring”[^3]. What they aimed to build was a real-time system for “collecting, analyzing, and understanding the opinions of large groups.” In 2016, at the request of the Taiwanese government, Polis was open-sourced, and it is now operated by a nonprofit organization called The Computational Democracy Project (CompDem)[^4].

### How Polis Works: A System for Discovering Points of Agreement

Polis is built on a design philosophy fundamentally different from that of conventional online petitions such as Change.org. The goal of Change.org is to “gather supporters by the numbers.” Success is measured by maximizing signatures, and the structure tends to reward messages that intensify conflict.

The goal of Polis is to “make the distribution of opinions and points of agreement visible.” Suppose, for example, that Polis is used on the topic of regulating online abuse on social media.

1. **Posting opinions**: Participants write and submit their views in a text box. These are short, one-sentence comments such as “To protect freedom of expression, regulation should be kept to a minimum” or “To protect victims, platforms should be required to remove harmful content.”
2. **Voting**: Other participants’ comments are shown one at a time on the screen, and users vote on each with “agree,” “disagree,” or “pass.” As all participants vote on all comments, the system accumulates a full dataset showing who agreed or disagreed with which statements.
3. **Visualizing groups**: People with similar voting patterns are placed near one another on a scatterplot and automatically color-coded into groups. In this example, clusters such as a “freedom of expression” group and a “victim protection” group would emerge.

The fundamental difference from Change.org is that, rather than “gathering supporters by the numbers,” Polis gathers “the reactions of diverse people to diverse opinions.”

Polis also identifies opinions that are supported by both sides of a conflict and highlights them. In the example of regulating online abuse, a statement such as “There should be some way to address malicious anonymous posts” might be supported by both the freedom-of-expression group and the victim-protection group. The ability to discover these “points of agreement hidden beneath conflict” is Polis’s essential strength. Audrey Tang calls this “uncommon ground”[^7]—that is, a shared foundation that people assumed they did not have with those on the other side, but in fact did.

Each time participants vote, they can see their position on the scatterplot update in real time. On ordinary social media, it is easy not to notice that one is inside a “filter bubble,” where algorithms mostly show similar opinions. In Polis, by contrast, the existence of groups with views different from one’s own is made visually explicit. At a glance, participants can recognize that people with a wide range of opinions really do exist.

### The Spread of Polis

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

### The Technical Evolution of Polis

From a technical perspective, the 2012 version of Polis was a statistical processing system centered on PCA (principal component analysis) and K-means clustering. It represented participants’ voting data as a matrix and generated a scatterplot of voting tendencies by compressing the data into two dimensions with PCA. K-means automatically detected groups on the scatterplot and statistically identified bridging opinions supported across multiple groups. In an era before LLMs existed, this was an innovative approach: visualizing the structure of opinion not through the meaning of comments, but through mathematical analysis of voting patterns alone. PCA and K-means are explained in detail in Chapter 12.

Polis has not evolved only as a standalone tool; it has also increasingly been integrated with other tools. In Taiwan’s vTaiwan, the process of combining Polis voting data with TTTC’s analysis of free-text responses has been attempted multiple times (see the previous section for details), and an ecosystem is beginning to take shape in which opinions can be understood from both voting data and open-ended responses.

**Polis 2.0**, announced by CompDem in 2024, fundamentally updates the design of Polis on the assumption that LLMs now exist, building on lessons from this kind of integration. EVōC (embedding vector-oriented clustering) vectorizes comment text and groups semantically similar comments together, automatically generating a hierarchy of topics. In effect, this brings an approach similar to TTTC inside Polis itself. LLMs generate real-time summaries and reports of the deliberation as a whole, and the system also includes AI-assisted moderation and multilingual translation. Its architecture and infrastructure have also been redesigned to support simultaneous participation on the scale of millions[^5].

---

[^1]: Liz Barry, “vTaiwan: Public Participation Methods on the Cyberpunk Frontier of Democracy,” Civicist, 2016. https://civichall.org/civicist/vtaiwan-democracy-frontier/
[^2]: GeekWire, “Startup Spotlight: Pol.is uses machine learning, data visualization to help large groups spur conversation,” 2014. https://www.geekwire.com/2014/startup-spotlight-polis/
[^3]: Colin Megill personal website. https://colinmegill.com/
[^4]: Participedia, “Pol.is,” https://participedia.net/method/polis ; The Computational Democracy Project, https://compdemocracy.org/polis/
[^5]: The Computational Democracy Project, “Polis 2.0.” https://pol.is/home2
[^6]: The Computational Democracy Project, “Case studies.” https://compdemocracy.org/Case-studies/
[^7]: “Common ground” is a standard expression in negotiation and dialogue meaning “shared ground” or “a point of agreement.” Tang plays on this phrase by adding “uncommon” (“rare,” “unexpected”), giving it the nuance of “an unexpected shared foundation that cannot be found unless one actively looks for it.”
