## Mapping Public Opinion—Making Public Opinion Visible in Election Campaigns with Polis at the Core

By Yasuo Nishio


“Public opinion” is often forced onto a single axis: right/left, conservative/liberal, for/against. But reality is not that simple.

The same person may stand with you on one issue and take the exact opposite side on another. People’s views do not line up neatly on a straight line; they hold different positions across a range of issues, distributed in complex ways. Could that complex structure be visualized as a “map”?

That question led to *Public Opinion Map*, released during the 2024 House of Representatives election as part of JAPAN CHOICE, one of Japan’s largest election information sites. It launched on November 18, 2024, and in about two weeks, 4,403 unique users cast votes. Because it uses Polis internally as a component, it can be considered one of the largest applications of Polis in Japan. Links to the demo page, explanatory article, and press releases are collected in the footnotes[^beta][^note][^press].

[^beta]: Public Opinion Map (beta) demo page: [https://japanchoice.jp/polis](https://japanchoice.jp/polis)
[^note]: Explanatory article (Mielka note): [https://note.com/mielka/n/n54313c84a5e5](https://note.com/mielka/n/n54313c84a5e5)
[^press]: Press releases (JAPAN CHOICE update / full feature release): [https://prtimes.jp/main/html/rd/p/000000013.000029162.html](https://prtimes.jp/main/html/rd/p/000000013.000029162.html) / [https://prtimes.jp/main/html/rd/p/000000014.000029162.html](https://prtimes.jp/main/html/rd/p/000000014.000029162.html)

I was involved in this development as an engineer.  
What I want to write about in this chapter is not simply, “We used Polis.”

In this chapter, I record what had to be changed in order to use Polis in the real-world context of Japanese elections, and what became visible as a result.

## How It Started

The story goes back to January 13, 2024.

Plurality Tokyo in April 2023 was held at the SmartNews office, where readers devoted to Ken Suzuki’s *A Smooth Society and Its Enemies* gathered around him. A study group emerged that aimed to use that book as a core text while connecting it to contemporary technology and knowledge. This later came to be known as the “Smooth Conference.” On January 13, 2024, the third Smooth Conference was held, and Toki Yuki, Representative Director of the nonprofit Mielka, which operates JAPAN CHOICE, participated and gave a talk. The discussion there became animated around the appeal of Polis, and a plan was launched to use Polis on JAPAN CHOICE in conjunction with the House of Representatives election in autumn 2024.

## The Relationship Between Polis and Voting Advice Applications

Polis, which has been used in Taiwan’s digital democracy for more than a decade, is a system that visualizes the distribution of opinion by having users vote agree/disagree on a set of statements.

The first half of that process—collecting agreement and disagreement on statements—is structurally the same as conventional voting advice applications. Traditional voting advice applications would take that information and display the result as a one-dimensional ranking: “The party closest to your views is Party X.” With Polis, by contrast, “your views” are plotted within a shared opinion space, and “party positions” are plotted within that same space as well. This makes it possible to see, in a two-dimensional space, which party your views are closest to.

I felt that this experience—seeing yourself and political parties positioned together within a shared opinion space—could be extremely compelling. But to make that experience possible, we could not simply use Polis as-is.

## Design Changes for Japanese Elections

I joined the discussions together with Shunsuke Takagi[^takagi], who is also one of the founders of Plurality Tokyo. As a result of those discussions, we decided at an early stage not to pursue a path of using Polis unchanged, and instead shifted to using Polis as a component. Below I explain the main changes.

### Working Backward from the Assumption That 85% of Users Are on Mobile

As an election information site, JAPAN CHOICE has an overwhelmingly high mobile usage rate. Of its roughly 3 million unique users, 85% are on mobile. When we checked Polis as it existed at the time on a smartphone screen, it became clear that it did not meet a satisfactory standard.

So we rebuilt the frontend from scratch to match Japanese usage patterns. This was what is commonly called mobile-first development: a design approach that assumes from the outset that users will be accessing the service on the narrow screens of smartphones.

We also decided to localize the UX so that no English would appear at all, given the Japanese context in which many users cannot read English, and many others will simply disengage if English appears on screen.

This decision to “rebuild the frontend” carried a high engineering cost, but it also made a high degree of experimentation possible.

## The Core Value Was the “Party Icons”

What we wanted to achieve with Public Opinion Map was not simply to “visualize a cloud of citizen points.” We wanted to overlay political parties on top of the distribution of citizen opinion.

![party icons](images/04_04_polis_party_icons.png)

The opinions users voted on were extracted from each party’s electoral platform. We then encoded whether each party’s position on each statement was in favor, opposed, or neutral, and displayed those positions as party icons on the scatterplot. This allowed users to see which party their own views were closest to, what kinds of opinion groups existed in society and which parties represented them, and which groups were not represented by any party at all.

![economic policy map](images/04_04_polis_economic_map.png)

This is a visualization of opinion groups on economic policy. You can see a cluster in the lower right with no party icon attached. (Details follow below.)

## Choosing to Disable Open Submission in Order to Maintain Quality

And in order to make that value proposition work, we were forced into some difficult design decisions.

In Polis, participants can freely submit their own statements. From the standpoint of deliberative ideals, this means opening up agenda-setting power to the public. If possible, we would have liked to create a space where free posting was allowed. But political parties are not going to vote agree/disagree on statements freely submitted by users. So if people keep posting new opinions, the party icons end up being visualized as “users who answered almost none of the questions.” That would destroy the value of displaying party icons in the first place.

There was another issue as well. User-submitted statements would likely include comments praising or attacking specific parties. Showing such content directly to other users would undermine the value of the site as a politically neutral source of election information. It would likely require moderators to review submissions carefully before publication. That is costly. Even in Taiwanese Polis operations, there has been discussion about the burden on moderators becoming too heavy.

For the 2024 release of Public Opinion Map, we therefore adopted a design that did not accept user-submitted opinions. Participants could only vote on statements prepared by the operators. It was a painful decision.

Of course, most users do not think of a voting advice site as a place to post their own opinions. Nor do they come expecting to vote agree/disagree on statements written by some unknown person on the internet rather than curated by the operators. In other words, people who come to a voting advice site are not expecting it to be a space for deliberation.

This suggests that Polis contains elements that fit well with voting advice applications and elements that do not. Its aspect as a deliberative platform with free posting was not a good fit for voting advice. Rather than aiming for an all-in-one ideal from the start, it is important to take a first step where implementation is feasible. Someday, when Japanese users are more accustomed to deliberation, I would like to build a deliberative platform that opens agenda-setting power to the public and observe what happens.

## Statement Extraction Was Done by Humans

The foundation of Public Opinion Map is the constraint that its statements are “derived from party electoral platforms.” So how were those statements extracted? Could AI do it easily?

In an election context, hallucinations are fatal. We ran experiments at the time of the 2024 House of Representatives election and the 2025 House of Councillors election, but judged the quality insufficient and did not adopt the approach. This is also an area where JAPAN CHOICE’s long-term experience and know-how—built through continuously operating voting advice applications since the 2012 House of Representatives election—becomes a real strength.

Of course, by 2030, we may well have a division of labor in which AI instantly produces a draft and humans carefully review whether it is correct.

### Computing Coordinates on the Client

In Public Opinion Map, your position on the scatterplot moves immediately in response to your votes. This is an experience Polis itself does not offer. Immediate feedback on the results of one’s actions improves the quality of the experience.

We achieved this by also providing the client with the projection matrix used in principal component analysis (PCA). The user’s voting vector is projected into two dimensions using that matrix, allowing the position to be updated without querying the server.

This design has an additional benefit: even if the Polis computation server is shut down, positions can still be updated in response to voting. We have in fact shut down the Polis computation server now that the election period is over. Even so, users can still vote their views and experience their position on the map updating. The fact that it can remain available after the election as a kind of “dynamic exhibit” was also an advantage from the standpoint of public value.

### Using AI to Generate Explanations of Opinion Groups

The original Polis display of cluster information was difficult for people unfamiliar with statistics or data analysis to interpret. So we decided to pass the “opinions that characterize a cluster” to an LLM and generate a short label and explanatory text to display below the scatterplot. I created this with reference to the explanation-generation phase of *Talk to the City*.

However, LLMs can misread numbers. For example, suppose a cluster is characterized by “within the cluster, 70% agree and 10% disagree; outside the cluster, 90% agree and 3% disagree.” In that case, what distinguishes the cluster from others is that it has “relatively more opposition.” But an LLM may summarize it as “there is a lot of support.” If the process of generating and publishing explanations were fully automated, such mistakes would make it into the public sphere.

So during the election period, we chose not to update in real time, but instead to update only after human review.

## What the Party Icon Display Revealed

The defining feature of Public Opinion Map was that it overlaid party icons on top of the distribution of citizen opinion. Polis is used around the world, but to my knowledge there had been no example of visualizing party positions within the same coordinate system. This design decision brought into view things that conventional voting advice applications could not show.

### The “Political Blank Space” Revealed by Visualization

On the map of economic policy, we observed a cluster with no party icon attached. In other words, the views of the people in that cluster may not be adequately represented by existing parties. This is different from simply saying, “There are people with no party they support.” Rather, it suggests that the parties themselves do not possess enough diversity to cover the full range of public opinion.

![economic policy map](images/04_04_polis_economic_map.png)
Repeated here: visualization of opinions on economic policy

This figure visualizes opinion groups on economic policy. You can see a cluster in the lower right with no party icon attached. In other words, the views of this group are not well represented by existing political parties. In representative democracy, the public chooses legislators who represent their views. Implicit in that is the assumption that “among the available choices, there are legislators who represent our views.” The data suggested that this assumption may not hold.

The AI-generated explanation for this group was as follows:

> Team Yellow: Cautious on Price Pass-Through  
> 🤖 Compared with other groups, this team shows a relatively cautious stance toward achieving wage increases by supporting small and medium-sized enterprises in passing on higher costs to prices. It also clearly opposes reducing or abolishing the consumption tax and is skeptical of cash benefits for low-income households.

The political blank space captured by Public Opinion Map is analogous, in private-sector product development terms, to a situation where “you know customers exist, but no competitor is offering the product those customers want.”

One of the defining features of this cluster was its clear opposition to cutting or abolishing the consumption tax. One year after this analysis, in the 2026 House of Representatives election, Team Mirai (meaning “Team Future”) alone took a position opposing a consumption tax cut. It is impossible to verify how much that contributed, but the party went on to win what could fairly be called a major breakthrough in seats. In this way, discovering opinion groups that other parties do not represent may offer a valuable opportunity—especially for smaller parties—to attract attention and support.

## Conclusion: Toward a Public Good Through Open Data and Iteration

On May 8, 2025, the voting data from Public Opinion Map 2024 was released as open data and open source. The CSV files and images are provided under CC BY 4.0[^opendata].

[^takagi]: He later founded Tagen Genjitsu LLC as its representative and has worked on the social implementation of broad listening technologies. Tagen Genjitsu is discussed in detail in Chapter 11.
[^opendata]: Public Opinion Map 2024 open data: [https://github.com/mielka/yoronchizu2024-data](https://github.com/mielka/yoronchizu2024-data)

I believe this means more than simply “distributing the output.” Every election brings many events and developments, but once the election is over, they are forgotten and disappear. I want to build a verifiable, reusable “infrastructure for democracy” that exists outside that cycle. I want a mechanism that accumulates and improves little by little with each election.

The release of Public Opinion Map during the 2024 House of Representatives election was only a first experiment. Questions remain: whether to allow free posting, how to incorporate LLM support, how to semi-automate statement extraction. The challenges are still there.

But I am certain of one thing.  
If we had simply imported Polis and used it unchanged, we would not have achieved these results. What matters is adapting, rebuilding, and improving it to fit the realities on the ground. And the record of those adaptations is itself practical knowledge for the next implementer.
