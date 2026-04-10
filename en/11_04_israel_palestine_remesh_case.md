# The Remesh Case in Israel and Palestine

## An Attempt to Connect Two Sides Thought to Be Beyond Dialogue

On October 7, 2023, Hamas’s attack on Israel pushed the Israel–Palestine conflict into a new phase. From that day forward, the channels of communication that had existed between the two sides were cut off one after another. Even among activists who had long worked on peacebuilding, trust collapsed, and physical contact became impossible. The language barrier between Hebrew and Arabic further separated the two sides. It was, in every sense, a situation in which dialogue seemed impossible.

And yet, amid this despair, the **AI Pulse** project sought to connect Israeli and Palestinian peacebuilders—NGO members working for peace on both sides—and produce a shared **statement**. The project was organized by ALLMEP (the Alliance for Middle East Peace), a network of more than 180 peacebuilding organizations active in Israel and Palestine[^1][^3]. Conducted from April to July 2024, it reportedly involved 138 NGO members engaged in peacebuilding and achieved a high agreement rate of over 92%[^1].

The dialogue platform used was an AI-based discussion system provided by the U.S. company Remesh. In 2023, Remesh received a $100,000 grant from OpenAI’s “Democratic Inputs to AI” program and developed a democratic dialogue system using LLMs[^4].

This section explains the project based on the paper “Using Collective Dialogues and AI to Find Common Ground Between Israeli and Palestinian Peacebuilders” (Konya et al., 2025, https://arxiv.org/abs/2503.01769).

---

## Project Design and Technical Innovation

### The Overall Flow of the Dialogue Process

![Architecture of the dialogue process](images/11_04_対話プロセスアーキテクチャ.png)
*Figure 1: The four dialogue cycles and the processing pipeline within each cycle (Source: Konya et al., 2025[^1], CC BY 4.0)*

As shown in the figure, the project consisted of four dialogue cycles. The first three cycles formed the Uninational Phase, in which each group held discussions separately. The fourth and final cycle was the Joint Phase, in which all groups participated together.

Within each cycle, the following processing pipeline was carried out:

1. **Collective Dialogue** — Participants freely entered text and shared opinions in real time
2. **Bridging-based Ranking** — An algorithm identified “bridging statements,” or views supported by both sides of a conflict
3. **Refinement by an LLM** — GPT-4 generated concise, clear “collective statements” while preserving participants’ wording
4. **Human expert review** — Native speakers checked for cultural appropriateness and accuracy
5. **Final participant vote** — Participants voted on the refined statements to measure the level of agreement

Through this five-stage pipeline, participants’ raw voices were transformed into validated collective statements. By combining AI with human expertise, the project aimed to achieve both technical efficiency and cultural appropriateness.

### The Phased Dialogue Format (Uninational Phase): Ensuring Psychological Safety

The project’s most important design innovation was the introduction of a **phased dialogue format**. If two sides in a severe conflict are brought together from the outset, there is a high risk that the exchange will devolve into emotional confrontation rather than constructive dialogue. To avoid this, the project first created a preparatory stage called the **Uninational Phase**, designed to ensure participants’ psychological safety.

The 138 participants, all NGO members engaged in peacebuilding, were divided into the following three groups and held **separate discussions**:

- **Cycle 1**: Israeli Jews (about 97 participants)
- **Cycle 2**: Palestinians from the West Bank and Gaza (28 participants)
- **Cycle 3**: Palestinian citizens of Israel (13 participants)

By gathering only with others from their own side, participants could candidly articulate their own “red lines” and their vision of a desired future without fear of or deference to the other side. Only after building shared understanding within each group did the process move to **Cycle 4, the Joint Phase**, in which all groups came together.

At this preparatory stage, the participants worked to define the **boundary conditions under which dialogue itself could take place**. The paper defines red lines as “actions (including speech) that could cause dialogue itself to break down”[^1]. Each group therefore clarified both the “outgroup red lines”—lines they did not want the other side to cross—and the “ingroup red lines”—lines they themselves would not cross.

| Position | Lines they did not want the other side to cross<br>(outgroup red lines) | Lines they themselves would not cross<br>(ingroup red lines) |
|------|------|------|
| **Israeli Jews** | - Harming Israeli civilians<br>- **Justifying the October 7 attack**<br>- Calling for the destruction of Israel | - Talking about expelling Palestinians<br>- Stripping Palestinians of their human rights<br>- Harming Palestinian civilians |
| **Palestinians (West Bank and Gaza)** | - Calling for forced displacement<br>- Denying Palestinian existence and refusing to recognize Palestinian rights<br>- **Supporting the war on Gaza and viewing military operations as moral** | - Supporting or justifying violence<br>- Refusing to acknowledge the Israeli side |

The analysis then found that **the two sides’ red lines corresponded to one another**[^1]. In concrete terms, what one side demanded as a line the other should not cross was matched by the other side’s pledge as a line it itself would not cross. For example, in response to the Israeli side’s most sensitive demand—“do not justify the October 7 attack”—the Palestinian side pledged “not to support or justify violence.” Likewise, in response to the Palestinian side’s strongest rejection—“do not support the war on Gaza or regard military operations as moral”—the Israeli side set red lines such as “not harming Palestinian civilians.”

The discovery of these mutually corresponding red lines helped restore some degree of trust and opened the way to constructive joint dialogue. Ironically, making clear what could not be agreed upon became the starting line for building agreement.

### The Algorithm That Bridges Division

At the core of the project’s technical innovation was an algorithm called **Bridging-based Ranking**[^2]. Adapted from a method used in Twitter/X’s Community Notes, it does not rely on simple majority rule. Instead, it **prioritizes opinions supported by people from different positions**. In this project, two different bridging algorithms were used in combination.

The first was a method called **Max-min Agreement** or **Equal Power Metrics**. Its mechanism was as follows: for each statement, the support rate was measured separately for the three groups—Israeli Jews, Palestinians from the West Bank and Gaza, and Palestinian citizens of Israel—and the statement’s score was defined as the **lowest support rate among the three groups**.

For example, if Statement A was supported by 95% of the Israeli side but only 40% of the Palestinian side, its score would be 40%. If Statement B was supported by 70% of the Israeli side and 65% of the Palestinian side, its score would be 65%, and Statement B would rank higher. In other words, by ranking statements according to the minimum agreement rate across all groups, “opinions supported by only one side” automatically receive low scores, which **prevents the majority group (97 Israeli participants) from overwhelming the minority group (41 Palestinian participants)**.

The second was a latent factor model already put into practical use in Twitter/X’s Community Notes. This method analyzes participants’ voting patterns to mathematically identify “statements that receive broad support regardless of political position.” By separating out individual voting tendencies, a statement’s general level of support, and the political alignment between participants and statements, it can identify “bridging statements” that cut across political divides[^1].

Together, these two algorithms extracted statements that could be supported across conflicting groups and across diverse political positions.

### How the LLM Was Used

A crucial point is that **the LLM was not used to facilitate the dialogue itself, but to refine statements after the dialogue**. Participants communicated directly with one another, while the LLM functioned as a “statement editor” that generated and polished collective statements based on the results.

Rather than using participants’ raw responses as-is, the project used GPT-4 to generate “collective statements”[^1]. However, the approach to dialogue and statement generation differed significantly between the Uninational Phase and the Joint Phase.

In the **Uninational Phase (Cycles 1–3)**, each group held discussions in a single language. Israeli Jews exchanged views in Hebrew, while Palestinians did so in Arabic; no translation intervened. Based on the views collected in these single-language discussions, GPT-4 generated collective statements in each group’s native language. Specifically, in the Hebrew cycle, the prompt, examples, processing, and output were all in Hebrew; in the Arabic cycle, they were all in Arabic.

This approach was adopted on the basis of experimental validation. When the researchers compared using English prompts with using prompts translated into participants’ native languages, **native speakers preferred the collective statements generated from prompts in their own language**[^1]. As a result, the project adopted a pipeline that processed each language directly, without translation.

In the **Joint Phase (Cycle 4)**, participants speaking Hebrew, Arabic, and English took part in the same dialogue, so the language barrier had to be overcome. Importantly, **participants were explicitly informed that machine translation would be used and were asked to use concise language**[^1]. Their comments were immediately rendered into all three languages using the Google Translate API so that others could read them in their own language. Rather than hiding AI’s imperfections, the project took a transparent approach that enlisted participants’ cooperation in improving translation quality.

From these multilingual inputs, GPT-4 generated collective statements in all three languages. Translation quality was then ensured through a refinement process. First, a draft was created in English; next, it was machine-translated using the Google Translate API; then native speakers refined the translations; and finally, human experts conducted a final review. This **multi-stage review** process compensated for the limitations of machine translation. The refined statements were then presented to participants and submitted for approval in a final vote, ensuring that the content generated by the LLM was ultimately validated by the participants themselves.

Although the paper does not discuss this explicitly, statement generation by an LLM may also have had the effect of **removing attribution—erasing who said what**. In conflicts between opposing groups, people may reject a proposal simply because it comes from someone on the other side, even if they would otherwise agree with its content. By integrating multiple opinions into a single statement, the LLM may have helped participants detach from personal reactions and evaluate the content itself. In a deeply polarized context like Israel–Palestine, this anonymizing effect may have contributed to consensus-building.

### Achieving High Agreement Rates

In the Joint Phase, the following results were reported[^1]:

**Five statements addressed to world leaders** (with at least 90% agreement in each group):

1. **An immediate ceasefire and release of all hostages** — ranked first by all participants
2. The creation of an international fund to lay the groundwork for a comprehensive long-term solution
3. The establishment of a global coalition to support Gaza’s reconstruction
4. The inclusion of civil society and NGOs in all peacebuilding and diplomatic discussions
5. The recognition of trauma on both sides

**Five statements addressed to local communities** (with at least 84% agreement in each group):  

The paper does not provide the exact wording, but reports that five statements received nearly equal support.

In addition, during the Uninational Phase, all three groups were found to share six **common values**: **peace, equality, human life, independence, security, and prosperity**.

These agreements were reached despite the pessimistic view at the time that finding common ground would likely be impossible. Judging by the numbers alone, the project was a major success.

**However, as the paper’s authors themselves acknowledge, this success has important limits.** The high agreement rates were achieved on abstract principles such as “an immediate ceasefire,” “recognition of trauma on both sides,” and “peace and human life.” These are important, but they are also **safe, unobjectionable principles that require no one to make a concrete sacrifice**.

Once the discussion moves to specific issues, however, consensus becomes difficult even with AI.

That is because the Bridging-based Ranking algorithm prioritizes “opinions supported by both sides.” The more abstract a principle is, the more room there is for different interpretations, and the easier it is for both sides to support it. But once proposals become concrete, interests become clearer, and the other side is more likely to oppose them. As a result, painful but specific agreements are less likely to emerge as the kind of “bridging statements” that AI is good at identifying.

The paper explicitly notes that “anodyne” or abstract statements generally attract broader agreement than statements that are more specific, normative, and practical[^1].

It therefore proposes, as a future task, developing methods to automatically measure a statement’s **concreteness** and **anodyneness**, and combining those measures with bridging metrics in order to identify “statements that are both bridging and concrete” more effectively[^1].

Abstract agreement matters as an entry point to dialogue. But by itself, it does not lead to actual conflict resolution. Agreeing that “we want peace” is a different matter from agreeing on “how, specifically, to achieve it.”

---

## Conclusion: The Difficulties That Lie Beyond Agreement

The AI Pulse project succeeded, in the “dialogue-impossible” conditions that followed October 7, 2023, in using the technical innovation of Bridging-based Ranking to derive shared recommendations among people in conflict. Both the Israeli and Palestinian sides rated “peace and human life” as important values by overwhelming margins of more than 92%, and both ranked “an immediate ceasefire and the release of all hostages” as their top request to world leaders. Even in situations that appear hopeless, openings for dialogue still exist.

That said, the agreement reached remained limited to abstract principles such as “peace,” “human life,” and “recognition of trauma on both sides.” Abstract agreement that “we want peace” is entirely different from the concrete question of “how to make peace and where to draw the border.” Border demarcation, the right of return for refugees, withdrawal from settlements—once discussion enters specifics, one side’s concession directly becomes the other side’s pain. A path for connecting 92% agreement to real-world policy and action still remains out of sight.

This challenge is not unique to Israel–Palestine. All three cases examined in this chapter confront the same structural barrier. In Taiwan’s vTaiwan, simply visualizing opinions through Polis did not by itself produce policy decisions; regulatory direction was determined only after stakeholders gathered for in-person public deliberation. In Bowling Green, more than 80% agreed on “expanding family-friendly activities,” but progress still depended on pillar groups working through the specifics by hand: what to build, where to build it, and who would pay for it.

This is not a flaw in the technology. It is a fundamental feature of human consensus-building. The higher the level of abstraction, the easier agreement becomes; the more concrete the issue, the more interests collide. AI is good at making the space of agreement visible. But deciding where to go from there, and actually taking the first steps, remains human work.

In Taiwan and Bowling Green, this gap between “agreement” and “implementation” was consciously built into the design. Taiwan’s collaborative meetings, Bowling Green’s pillar groups, and “We The People” district-based dialogues all provided institutional processes in which humans would take AI’s output and turn it into something concrete.

By contrast, the AI Pulse project did not build in an institutional mechanism, like vTaiwan’s, for directly connecting agreement to policy. The 138 participants in Phase 1 were all a selected group—NGO members already engaged in peacebuilding—and no direct link to governments or international institutions, the eventual recipients of the agreement, was built into the design.

Even so, indirect policy influence is beginning to emerge. Starting in 2025, ALLMEP launched Phase 2 for the general public and conducted large-scale polling in Israel, Palestine, and the United States. The results found that 73% of Israelis and 83% of Palestinians supported or accepted a comprehensive regional peace deal that included mutual recognition of statehood[^5]. These data have been cited by policymakers in the context of the Trump administration’s Middle East peace plan launched in 2025. This is a case in which public opinion made visible through broad listening began to function as material for policy debate even without a formal institutional bridge.

What AI provides is “lower-cost large-scale deliberation,” not “peace itself.” A multilingual dialogue among 138 participants that would previously have required hundreds of thousands of dollars and more than a year was carried out in four months (April–July 2024)[^1]. Even in situations that appear beyond dialogue, it can make visible the values shared by people with different positions. That may be only a small step, but where there is no dialogue, there are no steps at all. The success or failure of broad listening depends not on AI accuracy, but on how this “connection” is designed.

As of March 2026, the Gaza ceasefire agreement remains fragile but intact. The second phase of the ceasefire began in January 2026, stipulating a phased withdrawal and the disarmament of Hamas. However, the Israeli military has refused to withdraw from the buffer zone in eastern Gaza and still controls more than half of the territory. Even after the ceasefire, 673 Palestinians have reportedly been killed. The conflict has in fact expanded beyond Gaza: following a U.S.-Israeli strike on Iran at the end of February, war broke out in Lebanon with Hezbollah, and Iran retaliated with hundreds of missiles. Between the AI Pulse data showing that 73% of Israelis and 83% of Palestinians support peace, and the reality on the ground, a deep gulf still remains.

---

[^1]: Andrew Konya, Luke Thorburn, Wasim Almasri, Oded Adomi Leshem, Ariel D. Procaccia, Lisa Schirch, Michiel A. Bakker (2025). "Using Collective Dialogues and AI to Find Common Ground Between Israeli and Palestinian Peacebuilders." *ACM Conference on Fairness, Accountability, and Transparency (FAccT '25)*. The paper is available at: https://arxiv.org/html/2503.01769v3

[^2]: Bridging-based Ranking applies a method used in Twitter/X’s Community Notes to a peacebuilding project. For details, see the Community Notes guide (https://communitynotes.twitter.com/guide/en/about/introduction) and Konya et al. (2025).

[^3]: Alliance for Middle East Peace (ALLMEP). "ALLMEP hosts AI-assisted community dialogues with peacebuilders." https://www.allmep.org/ ALLMEP was founded in 2006 and networks more than 180 organizations working to resolve the Israel–Palestine conflict.

[^5]: ALLMEP, “AI Pulse US Poll Report,” September 2025 https://www.allmep.org/wp-content/uploads/2025/09/AI-Pulse-US-Poll-Report.pdf

[^4]: OpenAI. "Democratic inputs to AI" (2023). https://openai.com/index/democratic-inputs-to-ai/ Andrew Konya of Remesh (Co-founder & Chief Science Officer) received $100,000 from OpenAI’s “Democratic Inputs to AI” grant program and developed a GPT-4-based democratic dialogue system, “Democratic Policy Development using Collective Dialogues and AI.” The Israel–Palestine project builds on that technical foundation.
