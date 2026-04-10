# Team Mirai Initiatives in the 2026 House of Representatives Election

Written by: sumino


In the 2026 House of Representatives election as well, Team Mirai (meaning “Team Future”) launched its electoral platform website and invited proposals from voters. Building on the experience and lessons of the previous House of Councillors election, the concept was changed from an "interactive policy platform" to a "platform where your voice is heard." The goal was not only to collect voters' opinions, but also to create a mechanism that would ensure those voices actually reached the people responsible for policy.

![Top screen of the "platform where your voice is heard"](images/06_02_team_mirai_manifesto_top.png)

Behind this name change were two issues that had become clear during the House of Councillors election.

The first was the challenge of designing an experience that made it easy for voters to express their views. Even when people have vague frustrations or requests, it is not easy to put them into words from scratch in a chat interface. Many people find the act of articulating their thoughts to be a hurdle. To address this, it was necessary to design an experience that would allow voters to communicate their ideas and concerns more naturally.

The second was the challenge of creating a system capable of receiving and processing the voices that came in. During the House of Councillors election, a large number of proposals were submitted regarding the platform, but it was difficult to review all of them. To avoid overlooking valuable input, there was a need for a system that could help people grasp a wider range of opinions within limited time.

To solve these issues, three updates were introduced for the House of Representatives election.

## Evolution of the Chat Experience

In the "interactive policy platform" used during the House of Councillors election, voters entered free-form text into a chat box. For the House of Representatives election, the chat experience was redesigned to reduce the burden on voters as much as possible.

First, users could now drag to select text directly within the platform and simply press the displayed "Question" or "Proposal" icon to start a chat about that specific passage. As discussed in the House of Councillors election section on the "problem of change proposals being submitted to the table of contents," there had been an issue with ambiguity over which page a comment referred to. This design makes the intended target of the voter's comment clear.

During the chat as well, the AI is designed to display options (quick replies). Even voters who find it burdensome to type their own sentences can continue the conversation simply by tapping the options. If none of the suggested options fit, free-text input is also available.

![Chat screen in proposal mode](images/06_02_quick_reply.png)

Chat screen in proposal mode. The AI presents options so that voters can communicate their intentions without feeling burdened.

In addition, questions and proposals can now be clearly distinguished. In the House of Councillors election, the UI assumed that users were making proposals, so even voters who simply wanted to ask a question were sometimes pushed into creating a proposal. This time, users can switch between question mode and proposal mode, and if a question naturally develops into a proposal, the transition is seamless. The AI interprets the voter's intent and automatically moves into proposal mode.

In proposal mode, the AI probes more deeply into the voter's intent. It asks what the proposal is directed toward and why it is necessary, helping to make the proposal more concrete. In the House of Councillors election, the AI interviewer was already capable of listening and producing a change proposal, but in the House of Representatives election the quality of that interviewing improved. It became possible to elicit more specific and more insightful proposals.

## Implementation of AI-Based Opinion Analysis

The second update was AI-based analysis of opinions. Broadly speaking, two functions were implemented: classification and tagging of proposals, and topic-level aggregation of similar proposals.

First, proposals submitted by voters are automatically classified and tagged by AI. For example, they may be sorted into categories such as "implementation details to add," "objections to the policy direction," or "wording fixes such as typos." An automatic labeling system had also existed during the House of Councillors election, but for the House of Representatives election this mechanism was integrated into the application itself. With proposals classified in this way, policy staff can review them according to their own interests and areas of responsibility.

The other function is the ability to group similar proposals into topics. These grouped topics are visualized within the app, allowing policy staff to scan them for an overall picture and then drill down into the details.

![Topic and proposal visualization screen](images/06_02_topic_example.png)

Screen showing proposals and topics submitted to the platform. The topics under discussion are displayed in a list, and users can also drill down into individual proposals. This made it easier for policy staff to review a wider range of opinions in response to the problem identified during the House of Councillors election: "some voices fail to reach decision-makers because they are buried under large numbers of similar proposals."

## Admin Dashboard for Policy Staff

The third update was the implementation of an admin dashboard for policy staff.

In the House of Councillors election, proposals were compiled on GitHub, but for the House of Representatives election a dedicated admin interface was prepared. It allows filtering by platform type, proposal status, and AI-based proposal classification, enabling staff to focus their review on the specific platform sections they are responsible for.

![Proposal list screen for policy staff](images/06_02_proposal_list.png)

Proposal list screen for policy staff. The filtering function allows proposals to be narrowed down by type and classification.

In addition, a function was implemented that allows AI to generate draft revisions to the platform based on voters' proposals. These AI-generated revisions can be adopted as-is, or human editors can modify them as needed. By using AI while leaving the final judgment to people, the system makes it possible to develop policy that reflects a greater number of voices.

It is worth revisiting the issue discussed in the House of Councillors election section on "conflicts among concrete changes." Dozens of revision proposals that attempted to change the same passage in different ways ended up in conflict, and policy staff had to resolve them one by one by hand. The admin dashboard and AI-generated draft revisions are one answer to this problem. Through a human-in-the-loop workflow in which proposals are first aggregated, AI then presents draft revisions, and humans make the final decision, the foundation has been put in place for reviewing more voices and reflecting them in policy.

In practice, small revisions to the platform were made during the campaign period based on feedback that came in. Minor improvements—such as revising individual expressions or adding clarifications—were achieved through this system. At the same time, challenges remain in how to build consensus and reflect major changes affecting overall policy direction within the speed and pressure of an election campaign. A full pipeline has now been established for collecting voices, analyzing them, and reflecting them in policy, but how to handle changes that affect the whole remains a theme to be refined in future efforts.

## Progress Since the House of Councillors Election and Remaining Challenges

Looking back on the House of Representatives election initiative, it is clear that the lessons gained from the House of Councillors election were steadily reflected in the system design.

The challenge of designing a low-friction entry point, which had become clear in the House of Councillors election, was addressed through features such as starting chats from selected text and introducing quick replies. In addition, AI-based classification and aggregation, together with the admin dashboard, established a foundation that allows policy staff to grasp a wider range of opinions.

From an "interactive policy platform" to a "platform where your voice is heard." What this change represents is a direction aimed at consistently supporting the entire path from voters expressing their views to those views reaching the policy-making process. A system was built to connect the broad listening cycle—opinion collection, structuring, deliberation, decision-making, and feedback—more continuously and with fewer breaks.

At the same time, the reality remains that this cycle could not be fully completed during the election period. There is still a gap between the evolution of the system itself and the ability to fully make use of it in the actual operation of an election campaign. How to close that gap will likely become an important theme in future efforts.
