# Litela Inc.

Kai Tanaka (Representative, Litela Inc.)

English translation by Luke Closs

## Introduction

“Normally, you have to listen at the table for about five minutes before you can catch up, but with this, you can understand right away.”

These were the words of a facilitator at a residents’ meeting in Ota, Gunma Prefecture, after trying the new system.

![Facilitator feedback from the Ota residents’ meeting.](images/08_01_ota_facilitator_feedback.png)

This article introduces the features and field testing of Recogra (https://recogra.app), a system that uses AI to support in-person group discussions in real time. While broad listening excels at analyzing large volumes of text-based opinions gathered online, Recora focuses on something different: in-person discussions that are unfolding right now. What changes when the content of a discussion is visualized in real time in settings like residents’ meetings, where people gather around tables to talk face-to-face? Recora is the result of repeatedly testing that question in real-world settings.

## About Litela Inc.

Litela Inc. is a company that values spaces where people gather face-to-face, while aiming to strengthen the connections formed in those spaces through technology. Its core mission is to create systems for settings like residents’ meetings so that no one’s views are lost and participants can connect with one another.

The development of Recora began with an awareness of the challenges inherent in face-to-face settings. Broad listening tools that analyze large volumes of text opinions after the fact are useful once a meeting is over. But they do not solve the problem that, in discussions happening right now, someone’s comment may be overlooked, or a facilitator may fail to grasp the context and end up asking only superficial questions. How can we make better use of the discussions that emerge in face-to-face settings, while they are still happening? That was the question from which Recora began.

## Structural Challenges in In-Person Group Discussions

In residents’ meetings and educational settings, group discussions in small groups of four or five people are widely used. The format lowers the psychological burden of speaking in front of a large audience and makes it easier for each participant to bring their own views into the conversation.

In these settings, facilitators move from group to group, supporting the discussion. They may ask questions about a theme one group is struggling with, or introduce a perspective that emerged in another group, helping ideas cross between tables.

However, when multiple groups are running at the same time, facilitators face structural limits. If four groups are discussing for 20 minutes, a facilitator can spend only about five minutes with each group on average. Every time they arrive at a new table, they first have to figure out what the group is talking about, and often end up with only surface-level check-ins such as, “How is the discussion going?” or “Is there anything you’re having trouble with?” The deeper questions they would ideally ask—“What is the basis for that view?” or “How would this look from the opposite perspective?”—do not come naturally unless they already understand the context.

Another challenge is the loss of information during whole-group sharing. When a representative from each group presents back, time constraints mean that the points covered are selected at that person’s discretion. Topics that were actively discussed within the group, as well as minority views, are often omitted at this stage. And because facilitators themselves cannot fully track every group’s discussion, they may not even notice when the presentations are skewed.

## Recora’s Design Philosophy

In response to these challenges, Recora narrowed its goals to two points:

1. Enable facilitators, upon arriving at a group, to instantly grasp the recent flow of discussion and key issues so they can intervene more effectively
2. Ensure that, during whole-group sharing, everyone can understand the content of each group’s discussion without omissions

One design principle remained consistent throughout development: **AI should not become a participant in the discussion**. If AI joins the discussion directly, active exchange among participants decreases, and resistance can arise if people feel the AI is dominating the conversation. The priority is to preserve the quality of human-to-human dialogue, so AI’s role is deliberately limited to that of a “cognitive augmentation tool” that helps facilitators understand the situation.

## System Overview

Recora consists of two main functions.

The first is **real-time discussion visualization**. A device is placed on each group’s table, and a microphone records the conversation and transcribes it in real time. The content is then structured by AI, and the facilitator can view the status of all groups at a glance on a tablet. The interface is designed around two switchable views: one showing the recent flow of comments in chronological order, and another providing a bird’s-eye view of the discussion’s overall topic structure. This allows facilitators to understand what a group is currently talking about before they even arrive at the table.

![Screen for real-time discussion visualization. The overall topic structure appears on the left, while the recent flow of comments is shown chronologically on the right.](images/10_05_outward_visualization.png)

The second is **illustrated summaries**. After the group discussions end, the content from multiple groups is integrated and automatically generated as a single illustration. Traditionally, graphic recording required a skilled practitioner drawing by hand, but by using image-generation AI, this can now be produced in real time. Minority views and discussion points that are often omitted from representative presentations can instead be shared with the whole room visually.

![Example of an illustrated summary. Discussion content from multiple groups is integrated into a single illustration by AI.](images/10_05_recogra2.png)

## Field Demonstrations: Tawaramoto and Tomioka

Alongside development, Recora has been brought into multiple residents’ meetings and refined through feedback from the field. Here, I will introduce two especially sustained demonstration cases: Tawaramoto in Nara Prefecture and Tomioka in Gunma Prefecture.

### Tawaramoto Residents’ Council: Building Through Three Sessions

In Tawaramoto, Nara Prefecture, a residents’ council was convened as the town approached the end of its Fourth Comprehensive Plan and began formulating the next one together with residents. About 20 residents selected through random sampling were divided among four tables to discuss the theme of “community building through connection.” Recora was introduced over three sessions (December 2025, January 2026, and February 2026) in this council, which was organized in collaboration with the Japan Initiative Foundation.

A visualization device was placed on each group’s table, and facilitators circulated with tablets in hand. By checking the status of all groups on the tablet, they could decide which group to visit next, and upon arriving, review the recent flow of comments before joining the discussion. During whole-group sharing, they could also use the tablet to bridge perspectives across groups, saying things like, “Group 1 was also discussing this issue.”

![How the system was used in Tawaramoto. Devices were placed on each group’s table, and facilitators circulated with tablets, grasping the context before asking deeper follow-up questions.](images/10_05_tawaramoto_flow.png)

Compared with recordings from earlier sessions, interruptions such as “What are you talking about right now?” had decreased. Facilitators offered feedback such as the following:

- Because the status of each group is visible in real time, groups can stimulate one another even without physical movement between tables
- During observation and whole-group sharing, facilitators can quickly understand the recent context, reducing misalignment and allowing them to probe key points more deeply
- Facilitators can focus on more essential forms of support, such as the atmosphere of the room and interpersonal dynamics

In participant surveys, 80% said the illustrated summaries were “effective.” On the question of whether they understood what other groups had discussed, the share of respondents selecting “understood very well” increased. Rather than simply ensuring a baseline level of understanding for everyone, the system seemed to create stronger peaks of understanding.

### Tomioka: The Experience of Seeing Discussion Become Images

In the Tomioka demonstration in Gunma Prefecture, both functions—real-time visualization and illustrated summaries—were used, as in Tawaramoto. When the illustration was projected on the screen during whole-group sharing, the atmosphere in the room changed.

![Demonstration in Tomioka. During whole-group sharing, an illustration summarizing discussions from multiple groups is displayed on the screen.](images/10_05_tomioka_illustration.png)

Facilitators shared feedback such as the following:

- There is a uniquely AI-driven sense of surprise and enjoyment in seeing our dialogue turn into an illustration before our eyes, but it also helps clarify issues and make information visible
- Even within limited sharing time, the illustration makes it possible to grasp other groups’ ideas intuitively, which deepened the subsequent whole-group discussion
- When one’s own comments and those of others are expressed objectively as images, it encourages reflection and self-examination among participants

### What These Two Cases Revealed

What emerged consistently from both Tawaramoto and Tomioka was that “visualizing discussion” does more than simply organize information. People see their own words become images, and other groups’ conversations appear on the screen. As those experiences accumulate, participants begin to feel that “my words remained in this space,” which in turn encourages them to speak again and to return for future sessions.

For facilitators as well, being able to grasp the context in advance means they can arrive and immediately ask questions like, “Could you explore that point a little further?” They can engage deeply without interrupting the flow of discussion. That is the form this system aims to achieve.

## Future Outlook

As the demonstrations have continued, our thinking about what makes a good discussion space has also changed. Rather than aligning everyone around a single conclusion during whole-group sharing, the real value lies in the moments when someone encounters another group’s context and says, “I’d like to hear more about that later,” or “Let’s get tea sometime,” and relationships begin to form across groups. A good discussion space is one that leaves something behind after the session ends.

Technically, we also see potential in analyzing the audio itself to mark moments when laughter occurred or when conversation stalled, allowing facilitators to perceive nonverbal aspects of the room that transcription alone cannot capture. We are also considering formats such as poster-session or gallery-walk models, in which each group’s discussion is displayed in the room as a single image and participants move around, leaving sticky notes as they go. The goal is not simply to collect opinions and stop there, but to create more points of contact where human connections emerge and future collaboration can begin.

If broad listening provides the “breadth” to collect and survey large volumes of opinions, then what Recora aims to provide is the “depth” and “connection” of discussion in face-to-face settings. Participants put their views into words, those words remain in the space, and by intersecting with the words of others, they give rise to the next action. We will continue building systems that serve as the foundation for that kind of space, together with the people working in the field.
