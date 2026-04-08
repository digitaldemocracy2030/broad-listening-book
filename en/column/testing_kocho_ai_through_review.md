# Testing Broad Listening AI with Steam Reviews and Google Maps Reviews

—A small experiment to get past the “we don’t have any data yet” problem

Broad listening is an approach for organizing large volumes of free-text responses into “clusters by issue” so you can see the overall picture at a glance. Using systems like Talk to the City or Broad Listening AI, you can move beyond just the emotional temperature—“people are divided” or “this is blowing up”—and instead describe, in calmer and more concrete terms, what people are actually talking about.

At the same time, there is a wall that almost everyone hits at the beginning:
“So what exactly should I use as data?”

For this interview, we spoke with **Mr. Tanenobu (X: @mtane0412)**, an early user of Talk to the City and Broad Listening AI, and also a contributor to Broad Listening AI. He has been collecting data from Steam reviews and Google Maps reviews to create an entry point for using Broad Listening AI.

---

## First, find places where opinions gather around a shared topic

**Nishio:**
Steam reviews and Google Maps reviews—using those as data sources is a really interesting idea.

**Tanenobu:**
When I thought about how to try Broad Listening AI, reviews seemed like the most accessible kind of dataset where “people are expressing opinions about a shared topic.”
The first wall people hit when they want to try Broad Listening AI is that it’s actually pretty hard to find **places where lots of people are talking about the same thing**.

So at first, I was just experimenting—building tools to create datasets for Broad Listening AI from places like Reddit and Steam.

**Nishio:**
This story about the “starting point” is really good. For readers who are interested but haven’t taken the first step yet, it could be a big help.

---

## Steam reviews: a place where intensity—and pain—remain intact

**Tanenobu:**
Steam makes it easy to collect reviews. I first tried it with **Pogostuck**, a very difficult game that I personally like.
I figured that compared with a major blockbuster title, it would be the kind of game where deeply invested players leave especially passionate reviews.

The positive reviews did a very nice job of capturing what makes the game so compelling.
And the negative reviews weren’t just simple criticism—they included raw, visceral expressions of suffering. The results were fascinating.

**Nishio:**
I’d like to make this a little more concrete for beginners.
When you say “the data you fed into Broad Listening AI,” how much did that include? Was it just the review text, or did you also include rating labels and timing information?

**Tanenobu:**
Basically, it’s the **review text**. Each review is “someone’s voice,” so the first step is to gather those together and feed them in.
On top of that, Steam has categories like “recommended” and “not recommended,” so it helps if you can also keep track of the **difference between positive and negative reviews**.

As for preprocessing, I only do enough to make the data analyzable. I try not to strip away the writing style or the emotional intensity. In fact, it’s often more interesting later if the insults and slang are left in as part of the “atmosphere of the moment.”

---

## Breaking down a “Steam review meltdown” into its actual components: the *Monster Hunter Wilds* example

**Tanenobu:**
I also tried this with **Monster Hunter Wilds**[^mhwilds], which became a hot topic because its Steam reviews had turned into a complete mess. I thought it was actually a very good use case for Broad Listening AI.
Online articles mostly stopped at “Steam reviews are in chaos!” But by using Broad Listening AI, it became possible to see what users were actually complaining about, which I found really interesting.

[^mhwilds]: The Steam version of *Monster Hunter Wilds*, released on February 28, 2025, had fallen to just 19% positive reviews by around June and was labeled “Overwhelmingly Negative,” which briefly became a widely discussed topic.

I assume the developers are already doing this kind of analysis carefully on their side, but ordinary users mostly just experience emotional spikes on their social media timelines.
So I thought it would be interesting to have a more bird’s-eye view like this.

**Nishio:**
That makes sense. If all you know is that it’s “Overwhelmingly Negative,” the conversation gets stuck in a simple binary of for or against, and nothing really moves forward.

What Broad Listening AI is doing, I think, is **breaking issues down and drawing a map**.
Once you can see “what the problems actually are” and “where dissatisfaction is concentrated,” the starting point for conversation changes. It creates a foundation for much more useful discussion.

---

## Google Maps reviews: from the outside, “wakame” suddenly comes into focus

**Tanenobu:**
For Google Maps, I used a hot spring inn in a region where businesses take Google Maps review management quite seriously.
What was interesting was that both the positive and negative aspects that all the locals already know about showed up clearly.

Like how the ocean-side rooms have spectacular views, but the mountain-side rooms don’t overlook anything.
Or how the owner is a former chef who is now a fisherman, and the inn’s seafood meals are one of its strengths.
When I saw that you had picked up “wakame shabu-shabu,” Nishio, I realized, “Oh, so they were also serving shabu-shabu made with early-harvest wakame.”

**Nishio:**
When I looked at the report knowing absolutely nothing about the area, the first thing that caught my eye was the “local gourmet” cluster. Then I drilled down in the hierarchical view and found a group labeled “wakame,” and I thought, “What on earth is this ‘wakame’ group?” Then I opened it further and realized, “So wakame shabu-shabu is a thing!”

It had the same kind of appeal as reading a travel guidebook.
It’s not exactly easy reading, so I don’t think it’s aimed at ordinary tourists, but for people whose job is writing articles for tourists, it could be useful as a way to find story ideas grounded in “what actual visitors are saying.”

Once you start to see applications like this, it feels like an experiment with real growth potential—one that could make tourism developers see the value of collecting visitor feedback more proactively.

**Tanenobu:**
With Google Maps too, the main thing is still the review text. Since there are star ratings, it’s helpful to keep those as well if possible, because it makes it easier to see the “reasons for high ratings” and the “reasons for low ratings.”

From the local side, a lot of the results are things people would react to with, “Well, yes, of course.”
But what I found interesting is that for outsiders, those same things can emerge as genuine “discoveries.” Just like with “wakame,” local characteristics can rise up out of a bundle of reviews.

---

## Turning abusive comments into something people can bear to read: the value of having AI in the middle

**Nishio:**
In the Steam example,
I thought it was especially valuable that “review comments, which often contain abusive language, are transformed by AI into calmer wording and then aggregated and organized.”
That seems useful simply because it reduces the psychological damage for the people reading the reviews. It’s also great that Broad Listening AI isn’t limited by input language—you can generate a Japanese report even if the reviews are written in many different languages.

**Tanenobu:**
I sometimes give guidance to local businesses on things like Google Business Profile, and a lot of people hesitate because dealing with reviews feels mentally exhausting.

Just like with call centers, I think this is an area where having AI in the middle can make the whole process function better.

**Nishio:**
Lowering the cost of reading and receiving feedback seems extremely important to me.
Reviews should really be a source of hints for improvement and rediscovery of strengths, but because of users’ emotional phrasing and aggressiveness, the people on the receiving end can find them psychologically painful and start avoiding them altogether. If AI can act as a kind of “cushion” in the middle, it could lower the psychological cost of making use of reviews.

---

## Being able to reproduce “Yes, we already knew that” is what makes the next step possible

**Tanenobu:**
With both game reviews and tourist-destination reviews, the output mostly reflected things that were already familiar if you knew the subject well.
But that also means, “Users and customers really do feel that way, really are saying those things, and Broad Listening AI was able to extract them properly.”

I think it’s amazing that it can present the intuitive feel that individuals carry around in a form that can be shared with other people.

For people who are already deeply engaged with customer feedback, if you show them the results, they’ll probably just say, “Yeah, I know.”
But there aren’t that many other businesses in the region that are engaging at that level. If Broad Listening AI can expand people’s ability to engage with customer feedback, that could be huge.

**Nishio:**
I think that’s a really important point.
The value isn’t only in “discovering new facts.” There’s also value in **reproducing** what people on the ground already know intuitively in a form that others can share.
Especially in regions and organizations, a lot of knowledge exists only “inside the heads of the people who know,” and that becomes a bottleneck in handovers and consensus-building. Broad Listening AI is powerful because it makes that visible and puts it on the table as a basis for conversation.

**Tanenobu:**
In municipal tourism and commerce, it has been difficult until now to handle qualitative data well. So research has tended to center on things like fixed-point quantitative surveys conducted in the same way every year.
Now that qualitative data analysis is becoming easier, I feel there’s real potential.

If people can feel that leaving a review means their voice is actually being reflected, I think more people will be willing to leave reviews.

---

## Summary

* For people who want to try Broad Listening AI but stumble at the first hurdle—“we don’t have any data”—it was a smart move to use reviews of familiar, concrete subjects like local areas, tourist facilities, and games as the entry point, rather than abstract policy topics.
* Even in volatile spaces like Steam, having AI in the middle makes it possible to shift from “online outrage” to a bird’s-eye view of the issues.
* Broad listening based on Google Maps reviews can turn what locals take for granted into discoveries for outsiders. It also has room to grow from the perspective of tourism and regional management.
* The value is not only in finding “new answers,” but also in being able to reproduce “Yes, we already knew that.” That becomes a foundation for sharing on-the-ground knowledge and building consensus.
