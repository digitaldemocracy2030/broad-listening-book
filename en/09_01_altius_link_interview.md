# Using Public Listening AI for VOC Analysis in Contact Centers

Broad listening is about strengthening our ability to listen. So where do people listen to large numbers of voices as part of their job? One example is the contact center (call center), which handles customer inquiries every day. How did contact centers change after adopting Public Listening AI, a broad listening tool? This time, we conducted an online interview about that effort.

## Initiatives at Altius Link, Inc.

- Interview date: December 9, 2025
- Interviewee organization: Altius Link, Inc.
- Interviewees:
  - Yosuke Honma (General Manager, Data Analytics Unit, DX Service Development Department, DX Service Development Division, Business Planning Group)
  - Yoshiaki Tanaka (Assistant Manager, same unit)
  - Takahito Sawaguchi (General Manager, Operations Unit 2, CS Wide Area Department 6, CRM Division 1, KDDI Business Group)
- Interviewers: DD2030 Nakayama, Oki

## Company Overview and Contact Center Organizational Structure

Altius Link is a joint venture between KDDI and Mitsui & Co. that provides BPO services centered on contact centers. In addition to phone support, it handles a wide range of channels including email, chat, SMS, web forms, and AI chatbots. Its client base is primarily major B2C companies, spanning industries such as services, telecommunications, finance, manufacturing, and government agencies.

**Honma:** When we take on outsourced operations from a company, we refer to the department handling that work as a "desk." We have many desks—for example, one that operates KDDI’s customer inquiry desk, another that handles inquiries for a certain bank, and so on.

**Honma:** A desk is typically staffed by operators who answer calls, with supervisors (SVs) above them. SVs support operators and handle escalations. Above them are managers who oversee multiple SVs.

**Honma:** Operators are basically focused on handling calls, while process improvement is the responsibility of SVs and above. The use of Public Listening AI is also handled by SVs and above.

## Challenges in the Contact Center Industry: What Is VOC?

**Honma:** In the contact center industry, we distinguish between "contact reasons" and "VOC (Voice of Customer)."

**Honma:** A contact reason is the direct reason a customer got in touch—for example, a procedural matter like "I want to cancel." VOC, by contrast, focuses on the emotional background behind that reason, such as "the service is hard to use" or "the fees are too high" in the case of someone wanting to cancel.

**Honma:** Contact reasons are easy to visualize because they are accumulated as records. VOC, however, is ambiguous, difficult to categorize, and in some cases not verbalized at all. That makes it hard to collect and analyze, and the industry has struggled to make full use of it.

**Tanaka:** This is just my personal way of thinking about it, but contact reasons are like questions on a reading comprehension test where you find something explicitly written in the passage. VOC is more like, "Explain how the author feels"—you’re being asked to identify something that isn’t directly stated.

**Honma:** As far as I know, true VOC analysis has never really been achieved in the contact center industry. There have been efforts like text mining with morphological analysis or attempts at sentiment analysis, but to put it bluntly, the reality is that people have been extracting things that seem VOC-like intuitively, without ever clearly defining what VOC actually is.

## Why VOC Is Getting Attention Now

**Honma:** I was still working on the front lines five or six years ago, and back then there wasn’t much demand for VOC reports. The focus was more on contact reasons. VOC has only really become a hot topic in the last few years.

**Sawaguchi:** We see a lot of demand for VOC collection and reporting from client companies that are in mature industries and are now at a stage where they need to retain customers, improve LTV, and strengthen loyalty.

**Sawaguchi:** Traditionally, operators manually entered customer feedback into CRM systems and the like. But once people get used to it, it tends to become perfunctory—"Oh, this again," or "There’s no point in escalating this." Ideally, technology should collect it efficiently so that companies can focus on decision-making and resource allocation.

## How They Learned About Public Listening AI

**Honma:** It started when I saw Mr. Anno speak at an exhibition in February 2025. In a talk after the Tokyo governor election, he introduced an initiative using Talk to the City to gather the voices of Tokyo residents and reflect them in an electoral platform. I thought that could be applied directly to the challenge contact centers face in collecting VOC.

**Honma:** At the time, Talk to the City had not yet been localized into Japanese, so we gave up for the moment. Later, Tanaka told me that Mr. Anno’s team had released Public Listening AI, and that prompted us to try again.

## The Data Input Pipeline for Public Listening AI

**Tanaka:** Since we use Public Listening AI for the purpose of visualizing VOC, we’ve tailored the preprocessing accordingly.

<!--TODO: Turn the workflow into an illustration -->

**Tanaka:** We convert 1,000 to 2,000 calls per day into text using speech-to-text. In the first stage, we use an LLM to filter out data with no analytical value, such as voicemail calls. Anything that looks like voicemail is classified as "Yes," and anything else as "No." About 5% of the total is excluded at this stage.

**Tanaka:** Next, we run positive/negative sentiment classification. Since we view VOC as something that strongly reflects emotional aspects, only items classified as negative are sent on to the next stage. At that point, the volume drops to about 10–15%.

**Tanaka:** We then have Claude output the negatively classified call transcripts as structured data using Structured Output. It extracts opinions that appear to be either contact reasons or VOC, along with five or six metadata fields such as "what the opinion is about" and "what kind of evaluation it expresses." In the prompt, we include the definitions of contact reasons and VOC used at the desk managed by Mr. Sawaguchi.

![Structured data prepared for input into Public Listening AI. Columns organize VOC/contact-reason classifications and metadata extracted by an LLM from call transcripts.](images/09_広聴AI_入力用構造化データ.png)

**Tanaka:** There are two reasons we collect this information. One is that it can be entered into Public Listening AI as cross attributes. The other is that by feeding the structured data into a BI tool, we can visualize it from other perspectives as well.

**Tanaka:** We do the preprocessing every day, and feed data into Public Listening AI once every two weeks, analyzing about 2,000 VOC candidates each time.

**Tanaka:** Personal information is masked on the speech-to-text system side. We’ve had other desks ask us about trying something similar, but some react by saying, "It’s outrageous to send anything containing personal information into an LLM." If masking isn’t in place, the effort often stalls.

## Trial and Error in Improving Preprocessing

**Sawaguchi:** At first, we tried feeding the logs in as-is, but contact reasons and VOC got mixed together. We thought maybe we could fix it by refining the prompts inside Public Listening AI, but the more we did that, the more it started to feel like intentional manipulation. That didn’t seem right.

**Sawaguchi:** In the early days, when we just handed over PDF reports, the reaction was, "We already know that," or "We want something more concrete." That became the trigger for improving the preprocessing.

**Sawaguchi:** We decided to treat Public Listening AI strictly as a tool for visualization and discovery, keep the prompts simple, and improve the preprocessing so that VOC and contact reasons were separated beforehand. By narrowing the input down to things that really seemed VOC-like, and then further focusing only on negative items, we became able to capture not just "I can’t log in," but also the part where "the customer is having this kind of trouble because they can’t log in."

## The Value of Public Listening AI Lies in Being Able to "Touch" It

**Sawaguchi:** To be honest, I didn’t see that much value in it at first. The reason was that the strength of Public Listening AI is that you can interact with the reports dynamically, but at the beginning I was only seeing what Mr. Tanaka had created through screen sharing or screenshots.

**Sawaguchi:** Later, once I was able to generate and manipulate reports myself, I understood its value. I felt there was real value in the act itself—looking at things hierarchically, adjusting the density settings, and going to inspect clusters with especially dense customer feedback (VOC). If you don’t have an environment where you can actually interact with it, it’s hard to appreciate what makes it good.

![A VOC scatter plot generated by Public Listening AI. Clusters are color-coded, making it possible to grasp the overall landscape of customer feedback at a glance.](images/09_広聴AI_散布図.png)

**Tanaka:** At first, we were delivering screenshots in PDF form. But I don’t have domain knowledge, so I couldn’t tell which points were worth digging into. After discussing it with my supervisor, we arranged for Mr. Sawaguchi to use it directly. By chance, we had built it on an AWS server, so as long as the network connection was available, he could access it too. It really drove home for me that the best results come when someone with domain knowledge is the one actually using Public Listening AI.

**Sawaguchi:** I think there are two approaches to VOC analysis: one is analysis aimed at deriving suggestions and insights from the results, and the other is more like fixed-point observation—seeing whether activities changed and improved based on the insights obtained. When it comes to gaining insights, I found Public Listening AI’s exploratory, feeling-your-way approach extremely effective.

**Honma:** That said, there are challenges with fixed-point observation. Because Public Listening AI’s clustering can change each time, there’s no guarantee that the same cluster will appear again in the next run. There is demand for tracking changes over time, but that’s difficult with Public Listening AI alone. So for fixed-point observation, we also use Power BI and track changes using the categories created during preprocessing.

## A Case of Deeper Analysis: The Real Need Behind "I Can’t Log In"

**Sawaguchi:** One difference from conventional text mining is that a certain degree of interpretation comes into play.

For example, if you aggregate login-related inquiries by contact reason, you get categories like "I can’t log in" or "password reset." But the customer’s real goal wasn’t logging in itself—they wanted to log in in order to do something. We started asking whether making login easier would actually reduce these inquiries.

So we asked ourselves whether we could capture *why* customers wanted to log in. We ran Public Listening AI once, used the resulting categories to narrow the focus to areas we wanted to explore more deeply—such as login-related issues—and then fed that subset into Public Listening AI a second time.

![Cluster names and descriptions generated by Public Listening AI. Along with the number of comments in each cluster, the tendencies in customer feedback are summarized.](images/09_広聴AI_クラスタ名-説明.png)By adding specific intent and few-shot examples to the prompt so that it would capture *why* customers wanted to do something, we became able to see things like "what percentage are concerned about making a payment" and "what they really wanted was to check their statement." That opens the door to proposals such as sending statements by email so customers don’t need to log in at all.

## Tips for Effective Use: Narrowing the Scope and Combining with BI Tools

We asked Mr. Sawaguchi what advice he would give to someone new to using Public Listening AI.

**Sawaguchi:** Even two weeks’ worth of data is a substantial volume, so the things that first appear tend to be generalized and abstracted expressions. It’s too early to look at that and conclude, "Oh, so that’s what’s going on." The value lies in the act of feeling your way toward insights, so the key is learning how to explore.

**Sawaguchi:** Specifically, if you use the attribute columns created during preprocessing as extraction conditions in Public Listening AI, they function almost like category selectors. First, use the BI tool to identify categories whose volume has increased, then select and narrow them down in Public Listening AI, view them at a density of 20–30%, and inspect the specific customer comments. That’s the trick to narrowing things down.

**Sawaguchi:** The flow is to use Power BI to check dominant categories of feedback or categories that have changed, and then use those attributes to filter Public Listening AI and examine the content.

![A VOC dashboard in Power BI. It enables fixed-point observation of category composition ratios and changes over time.](images/09_BI画像_モザイク無し.png)

**Sawaguchi:** It’s difficult if you try to use Public Listening AI alone to go looking for the answer. The important mindset is to go looking for hints. A lot of people assume the answer must be there, and then get disappointed when they don’t find one.

## Challenges in Talent Development

**Sawaguchi:** It’s extremely difficult to develop people who can handle both design (data design for preprocessing) and operations (tool usage). If I were to train someone on my team to use Public Listening AI, I think the training would focus on the dynamic manipulation and filtering of reports.

**Sawaguchi:** What we may need first are people who can communicate requirements to engineers to some extent, and who can give feedback on the gap between expected and actual output so that cycle can keep turning. Without the will to dig out value from customer voices ourselves, it just ends with receiving a report.

**Tanaka:** I also found it very easy to work with Mr. Sawaguchi. When I produced preprocessing results, he would quickly give feedback like, "I’d like you to dig a little deeper from this angle," or "Could we add this perspective?" That allowed me to add processing steps or incorporate things into the prompt. With people from other desks, feedback often doesn’t come, and it ends with whatever a generic prompt produces.

Why was Mr. Sawaguchi able to collaborate in this way? We asked about his background.

**Sawaguchi:** I’ve spent a long time in operations, but I’ve always loved technology and had a strong interest in it. Recently, I’ve been attending a domestic MBA graduate program, and what I learned there in data science and the use of technology matched well with this initiative, so it was easy for me to get into it. I also like mathematics, and I’m the kind of person who gets excited when looking at raw data.

## Rollout to the Front Lines and Reactions

**Sawaguchi:** The other day, I showed it to our operations site in Fukuoka, and they were really surprised. In particular, operators currently enter VOC directly into CRM tools like Salesforce, so the reaction was, "Does this mean we won’t need to do that anymore?" That input work takes quite a bit of effort, so they saw it as a relief.

**Sawaguchi:** Right now, we can already summarize speech recognition data after a call ends and paste it into the CRM. But extracting VOC according to definitions tailored to each client company’s characteristics is difficult with generic call center products. I felt this project gave us enough insight that we might even be able to co-develop something with the company that makes the speech recognition software we currently use. There may be knowledge sleeping here—such as what the definition of VOC really ought to be—that product developers don’t yet know.

**Sawaguchi:** If we’re going to show something to operators, it’s more effective to feed back not the collected voices themselves, but how those voices were used for improvement. If we can show them, "We’re going to make these improvements based on this feedback," they’ll be more likely to think, "Then I should keep raising these voices." That’s better for motivation on the front lines.

## Effects of Adoption and Future Outlook

We asked about the quantitative effects of introducing Public Listening AI.

**Sawaguchi:** We haven’t yet reached the point of seeing improvements in the numbers. As for measurable effects... well, my overtime has increased. You can do this endlessly, and it’s so interesting that I just keep going.

**Sawaguchi:** The original goal is to increase customer loyalty and reduce cancellations. That’s the effect we’re aiming for now, but we’re not there yet.

Asked what functions he hopes Public Listening AI will have in the future, Mr. Tanaka said this:

**Tanaka:** It would be great if broad listening could be done directly from audio files as they are. When you convert speech to text, some information is lost. Emotional aspects appear in vocal intonation and similar features, but in text they become just characters. If you could input audio files directly and output VOC using Fourier-transform spectra as data, that would be incredible.

**Sawaguchi:** One thing I’m really glad we did this time is that it made us realize how underutilized speech recognition logs are. They’ve been called "information assets," but not really used. Now our client companies are also realizing, "This much information can be extracted," and "Maybe this is actually a gold mine."

In what is often called the Fourth Industrial Revolution, I think information from customer touchpoints will only become more valuable. Altius Link, with one of the largest contact center resources in Japan, holds a wealth of that kind of data. Right now, we’re also concretely exploring ways to use it beyond simply extracting VOC. This was an excellent opportunity—and entry point—for making use of unstructured text data.

## Post-Interview Reflection: Why We Chose to Cover VOC Analysis in Contact Centers

"Broad listening"—listening widely—originally developed in the context of politics and public administration. But its essence is not confined to those fields. We have wanted to find out whether this way of thinking can apply to any setting where people engage seriously with human voices.

The field of VOC (Voice of Customer) analysis in contact centers turned out to offer an especially suggestive example in response to that question.

Contact centers are places where enormous volumes of "voices" gather every day. At the same time, many of those voices are not being fully put to use. While inquiry reasons (contact reasons) are recorded and aggregated, the emotions and background behind them—that is, VOC—have long remained something ambiguous, difficult to define, and "understood to be important, but too hard to handle."

This is strikingly similar to what we have seen in municipalities and citizen participation settings. Opinions are collected. But then people say, "We don’t really know what is being said," or "We don’t know how to use this in the end." As a result, the voices themselves become formalized and hollowed out.

What drew us to this case was that it was not simply a story of "they introduced Public Listening AI and it worked." Rather, what emerged through the interview were things like:

- the difficulty of defining what VOC actually is
- the trial and error involved in discovering that leaving too much to AI does not work
- the fact that preprocessing and design required a great deal of ingenuity and dialogue
- the fact that the value lay less in the analysis results than in the act of exploration itself

These are the "unflashy but essential aspects" of broad listening. What is happening here is not analysis designed to automatically produce answers. Nor is it merely the introduction of a tool that relies on expert intuition.

People with domain knowledge formulate their own questions, use AI as a guide for thinking, move back and forth through clusters of voices, and develop hypotheses.

This process has essentially the same structure as the practice we have aimed for in politics and public administration: engaging seriously with the voices of citizens. There is no need to read this case merely as a corporate example. Rather, we hope it will be read as a grounded case for thinking about what it means for broad listening to take root in an organization, and how AI and humans should divide roles.

The answers are not prepared in advance. But the method for posing questions and continuing to engage with people’s voices is very much here.
