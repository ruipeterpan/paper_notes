# Towards applying to CS Ph.D. programs

After going through tens, if not hundreds of blog posts on applying to Ph.D. programs, it would almost be inappropriate if I didn't write down something and throw in my two cents about this exhausting but ultimately self-enriching and fascinating process.

This blog post will hopefully be a useful guide to the students who are planning to apply to Ph.D. programs. (I was trying to target those who are pondering the million-dollar question of whether to do a Ph.D., but I don't feel entitled to write about that yet...) I will try to give you a glimpse of what the application process looks like and offer some advice on how to best approach, embrace, and enjoy this unique journey. I hope this helps you! If so, please consider paying it forward (maybe start by giving a hand to junior undergraduates in your research group).

Note that whatever I write down is biased because of my background and experience. For context, I grew up and went to high school in China, and then did my bachelor's in the United States, majoring in computer science and mathematics. During my undergraduate, I worked on systems (cluster resource management) for ML starting from the summer of my second year, and I fully committed to doing a Ph.D. in my third year. My research interests fall under the big topic of "Systems and Networking", and for my Ph.D. application, I applied to professors whose areas of interest range broadly across all system aspects of big data, e.g., sys for ml (training, inference, video analytics, etc.), ml for sys (congestion control, video streaming, etc.), cloud computing/data center resource management (e.g., serverless, scheduling training/inference), etc. Also, my honest opinions can be straight-up wrong, so take them with a grain of salt.

## Overview (WIP)

point to meta references; introduce what's in each chapter



## Chapter 1: Why do a Ph.D. at all?

In my opinion, you should do a Ph.D. if you have already done some research, been through the ups and downs (or at least know a bit about what they are like), and you still absolutely love doing research. If your Asian parents are forcing you to do a Ph.D. or if you want to stay in academia because you didn't get an industry job, think twice.

Again, I don't feel entitled to write more about this, so please take a look at all the references below.&#x20;

### References

* [读博，你真的想好了吗？- 张焕晨的文章 - 知乎](https://zhuanlan.zhihu.com/p/372884253) (Are you really sure about doing a Ph.D.? by Prof. [Huanchen Zhang](http://people.iiis.tsinghua.edu.cn/\~huanchen/))

## Chapter 2: Narrowing down the programs/professors of interest

Most people agree that when applying to Ph.D. programs, the advisor is the most significant factor (even more so than the school/department itself). Thus, picking awesome professors/person of interest (POI) is arguably the most crucial part of the application process: pick good (in terms of research interest match and personality match), and you might be happy for life.

95% of the students I know apply for 5-20 programs, and they typically target 1-3 POI for each program. In my case, I checked out \~50 professors in my field of interest and ended up boiling the list down to \~30 professors that I especially like from \~15 schools. This chapter will try to answer two questions: (1) how do you put up the big list of professors that you are generally interested in working with, and (2) how to narrow the list down to those who you are particularly interested in working with?

### How do you put up the big list of POI?

* Check out your advisor's lab mates, recent collaborators, and direct connections in their network. These people are likely to share similar interests with your current advisor, plus they know your advisor personally, so these folks should be fun to work with, assuming you love what you are doing right now.
* The math folks have a great thing called [The Mathematics Genealogy Project](https://www.mathgenealogy.org) where you can see this tree of academic relationships. You can also trace the tree of professors in cs, starting from very senior professors who work in your field of interest, and then go down the genealogy tree to look for those holding tenure-track positions. For me, I started with Ion Stoica. Fun fact: as many as six of my professors of interest have had direct connections with Ion!
* [CSRankings](https://csrankings.org) is also a great place to visit. You would want to first list the target conferences that you mostly read papers from (for me it was SOSP, OSDI, EuroSys, ATC, SoCC, NSDI, SIGCOMM). Then, go to CSRankings, select your target conferences, check out the professors from each school who has published in these venues, and go through them one by one. I spent a day at each school going over everyone. Unless you already have a specific topic that you would like to work on, IMO you should be open-minded in this part of your search and try to check out as many professors as possible. When I was going over csrankings.org, I had also marked professors who publish in venues like SIGMOD & VLDB, SIGMETRICS, HPC conferences, and ML conferences like ICML. Although I still ended up applying to system professors, it was fun to get to know what other folks in your field are working on.
* Take the same list of venues that you like. Then, take a look at the program committee of the recent conferences, and go to their personal sites one by one. Doing so will produce a list that overlaps very much with the one you got from csrankings.org
* Talk with other people, e.g. your lab mates, your current advisor, or Ph.D. students who were in your lab during their undergraduate. You will genuinely get a lot from this! My personal story is that I didn't consider applying to the program I ended up committing to (?) until after a friend of mine strongly recommended that I shoot this POI an email. I ended up getting in touch with the POI and found that I liked him a lot. So yes, talk to people!
* TODO: add something about the school list, e.g. proportion of lottery/safe schools

### How do you narrow the list down?

* On one end, you should work with people whose research interests truly excite you. Although I had a few professors who work on databases/HPC on my first list, I ended up throwing them away because I prefer some topics over others.
* On the other end, you should not work with people who are bad advisors. Advisors who ghosts/abuses students are a big no-no! I used two approaches to filter out these professors:
  * Check out their RateMyProfessors reviews. I get that teaching isn't for everyone and some professors put more emphasis on doing research, which is fine by me -- but ultimately, I honestly don't want to be working with someone who is a 1.2, because I think if a professor fails to create a supportive learning environment for their students, then it's unlikely for them to do so for their advisees. IMO a score that starts with a 1 is somewhat of a red flag.
  * Talk with their current students or search for their posts online. More practically, talk with people who might have heard about some bad news -- they tend to travel very fast. There has to be something wrong if many Ph.D. students quit a lab halfway through.

### References

* [finding CS Ph.D. programs to apply to](https://www.youtube.com/watch?v=hOSl3xPmHiQ)
* [How to pick a grad school for a Ph.D. in Computer Science](https://vijayc.medium.com/how-to-pick-a-grad-school-for-a-phd-in-computer-science-a5ce7dceb246)

## Chapter 3: Getting in touch with the POI

Now that you have narrowed down a list of 10\~30 professors you want to work with, it is time to get in touch with them. Some people don't bother to do this at all -- I do not recommend this, as I know professors who will only skim through your application package if you haven't reached out. Plus, getting in touch with them helps you figure out how many students they are taking this season, their ongoing/future interests, how enthusiastic they feel about your background, etc. So please, please reach out!

### Make a webpage and a CV first

Before you start emailing people, I strongly recommend making a personal webpage. Quite a few professors have also talked about the importance of personal sites in academia. A good Ph.D. student/research should be visible in the community/on the Internet, and Linkedin/Facebook/department-generated webpages are just not enough for that. Some good templates include [Jon Barron's website](https://jonbarron.info) and [academicpages.github.io](https://academicpages.github.io). Prof. Timothy Roscoe recommended including the following information on a personal webpage:

* A picture, preferably a recent one that actually looks like you. If you have a goofy picture that you really like, consider hiding it behind your main picture and make it show on hover.
* A list of publications. No pressure if you haven't published, especially for undergraduates who do research in systems.
* Some biographical information:
  * How long have you been a student? And how long have you got left?
  * Who do you work with? And what do you work on?
  * Random (but non-embarrassing) details for color. Do not go out of your way and put your pornhub account on your webpage, people may not like it...

TODO: add more toward the CV part

### How and when should you reach out?

The de facto way to get in touch with professors is to cold email them. If you are lucky to have the opportunity to meet with professors during a conference or have already known the professor, that would be great! But most people send emails anyway unless they know a POI very well, so getting the email (and the first impression) right is vital. Here are a couple of tips.

* Figure out the timing. My recommendation is to start as early as you feel comfortable once the fall semester begins and professors start to check their inboxes regularly so they will hopefully have enough time to get back to you. If you do this a few weeks after you submit the application, the POI might already have a batch of good candidates in mind (but still, better late than never). On a more fine-grained level, a good trick is to send a timed email scheduled for something like 8 AM, so that your email will be on the top of the inbox when professors check their inboxes.
* Be concise but on-point. There is a reference below that covers how to send out cold emails, but take it with a grain of salt, as that's more for emailing professors for undergraduate research opportunities. My suggestions are:
  * You must mention who you are working with right now and your research interests.
  * You must mention why you would like to work with the POI, and you should probably mention a bit of their existing work and why you like them. Better yet, go in-depth and ask a technical question/ask if xxx is an interesting idea for possible follow-up work.
  * You must attach your CV. I had attached a draft research statement just in case the POI has some time to take a look, but in retrospect, I don't think any of them had the time to read it (?).
  * You should probably put a link to your webpage in your email signature. Better yet, include a link to your calendar.
* Know the email etiquette (there are two articles in the references that cover this). To that end, triple-check your email for missed attachments and spelling mistakes before sending it out! Better yet, have your roommate proofread it.
* Within a school, reach out to professors one at a time and only move on to the next professor if the previous one doesn't get back to you in a week or so. Please don't reach out to five professors whose offices are right next to each other at the same time -- they talk.
* Please don't send three emails in a day to try to catch someone's eye. Professors are busy, but they will get back to you if they see a good fit. They might forget about things, and in that case, send a kind reminder after some time (say a week?) of not hearing back.
* Use your institutional email account. Gmail/outlook addresses are fine but don't use your QQMail.

### References

* [How to Cold Email a Professor](https://research.berkeley.edu/how-cold-email-professor)
* [How to Email a Professor](https://academicpositions.com/career-advice/how-to-email-a-professor)
* [千万别犯写邮件的大忌](https://www.zhihu.com/question/68514971/answer/469896862) (The DON'TS when writing emails)

## Chapter 4: Asking for recommendation letters and sending out requests

Most CS Ph.D. programs ask for three letters of recommendation.

First of all, IMO it is necessary to understand what the other side looks like about this recommendation system. Except for the actual letter itself, your recommenders will also be asked about your clarity of goals for graduate study, English skills, creativity, etc. The "scores" recommenders can give are chosen from truly exceptional (top 1%), outstanding (top 10%), above average (top 25%), not applicable/unable to respond, etc. Some other questions include ([source](https://www.1point3acres.com/bbs/forum.php?mod=viewthread\&tid=581428)):

* How long have you known the applicant?
* What group are you using for comparison?
* Admission recommendation to program is {strongly recommended, recommended, recommended with reservation, etc.}

### Who should you ask for letters?

Moving on. Before sending out requests for letters, you have to figure out who your letter writers will be. It would be best if your letter writers are:

* Well-known in the research community. Having a senior professor/big name write a strong recommendation letter for you helps immensely. Either that or some junior professors who are actively publishing in a relevant field. These professors have a strong network of connections, which is valuable considering how critical connections are. In contrast, a letter from a postdoc is probably less helpful -- but if a postdoc is writing a letter for you, try to have them co-write a letter with their advisor.
* Someone who knows you well. Whoever has worked with you extensively will have plenty of chance to know you and see that bright side of yours. If you mainly worked with Ph.D. students/postdocs during your research and a professor who wasn't very hands-on toward your research is writing the letter, I would suggest coordinating with those people so that the professor can put some insights from the others into the letter.
* Preferably someone in academia instead of the industry, and if they are from the industry, they should be involved in doing research (e.g., leading a research team. At least they should have a Ph.D. degree?). I'm not too sure about this, but rumors say that letters from people in the industry are pretty much useless. This makes some sense because the qualifications of a good Ph.D. student are somewhat different from those of a good intern, and letter writers from the industry would have no idea of what the program committee is hoping to see. Note that when I say people from the industry, I'm referring more to an average SWE intern's mentor -- if Lidong Zhou or Amar Phanishayee writes you a letter, then obviously it's very good!

### How should you ask for letters?

Now that you know who you will be asking, it would be best to let them know about it. Here are some tips for doing that.

* Ask early. You don't need to know your exact school list when you send your first email request, but you should at least give people a rough idea of the number of letters and when they will be due. Professors are busy, like really busy. Please ask for letters as early as possible so they can plan accordingly. Two months in advance is better than one, and one month is better than two weeks. Imagine being a professor who's going through the finals week and a couple of conference deadlines, and boom, five students show up to ask for a total of 100 letters that are due in one week. Just thinking about that makes me uncomfortable. Besides that, if you apply to 30 (?!) programs, some professors may not be too happy about submitting all those letters and will only agree to submit 10 of them. In that case, you should, of course, ask for some other professors to fill in the gap, but anyway, you don't want to know about this a week before the application deadline.
* Communicate clearly. It would suck if professors finish your letter but don't know where to submit them. My advice is: (1) It is preferable to send official requests through application portals in batches, so those emails don't get lost in the professor's email inbox. (2) Create a table in Google Sheet to keep track of the programs you are applying to, the application deadlines, and the status of the requests. For example, knowing the date of when you sent a request will be helpful when a professor looks up that email in their inbox. Of course, share this google sheet with your letter writers. (3) Send email reminders to remind professors about an upcoming application deadline, preferably a week in advance, if not more. Note that for most programs, the deadline for people to submit letters is some time after the application deadline, so don't panic if a professor uploads a bit late.
* Provide enough information. Once a professor agrees to write you a letter, attach many files when you get back to them, e.g., your cv, draft SoP, transcript, final project report. Some professors will also ask you to provide a list of relevant personal information, e.g., classes you have taken, major accomplishments, etc. Besides those, remind professors about how your opportunity with them helped you grow as a scholar.
* Ask for strong letters. It goes without saying that strong letters from professors say a lot about who you are as an applicant -- they will make a difference in your application. When you ask for letters, explicitly ask for strong letters, so that professors who will otherwise write average letters can give you the chance to pivot.

### What's next?

And at last, you should write them a thank you letter. Send a small gift (something under $20 should be fine depending on the department policy, e.g., a box of chocolate provided that they are not allergic to chocolate :P). When you commit to a program a few months later, also remember to send another email to let your writers know about the good news -- they will like it very much!

### References

* [How to get a great letter of recommendation](https://matt.might.net/articles/how-to-recommendation-letter/)
* [Requesting a letter of recommendation](https://homes.cs.washington.edu/\~mernst/advice/request-recommendation.html)

## Chapter 5: Writing up the statement of purpose

### How should you write a statement?

* Before you start writing your statement, make sure to read through many other people's (good) SOP. Prof. Phillip Guo had shared a few statements that are really nice examples. Although he has taken down most of the content and has asked people not to distribute them, some of these statements are scattered across the Internet, and I'm sure you are good at Googling stuff. There are also some other good examples online that I will try to link to in the future. TODO
* Start early to write up the first draft. I was applying to a pre-doctoral summer program so I was very lucky to have my first draft ready in the summer before the application season, but still, I didn't finalize my statement until a few days before Dec 15: it takes a lot of time to do the endless revisions. Another thing is that except for the research statement of purpose, different schools might ask for additional materials such as diversity statements, short answers to random questions, etc., so make sure to figure out those requirements way before the deadline.
* Go through a lot of iterations. Just like writing up a paper, the first draft is guaranteed to be bad, and good papers go through O(10) rounds of revisions. To that end, try to get a lot of feedback by having as many people read your statement as possible: writing center, (former) lab mates, high school classmates who are majoring in computer science, roommates, etc. Five people can probably offer you 50 suggestions, and if you go by 20 of them, your sop will rise to a new level.

### What should be in your statement?

* Keep in mind that professors will use this to judge your writing skills. If you are good at writing, it's time to show off. Otherwise, at least make sure there are no grammatical/spelling mistakes: that would look terrible. Consider using something like Grammarly to fix those mistakes and clarify your writing.
* A significant portion of the statement should be on your past research. People usually spend one paragraph for each project, and the ranking is either by relevance or by date. IMO if your projects connect well, by date is a more natural way since you get to tell the story behind your motivation to get a Ph.D., but don't worry if you rank them by relevance.
* In each paragraph, describe the project (e.g., collaborators, short background & motivation, major technical contribution, your contributions & takeaways). Keep in mind that the people reading your statement might be working in a different subfield (I don't suppose the bioinformatics people will know anything about Cuckoo hashing), so don't just copy-paste the abstract of your past papers.
* Depending on how much space you have left, IMO you should talk about your research interest using at least one sentence and at most one paragraph. If possible, IMO you should identify some rising research topics in the next few years and why they interest you. The POI might resonate with you if you come up with some good ones.
* It's a good idea to mention which professors you would like to work with. Try to aim for 1-3 professors in the statement, and briefly discuss (in one sentence) why you are interested in working with them.

### References

There are also a bunch of blog posts by people who know more about writing up statements than I do, so make sure to take a look at these articles, including but not limited to:

* [Inside Ph.D. admissions: What readers look for in a Statement of Purpose](https://nschneid.medium.com/inside-ph-d-admissions-what-readers-look-for-in-a-statement-of-purpose-3db4e6081f80)
* [How to Write a Statement of Purpose for Grad School](https://swapneelm.github.io/how-to-write-a-statement-of-purpose-for-grad-school)
* [Tips for Writing a Statement of Purpose](https://users.ece.cmu.edu/\~mabdelm/statement-of-purpose-tips.html)
* [Ph.D. Statement of Purpose](https://blog.nelsonliu.me/2020/11/11/phd-personal-statement/)
* [Writing a Statement of Purpose](https://djunicode.github.io/2018/10/16/writing-a-statement-of-purpose.html)

## Chapter 6: Preparing for interviews

Congratulations on sending out all of your applications! Take a little break first, both physically & mentally. Then, it's time to start preparing for interviews!

### What's a typical interview like?

* Duration: A typical interview lasts an hour or so. The ones I had been in lasted as short as 20 minutes and as long as two hours.
* Content: You usually start with some chit-chat, followed by a quick self-introduction. Then, the POI will likely ask you to talk about your research project(s), during which they will evaluate both your hard and soft skills. Afterward, you can ask the POI some questions, including the lab culture & dynamics, the graduate program, their research, etc.
* Will there be coding/technical questions? (???) It depends, but most professors don't ask these kinds of questions. From what I've heard, there are professors who ask about things like page fault handling in operating systems or ask you something from LeetCode (those are extreme outliers though). But there will of course be technical questions for your past research projects!

### What to do before, during, and after an interview

* Before an interview: Go over your statement and resume, since the professors will likely refer to them if they ask questions about you. If you included a topic that you are less familiar with in your future research interests, it doesn't hurt to delve a little bit deeper into that topic.
  * For every project listed on your resume, prepare the following:
    * One sentence that summarizes the project. This is like the title of your project but maybe with some more info for context.
    * A 3-minute introduction that expands a little bit more, say on the background, motivation, technical contributions, and results.
    * A 10-minute overview of the project that can be expanded into a 30-minute discussion. Totally write stuff down beforehand if you feel like it.&#x20;
    * Your contributions to the project. Undergraduates often get carried by Ph.D. students/postdocs in their research, so it's important to highlight what you did and what you got out of a project.
  * Note that the interviews vary in duration, and you might have multiple projects to talk about (I only focused on the most significant one), so be flexible about the timing.
* Before an interview: Read your POI's work. It's ok to prioritize POI who you are really interested in or who showed great interest in you. They won't ask you about the technical questions in their past research projects, but still, getting to know about what a POI used to work on shows your seriousness and enthusiasm. Different students spend various amounts of time on this phase of the preparation, but I think you should at least do the following. For each POI:
  * Do a quick pass through the title/abstract/collaborators/venues of all their past work.
  * Pick 2-4 of their projects to dive deeper into. I had focused on (1) their most highly-cited paper, (2) their most highly-cited first-author paper, (3) a highly-cited paper in the recent two years, and (4) a recent work that you are particularly interested in, either because you can relate to the project regarding motivations/techniques or because you are genuinely captivated. And by diving deeper into it, I meant going over all the figures, learning about the background/motivation/nuggets (high-level contributions and techniques), etc.
  * If you like this POI very much, you can totally go over the technical details of some of their papers. Because why not? Reading papers are fun! If not, then you should probably think twice about your application.
  * Take a quick skim at their Ph.D. thesis, especially the acknowledgments section to know more about them as a person.
* Before an interview: Do mock interviews (or research presentation talks) with your friends/labmates. I didn't, so I totally screwed up my first interview, but it was a good practice and I got the chance to learn from my mistakes. In retrospect, I really should have done an actual mock interview with some labmates and had them ask all kinds of questions.
* Before an interview: Make sure you have a quiet environment, a stable internet connection, and a good microphone. If you have noisy roommates or bad routers, you might want to reserve a quiet study room in a library in advance.
* Before an interview: Dress properly. FYI, the chats are mostly casual unless the interviewer specifically mentioned a (multi-person) serious interview.
* During an interview: Chill out, and be yourself. IMO a significant purpose of interviews is so that you can get a sense of the vibe/chemistry between you and the POI (vice versa), so don't force things like saying you are interested in something that you are not.
* After an interview: Send the POI a thank-you email. If you had a good chat, maybe it's time for some follow-ups. Anyway, you should let the professor know your thankfulness and reassure your enthusiasm for collaborating with them.

### References

* [What is a typical interview (informal chat) for a PhD in computer science like?](https://www.quora.com/What-is-a-typical-interview-informal-chat-for-a-PhD-in-computer-science-like-What-do-the-professors-generally-ask-Do-I-need-to-have-concrete-research-ideas-of-my-own-Should-I-read-a-lot-of-research-papers-by-the-professor)
* [TOP校CS Ph.D.面试经验教训总结](https://www.1point3acres.com/bbs/thread-628184-1-1.html) (Tips for interviewing at top-tier CS Ph.D. programs)
* [关于我自己观察到的中国学生在PhD面试过程中的一点特点](https://www.1point3acres.com/bbs/thread-590002-1-1.html) (My observations on Chinese students' traits in Ph.D. interviews)
* TODO: add more references that are in English

## Chapter 7: Having good mental health during the application season

My mental health was surprisingly good during my application season (probably because I only took three credits in the fall and spent most of my time polishing up a paper & applying to Ph.D. programs). Although I am no expert, here is some advice for staying positive during the six months. If things get too tough, please talk to the professionals.&#x20;

### Before sending out applications

* Make plans to abide by so that you don't stay up and rush things. "Rushing is the path to the dark side. Rushing leads to staying up. Staying up leads to bad health. Bad health leads to suffering." - Master Yoda
* Maintain a consistent, healthy sleeping schedule.
* Talk with supportive people around you and be supportive to each other. You are not alone, and every applicant is fighting the same battle. Surround yourself with people who can help relieve your anxiety, not trigger them.
* Workout. Participate in team sports, work on bodybuilding, take a random walk outside, etc.
* If things go well, this will be your second last semester in undergraduate. Since most people go to a different school for Ph.D., it will also likely be your last fall/winter in your current city. On weekends, spend some quality time with your friends here to create some enjoyable memories for future reminiscing. If you are studying at UW-Madison, [here is an article](https://zhuanlan.zhihu.com/p/425849399) I wrote on the 20 must-dos before you graduate.

### After sending out applications

* After sending out applications, you will likely have huge chunks of free time since the semester is over and Christmas is coming up. Although the interviews will be coming shortly, IMO you should first take a week-long mental break. Congratulations on sending out all those applications!&#x20;
* Once you get back from your mental break, you should get back to studying. You likely won't have a lot of things to work on, and your motivation might be low -- after all, your application is already out, and there is not much you can do to make it a lot better. Instead of spending all the free time being anxious about the applications, try to divert your anxiety, say by developing a new hobby. Read a book or something, or learn to cook.
* [1point3acres](https://www.1point3acres.com/bbs/). [zhihu.com](https://www.zhihu.com/question/379814619), and [The GradCafe](https://www.thegradcafe.com) have a lot of good information and stuff, but please consider restraining yourself from visiting these sites too often. Social media takes a toll on people.
* Also, it might be worthwhile to turn off instant notifications for your email inbox and check it a few times a day at regular times.
* Compare to yourself, not others. This is in general a great suggestion on how to live a happy life.
* _Spider-Man: No Way Home_ was in theatres during my application season, and it had a great line: "If you expect disappointment, then you can never really get disappointed". Don't get too hyped up if a POI reached out to you or if you did well in an interview. Otherwise, you will feel really bummed when you get rejected.

### References

* [How to effectively deal with Imposter Syndrome and feelings of inadequacy](https://academia.stackexchange.com/questions/11765/how-to-effectively-deal-with-imposter-syndrome-and-feelings-of-inadequacy-ive)

## Chapter 8: Choosing a Ph.D. program (WIP)



oh man, I should really be reading some more articles on this&#x20;

* talk with a lot of people: parents, relatives who work in academia, significant others, friends, people from the internet, labmates, current advisor(s), labmates, POI, POI's students, current students in the program but not in your POI's lab, current students at other places
* do your research
* try to attend the visiting days in person if possible. the vibe is an important thing

### What do to on the visiting day

TODO

### References

* [How should I choose between multiple Ph.D. programs I was admitted to?](https://academia.stackexchange.com/questions/66926/ive-been-admitted-to-multiple-phd-programs-how-should-i-choose-between-them)
* [The Definitive ‘what do I ask/look for’ in a PhD Advisor Guide](https://www.cs.columbia.edu/wp-content/uploads/2019/03/Get-Advisor.pdf)
* [All About Graduate School Visits (for CS PhD programs)](https://koronkevi.ch/posts/grad-school-visits.html)
* [地表最全奖学金攻略：我的二十二万刀经验分享](https://www.1point3acres.com/bbs/thread-763415-1-1.html) (Negotiating with Ph.D. programs for more fellowships: How I got a total of $220K from all offers)

## Appendix I: My application timeline

Don't feel obliged to exactly copy my timeline -- this is just for your reference.

* **Mid August**: First draft of the school list; First draft of SoP
* **Mid September - Early October**: Confirmation of recommendation letter from 3 professors
* **Early October - Mid November**: Send out cold emails to professors of interest. On average, it's two letters per week.
* **Early November**: Finalized school list
* **Early December**: Sent out all rec letter requests
* **Mid December**: Finalized SoP; Sent out all applications to programs in the U.S.
* **Late December - Mid February**: Interviews
* **Late January**: First unofficial offer
* **Mid February**: First official offer -- the offers came in all the way to early March, although I withdrew/turned most of them down.
* **Late March**: Committed to xxx!

## Appendix II: Meta-references

* [Chris Liu's list of resources for CS grad school application](https://chrisliu298.io/posts/grad-school-application.html)
* [Matt Might's HOWTO: Apply for and get into grad school for science, engineering, math, and computer science](https://matt.might.net/articles/how-to-apply-and-get-in-to-graduate-school-in-science-mathematics-engineering-or-computer-science/)
* [Advice on Research Communication Skills | Computer Science Department at Princeton University](https://www.cs.princeton.edu/grad/advice-on-research-communications-skills)
* [Reflections on my CS PhD Application Process](https://www.bodunhu.com/blog/posts/reflections-on-my-cs-phd-application-process/) | [Bodun Hu](https://www.bodunhu.com)'s Blog

For those of you who read Mandarin Chinese:

* [Top tier CS PhD招生官--我是如何审材料的](https://www.1point3acres.com/bbs/thread-585435-1-1.html) (How I review application packages as a senior student volunteer in the application committee of a top-tier CS Ph.D. program)
* [从审材料的角度谈谈研究生申请](https://www.1point3acres.com/bbs/thread-463109-1-1.html) (Grad school application from an application review's point of view)
* [也从审材料的角度讲讲如何准备cs phd申请！](https://www.1point3acres.com/bbs/thread-585851-1-1.html)(CS Ph.D. application from an application reviewer's point of view)
*
* [CS Ph.D. 2019 Fall 申请季总结 - James.Qiu的文章 - 知乎](https://zhuanlan.zhihu.com/p/60961921) (Review of my CS Ph.D. application in 2019 Fall by [Haoran Qiu](https://haoran-qiu.com))
* [CS Ph.D. 申请总结 (2021 Fall) - Romero的文章 - 知乎](https://zhuanlan.zhihu.com/p/362189295) (Review of my CS Ph.D. application in 2021 Fall by [Xiangfeng Zhu](https://xzhu27.me))
* [2022 Fall你都申请了哪些学校的MA/MS/Ph.D.？- ruipeterpan的回答 - 知乎](https://www.zhihu.com/question/379814619/answer/2325160660) (My personal review of my CS Ph.D. application in 2022 Fall)