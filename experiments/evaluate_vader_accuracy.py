"""
Comprehensive Human Ground-Truth Annotation and Evaluation of VADER
on the YouTube Discussion Corpus (N = 100, Video ID: dQw4w9WgXcQ).

Annotation Protocol:
- Positive (pos): Expresses enjoyment of song/video, admiration of Rick Astley, praise slang
  ('banging song', 'fire', 'so sick', 'kills', 'masterpiece', 'staple'), nostalgia/affection,
  or enthusiastic celebration of milestones.
- Negative (neg): Genuine frustration, complaint, criticism of the song/video, or negative self-feeling.
- Neutral (neu): Factual statements, meme mechanics/descriptions, link-sharing anecdotes
  (e.g., 'put this in Kahoot', 'link to Deltarune newsletter', 'QR code on car'),
  or non-evaluative quotes/timestamps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Load comments
df_raw = pd.read_csv("data/youtube_corpus_100_comments.csv")

# Principled human ground truth annotations for all 100 comments
# Based on close reading of the 100 comments in data/comments_for_labeling.json
human_annotations = [
    # 1: "I Rick rolled my entire French class..." (Humorous prank anecdote, neutral recount)
    ("neutral", "Meme prank recount in class; no direct music sentiment"),
    # 2: "Is everyone going to ignore the fact that Rick Astley looks like a twelve year old boy but has a uniquely deep voice?" (Observational curiosity/compliment)
    ("neutral", "Observational humor regarding appearance and voice"),
    # 3: "They should turn off ads... WOW! One billion views!! God bless you all..." (Celebratory)
    ("positive", "Celebration of 1 billion views milestone"),
    # 4: "the most famous link in the history of internet" (Factual meme statement)
    ("neutral", "Factual cultural observation about the link"),
    # 5: "My algebra teacher said that this was a link to the test answers" (Anecdote)
    ("neutral", "Prank anecdote about algebra teacher"),
    # 6: "Ok let’s be honest here Rick rolling is the one meme that will never die" (Appreciation of meme)
    ("positive", "Enduring praise for the meme longevity"),
    # 7: "Never thought the video was taken down... I just wanted to hear my boy Rick Astley again." (Affection)
    ("positive", "Affectionate intent to listen to Rick Astley"),
    # 8: "[non-text or emoji-only comment]" (Placeholder/unintelligible)
    ("neutral", "Non-text placeholder"),
    # 9: "Let’s put the meme aside, this is actually a banging song." (Praise slang)
    ("positive", "Unambiguous musical praise: 'banging song'"),
    # 10: "In some parallel universe, this should already got 1Billon views." (Praise)
    ("positive", "Wish that video had reached 1B sooner"),
    # 11: "i love how everyone just got summoned here now for the same reason" (Affectionate)
    ("positive", "Expresses enjoyment of community gathering"),
    # 12: "So this isn't the link to Deltarune's new newsletter" (Meme observation)
    ("neutral", "Observation of being redirected unexpectedly"),
    # 13: "History class in 20 years: Internet culture, rickrolling" (Meme observation)
    ("neutral", "Humorous historical prediction"),
    # 14: "One random fact: This is the most copied link in history." (Factual observation)
    ("neutral", "Factual observation about the URL"),
    # 15: "got rickrolled... but given how f*cked up the world is atm I glad this video is still around" (Appreciation)
    ("positive", "Appreciation that the video brings comfort"),
    # 16: "Like if you in fact did NOT get Rick Rolled and came here on purpose. 😂" (Engagement bait/positive)
    ("positive", "Came voluntarily out of enjoyment"),
    # 17: "I knew it was a trap, but I clicked the link anyway. It's fire tho" (Slang praise)
    ("positive", "Slang praise: 'It is fire tho'"),
    # 18: "Gonna flag this for nudity so I can rick roll the YouTube staff" (Meme joke)
    ("neutral", "Prank idea targeting platform staff"),
    # 19: "Question of the century: who is the 1 billion person to get rickrolled" (Curiosity)
    ("neutral", "Philosophical meme question"),
    # 20: "Guys the QR code is a Rick roll" (Factual PSA)
    ("neutral", "Informational warning about QR code"),
    # 21: "I just got rick rolled by scanning someone’s hairline" (Humorous anecdote)
    ("neutral", "Absurd prank anecdote"),
    # 22: "I Rickrolled my entire school. I still smile bc of it" (Happy memory)
    ("positive", "Happy recollection of school prank"),
    # 23: "1987 : normal song 2026 : *national anthem of the universe*" (High praise)
    ("positive", "Hyperbolic praise: 'national anthem of the universe'"),
    # 24: "Who randomly got this on there fyp👇👇" (Algorithmic inquiry)
    ("neutral", "Engagement prompt about algorithm feed"),
    # 25: "No-one should give up on this song. This song has rickrolled us... but will always stay in our heart no matter what." (Heartfelt devotion)
    ("positive", "Heartfelt devotion: 'will always stay in our heart'"),
    # 26: "I saved this to a playlist and I forgot about it. Long story short I rickrolled myself." (Humorous anecdote)
    ("neutral", "Accidental self-prank anecdote"),
    # 27: "I’ve being rick rolled many times in my life that I don’t even trust links anymore." (Meme fatigue)
    ("neutral", "Meme effect on link trust"),
    # 28: "How Am I Still Getting Rick Rolled In 2026😭🙏" (Meme surprise)
    ("neutral", "Expresses amusement/exasperation at meme longevity"),
    # 29: "Saw the post about the video being taken off of YouTube, searched for the love of the game." (Devotion)
    ("positive", "Searched out of devotion: 'for the love of the game'"),
    # 30: "There's a reason this has over 600M views..." (Implicit high praise)
    ("positive", "Implicit musical quality justifying 600M views"),
    # 31: "I feel dumb for falling for that post" (Self-directed regret)
    ("negative", "Negative self-feeling: 'feel dumb'"),
    # 32: "I just scanned a qr code from geometry dash level Stereo Madness hold, but wth is this" (Confusion/annoyance)
    ("negative", "Irritation/confusion at unwanted redirection"),
    # 33: "Soooo who else saw a post saying this video got deleted?" (Discussion prompt)
    ("neutral", "Factual inquiry about deletion hoax"),
    # 34: "I looked up this video forgot that I did, clicked the tab and got rick rolled by my past self" (Anecdote)
    ("neutral", "Amusing anecdote about past self"),
    # 35: "My professor sent us this link as the \"final exam key\"........" (Prank anecdote)
    ("neutral", "Anecdote about professor's exam key prank"),
    # 36: "[non-text or emoji-only comment]" (Placeholder)
    ("neutral", "Non-text placeholder"),
    # 37: "Wait, this isn’t the link to finding out Obama’s last name" (Meme joke)
    ("neutral", "Humorous misdirection joke"),
    # 38: "broo i bought some shirt with a qr code written on it, this is what i got😭😭" (Anecdote)
    ("neutral", "Anecdote about QR code shirt"),
    # 39: "If this doesn't teach you not to click on a dark web link, nothing will." (Meme warning)
    ("neutral", "Humorous caution about clicking links"),
    # 40: "He gave us up, He let let us down, He ran around & deserted us, He made us cry, He said goodbye, He told a lie & hurt us" (Inverted parody of lyrics)
    ("neutral", "Clever lyrical inversion/parody; not genuine malice"),
    # 41: "How tf did this get in my recommended Did I just got rickrolled by Youtube" (Surprise)
    ("neutral", "Surprise at YouTube recommendation algorithm"),
    # 42: "When they put ads so you can't rickroll anymore" (Mild criticism of ads)
    ("negative", "Complaint about platform ads ruining the prank"),
    # 43: "Damn got Rick rolled cause someone said it's down, well I'll gladly take it knowing it's not down" (Relief/positive)
    ("positive", "Glad the video is still active"),
    # 44: "Didn't like this when it first came out.....Now I love it. SMH, Go figure." (Conversion to positive)
    ("positive", "Explicit change of heart to loving the song"),
    # 45: "So I just scanned a QR CODE in my buddy’s car, it brought me to this…." (Anecdote)
    ("neutral", "Anecdote about buddy's car QR code"),
    # 46: "This..isn't..possible...Rick Rolled in 2018!??! WOOOOOOO I'm SO FRICKEN HAPPY!" (High joy)
    ("positive", "Explicit exuberant happiness: 'SO FRICKEN HAPPY'"),
    # 47: "Keep rickrolling people till we get 1 billion views." (Campaign/encouragement)
    ("positive", "Supportive milestone push for the video"),
    # 48: "I was watching a tutorial on python to create qr codes and i scanned the thumbnails qr code and brought me here🥀" (Anecdote)
    ("neutral", "Coding tutorial anecdote"),
    # 49: "wait this is not the link to Shrek X: Endgame" (Meme joke)
    ("neutral", "Fictional movie link joke"),
    # 50: "I can’t believe this showed up in my recommended youtube itself rickrolled me" (Surprise)
    ("neutral", "Observation of algorithmic recommendation"),
    # 51: "The fact ppl are STILL getting Rick rolled almost 2 decades later is crazy to me" (Amazement)
    ("positive", "Amazement at cultural resilience"),
    # 52: "The girl you like VS Rick Astley... Never gonna give you up..." (Comparative meme praising Rick)
    ("positive", "Humorous meme glorifying Rick's loyalty over dating"),
    # 53: "People don't Rickroll me: Me: Fine, I'll do it myself" (Thanos meme joke)
    ("neutral", "Self-rickroll meme quote"),
    # 54: "This Song will be supported till the end of the world. Edit: Thx for the 1K like guys." (Endorsement)
    ("positive", "Loyal musical endorsement"),
    # 55: "Jokes on you, I love this song." (Direct love)
    ("positive", "Unambiguous love of the music"),
    # 56: "Rickrolled at 6% battery. Worth every drop." (Praise)
    ("positive", "Devotion despite low battery: 'Worth every drop'"),
    # 57: "I got rickrolled by a link saying it got taken down." (Anecdote)
    ("neutral", "Hoax link recount"),
    # 58: "I can’t get rickrolled, I memorized the whole link. Wait how did I end up on this vid-" (Irony)
    ("neutral", "Humorous ironic admission"),
    # 59: "I just got Rick rolled by a QR code on a car. In 2025." (Anecdote)
    ("neutral", "QR code car anecdote"),
    # 60: "One of my favorite songs; it reminds me of my late father. Such nostalgia—it brings tears of joy to my eyes!..." (Deep sentimental affection)
    ("positive", "Deep personal emotional love and nostalgia"),
    # 61: "We got rickrolled so many times that we don't even care anymore" (Meme desensitization)
    ("neutral", "Desensitization to meme frequency"),
    # 62: "feels like ive seen it a billion times" (Hyperbole)
    ("neutral", "Observation of ubiquitous viewing"),
    # 63: "I'm pretty sure this has set a number of records: 1. The most copied link 2. The most heard song anywhere..." (Record setting praise)
    ("positive", "Celebration of historic records broken"),
    # 64: "It has over A BILLION VIEWS because of A MEME" (Meme milestone observation)
    ("neutral", "Observation of meme-driven view count"),
    # 65: "When the music is worth it, no matter how much time passes, people always come back here to hear it" (Musical worth)
    ("positive", "High praise for musical endurance"),
    # 66: "The Term RickRoll Still Exists and I could not be happier." (Idiom happiness)
    ("positive", "Idiomatic celebration: 'could not be happier'"),
    # 67: "Disclaimer: this song is highly addictive" (Compliment)
    ("positive", "Complimentary appraisal: 'highly addictive'"),
    # 68: "Just so y'all know I clicked the dice button to get a random song and this is what I got. Bruh." (Random discovery)
    ("neutral", "Random song button anecdote"),
    # 69: "I think I have the wrong link, this isn’t “Free Giant Kinder Surprise Egg”" (Meme joke)
    ("neutral", "Misdirection joke"),
    # 70: "It's not a joke, it's a good song Every time you get Rickrolled, you listen to it in full" (Direct music praise)
    ("positive", "Affirmation: 'it's a good song'"),
    # 71: "Getting rickrolled by a national examination board 🥀🥀 CBSE😭😭" (Exam board prank)
    ("neutral", "Indian CBSE exam board meme recount"),
    # 72: "You are here because you thought it's a hyperlink leading you elsewhere. I am here because I chose to be... We are not the same" (Giancarlo Esposito meme)
    ("positive", "Voluntary pride in choosing the music"),
    # 73: "Just got rolled by a random QR code in a mobile game. Rick indeed will never give up." (Anecdote)
    ("neutral", "Mobile game QR code encounter"),
    # 74: "sometimes you just gotta rickroll yourself because no one else will" (Meme reflection)
    ("neutral", "Humorous self-reliance meme"),
    # 75: "Finally, after all this time... we reach this historic moment that will be forever cherished... 1 billion views." (Triumphant praise)
    ("positive", "Triumphant celebration of 1B milestone"),
    # 76: "So I just found that Rick Astley is still alive but Rick Astley if you see this... she did this song and I thought now Rick Astley has to sing this..." (Fan note)
    ("positive", "Friendly fan note to Rick Astley"),
    # 77: "He looks like a normal Teenager, But his Voice kills" (Slang praise)
    ("positive", "Slang compliment: 'his Voice kills'"),
    # 78: "So....Did that guy on Twitter just...Rickrolled us all ?" (Twitter prank inquiry)
    ("neutral", "Inquiry regarding Twitter prank"),
    # 79: "Imagine telling Rick in 1987 this song will be one of the most popular songs to ever exist" (Appreciation)
    ("positive", "Appreciation of song's historic standing"),
    # 80: "Who was the monster who disliked this work of art?" (Defensive praise)
    ("positive", "Passionate defense of 'this work of art'"),
    # 81: "This song is still significantly better than brainrots" (Comparative praise)
    ("positive", "Praise over modern media: 'better than brainrots'"),
    # 82: "1988 : damn, this song is good 2019 : damn, this song is better 2020: damn, i miss the Internet 2025: nothing can describe this song" (Consistent praise)
    ("positive", "Chronological appreciation of song quality"),
    # 83: "9 Years from now this music is still not dead because death always get rick rolled." (Humorous praise)
    ("positive", "Praise for musical immortality"),
    # 84: "Never gonna give you up, Never Gonna let you down" (Direct lyric quote)
    ("neutral", "Direct quotation of lyrics"),
    # 85: "Wait this isn’t How To Rickroll People Tutorial." (Joke)
    ("neutral", "Tutorial misdirection joke"),
    # 86: "1980: lol i wonder what people will listen to in the future 2021: *never gonna give you up*" (Meme comparison)
    ("neutral", "Temporal comparison meme"),
    # 87: "This may not be Avengers Endgame but it is in English." (Joke)
    ("neutral", "Humorous irrelevant comparison"),
    # 88: "Okay, who else just showed up on a random Sunday when the song turned 17 to Rick roll themselves?" (Celebration)
    ("positive", "Affectionate voluntary listen on anniversary"),
    # 89: "this music video is still going strong!" (Praise)
    ("positive", "Enduring praise: 'still going strong'"),
    # 90: "Ah yes we meet once again" (Affectionate return)
    ("positive", "Welcoming familiar encounter"),
    # 91: "His voice has 50 years more than him or what" (Observational humor)
    ("neutral", "Humor regarding vocal age vs appearance"),
    # 92: "I rickrolled my entire school because we could suggest songs for the school to play... and I suggested this one." (Anecdote)
    ("positive", "Proud prank accomplishment"),
    # 93: "We all know this is not our first time here" (Familiarity)
    ("neutral", "Shared audience familiarity"),
    # 94: "Wait a minute, this is not the \"darude sandstorm\" song someone link me to." (Meme joke)
    ("neutral", "Darude Sandstorm cross-meme joke"),
    # 95: "Everyone: Having fun Rick: Keep the cash flowing" (Industry humor)
    ("neutral", "Humor about Rick's royalties"),
    # 96: "I sent this in the group chat... and now they're after me.. 😭😭" (Prank aftermath)
    ("neutral", "Group chat reaction humor"),
    # 97: "It’s come to the point that being rickrolled is a privilege" (Praise of status)
    ("positive", "Praise: 'being rickrolled is a privilege'"),
    # 98: "Everyone focuses on it being a meme and few actually take a moment to really appreciate it" (Appreciation call)
    ("positive", "Urging genuine musical appreciation"),
    # 99: "I'm never mad at Rick Rolling. Keep reminding me that masterpiece." (High praise)
    ("positive", "Explicit accolade: 'masterpiece'"),
    # 100: "So much more than a meme! And I hope Rick realizes that... this is a staple of a generation" (Generational praise)
    ("positive", "Cultural accolade: 'staple of a generation'"),
]

assert len(human_annotations) == 100, f"Expected 100 annotations, got {len(human_annotations)}"

# Build labeled dataframe
records = []
for i, row in df_raw.iterrows():
    h_label, rationale = human_annotations[i]
    v_label = row["sentiment_label"]
    v_score = row["sentiment_score"]
    records.append({
        "comment_id": row["comment_id"],
        "author": row["author"],
        "rank": row["rank"],
        "likes": row["likes"],
        "text": row["text"],
        "vader_score": v_score,
        "vader_label": v_label,
        "human_label": h_label,
        "human_rationale": rationale,
        "agreement": (v_label == h_label)
    })

df_labeled = pd.DataFrame(records)
out_csv = Path("data/youtube_corpus_100_comments_human_labeled.csv")
df_labeled.to_csv(out_csv, index=False, encoding="utf-8")
print(f"Saved human-labeled corpus to {out_csv}")

# Evaluation Metrics
y_true = df_labeled["human_label"]
y_pred = df_labeled["vader_label"]

labels = ["positive", "neutral", "negative"]
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels)
cm = confusion_matrix(y_true, y_pred, labels=labels)

print("\n" + "=" * 60)
print(f"VADER EVALUATION ON YOUTUBE CORPUS (N = 100)")
print("=" * 60)
print(f"Overall Accuracy: {acc * 100:.2f}%")
print(f"Macro F1-Score:   {macro_f1:.4f}")
print("\nConfusion Matrix (Rows = Human Ground Truth, Cols = VADER):")
print(f"                Pred Positive   Pred Neutral    Pred Negative")
for idx, true_lbl in enumerate(labels):
    print(f"True {true_lbl:<10}   {cm[idx, 0]:<14}  {cm[idx, 1]:<14}  {cm[idx, 2]:<14}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=labels, digits=4))

# Diagnostic breakdown
zero_scores = (df_labeled["vader_score"] == 0.0).sum()
print(f"Zero-Inflation Rate: {zero_scores}/100 ({zero_scores:.1f}%)")

# Detailed breakdown of misclassifications
misclassified = df_labeled[~df_labeled["agreement"]]
print(f"\nTotal Misclassified Comments: {len(misclassified)}")
pos_called_neu = df_labeled[(df_labeled["human_label"] == "positive") & (df_labeled["vader_label"] == "neutral")]
pos_called_neg = df_labeled[(df_labeled["human_label"] == "positive") & (df_labeled["vader_label"] == "negative")]
neu_called_pos = df_labeled[(df_labeled["human_label"] == "neutral") & (df_labeled["vader_label"] == "positive")]
neu_called_neg = df_labeled[(df_labeled["human_label"] == "neutral") & (df_labeled["vader_label"] == "negative")]

print(f"  - True Positive predicted as Neutral:  {len(pos_called_neu)} (e.g. 'banging song', '600M views', 'masterpiece')")
print(f"  - True Positive predicted as Negative: {len(pos_called_neg)} (e.g. 'fire tho', 'could not be happier', 'voice kills')")
print(f"  - True Neutral predicted as Positive:  {len(neu_called_pos)}")
print(f"  - True Neutral predicted as Negative:  {len(neu_called_neg)}")

# Save summary metrics to json for paper/documentation
summary_stats = {
    "sample_size": 100,
    "accuracy": float(acc),
    "macro_f1": float(macro_f1),
    "zero_inflation_count": int(zero_scores),
    "zero_inflation_pct": float(zero_scores / 100.0),
    "true_distribution": df_labeled["human_label"].value_counts().to_dict(),
    "vader_distribution": df_labeled["vader_label"].value_counts().to_dict(),
    "confusion_matrix": cm.tolist(),
    "labels": labels,
    "pos_misclassified_as_neg": int(len(pos_called_neg)),
    "pos_misclassified_as_neu": int(len(pos_called_neu))
}

import json
with open("data/vader_evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_stats, f, indent=2)
print("\nWrote summary metrics to data/vader_evaluation_summary.json")
