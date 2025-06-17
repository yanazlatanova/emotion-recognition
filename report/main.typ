#import "@preview/datify:0.1.3": custom-date-format
#import "acm.typ": acm

#show link: set text(hyphenate: false)


#show: acm.with(
  title: [Modeling Emotion Recognition Under Uncertainty in Internet Comments],
  authors: (
    (name: "Yana Zlatanova", email: "yana.zlatanova.gold@gmail.com"), (name: "André Plancha", email: "andre.plancha@hotmail.com"), (name: "Vérane Flaujac", email: "vflaujac@gmail.com")
  ),
  abstract: [Emotion recognition in text is a crucial challenge in natural language processing, particularly when dealing with noisy, subjective annotations such as those found in internet comments. We explore uncertainty-aware emotion classification using the GoEmotions dataset, a large, multi-label corpus of Reddit comments labeled with 27 nuanced emotion categories. We fine-tune DistilBERT models using multiple emotion taxonomies and label aggregation strategies that preserve annotator disagreement by converting classification into a multi-label regression problem. Three loss functions -- Mean Binary Cross Entropy, SoftRank-approximated nDCG, and Weighted Mean Squared Error -- were evaluated to assess their impact on model performance across classification, ranking, and confidence-based metrics. Results show that despite dataset noise and imbalance, DistilBERT can learn meaningful emotion representations and effectively capture emotion uncertainty. ],
  keywords: ("Emotion Recognition", "NLP", "GoEmotions" , "Internet Comments", "Machine Learning"," HuggingFace", "DistillBERT", "Evaluation Metrics"), 
  bibliography: bibliography("refs.bib", style: "acm.csl")
)

//Understanding user sentiment and emotion is a growing challenge in machine learning, especially for enabling chatbots to respond with empathy and relevance. In this study, we focus on classifying emotions in short sentences using the GoEmotions dataset, which contains Reddit comments labeled with 28 distinct emotions. Throughout our analysis, we discovered that emotion classification is inherently subjective. Multiple annotators often assigned different labels to the same comment, and some emotions appeared ambiguous. After preprocessing the data and performing exploratory visualizations to understand its structure, we trained a model using distillBERT for emotion classification. We then evaluated the model's performance using various metrics to assess its effectiveness. *Our work also explores key questions such as: Which emotions are represented? Does the dataset show demographic bias? And is the model accurate enough for deployment in real-world chatbot systems? This study builds upon existing research while offering our own perspective on implementing and analyzing emotion recognition models.*

// We also develop a user interface to visualize predictions and identify promising directions for future work, including calibration, error analysis, and the use of more expressive language models.

= Introduction
/*==Problem*/
As the flavor of speech that determines the meaning beyond the literal translation, emotions are essential to human communication. Due to helping machines better understand human text and speech, emotion recognition is a crucial subfield of natural language processing (NLP). This is especially helpful in chatbots, customer feedback analysis, and mental health monitoring@EmotionAnalysisinNLP. However, there are difficulties in recognizing emotions. Conventional machine learning models are ineffective and unable to handle the complexity of human language because they require a great deal of feature engineering. However, even though deep neural networks perform better, they usually need a lot of labeled training data, which is hard to obtain in a real world context @multi_label_emotion.

/*==Our Method*/
One of the ways to address this challenge is with transfer learning, by using pre-trained transformer-based natural language models such as BERT, RoBERTa, DistilBERT, XLNet, etc. These models are trained on massive amounts of data and are capable of learning complex linguistic structures and contextual relationships. Their knowledge can be transferred by fine-tuning them on a specialized emotion-labeled dataset for effective emotion classification with no training from scratch @multi_label_emotion.

/*== Our proposed idea*/
In order to address the following research questions, this study investigated emotion classification in text using DistilBERT-based models @distilbert, a smaller and faster version of BERT.

- What are the challenges in emotion recognition with a human annotated dataset?
- Can fine-tuned models reliably classify emotions from Reddit comments, despite annotator disagreement?
- How does emotion classification performance differ across taxonomies?
- What is the effect of different loss functions on the fine-tuning performance of DistilBERT?

These models were fine-tuned on the GoEmotions dataset @GoEmotionsDatasetOrigin, which contains approximately 58,000 Reddit comments annotated with 28 emotion labels.

This report serves as a comprehensive overview of our study, detailing the dataset and our exploration of it, previous research done on both the GoEmotions dataset and in sentiment and emotion analysis the transformations applied and task formulation, the model architectures and performance metrics, and the results of our experiments. In short, it describes the various issues we found with the dataset, how we solved them, including how we decided to incorporate human annotator disagreements, how we decided to model and evaluate the models, and the general conclusions of the study.

/*
Emotions play a vital role in human communication as the flavor of speech that determines the meaning beyond the literal translation. This makes emotion recognition an important subdomain of natural language processing (NLP), as it enables machines to better interpret human text and speech which is particularly useful in mental health monitoring, customer feedback analysis, and chatbots @EmotionAnalysisinNLP. However, emotion recognition comes with its challenges. Traditional machine learning models require extensive feature engineering, making them inefficient and unable to keep up with the complexity of human language. On the other hand, although deep neural networks offer better performance, they typically require large amounts of labeled training data which is difficult to obtain  @multi_label_emotion.

One solution to this challenge is transfer learning, which offers pre-trained transformer-based models such as BERT, RoBERTa, DistilBERT, and XLNet. These models have been trained on vast amount of data and are capable of understanding complex linguistic structures and contextual relationships. Their knowledge can be transferred by fine-tuning them on a specialized emotion-labeled dataset to achieve efficient emotion classification without training from scratch @multi_label_emotion.

In this study explored emotion classification in text using DistilBERT-based models @distilbert, a smaller and faster variant of BERT, to answer the following research questions:

- What are the challenges in emotion recognition with a human annotated dataset?
- Can fine-tuned models reliably classify emotions from Reddit comments, despite annotator disagreement?
- How does emotion classification performance differ across taxonomies?
- What is the effect of different loss functions on the fine-tuning performance of DistilBERT?

These models were fine-tuned on the GoEmotions dataset @GoEmotionsDatasetOrigin, which contains approximately 58,000 Reddit comments annotated with 28 emotion labels.

This report serves as a comprehensive overview of our study, detailing the dataset and our exploration of it, previous research done on both the GoEmotions dataset and in sentiment and emotion analysis the transformations applied and task formulation, the model architectures and performance metrics, and the results of our experiments. In short, it describes the various issues we found with the dataset, how we solved them, including how we decided to incorporate human annotator disagreements, how we decided to model and evaluate the models, and the general conclusions of the study.
*/

/* LAST SUBMITION
Emotion recognition, a key task within language categorization, is a fundamental component of machine learning, as it seeks to understand and identify how people feel based on what they say or write. In recent years, this field has received growing attention due to advancements in natural language processing and the availability of large scale datasets  @EmotionAnalysisinNLP. Numerous sentiment analysis datasets have emerged, drawn from sources such as Twitter posts, movie reviews, and news headlines@GoEmotionsDatasetOrigin .

In this study, we aim to contribute to the field of emotion recognition by using pre-trained language models to classify emotions in text. Specifically, we will explore the capabilities of the DistilBERT model @distilbert, a smaller and faster version of the BERT model. For this task, we used the GoEmotions dataset @GoEmotionsDatasetOrigin, comprised of approximately $58,000$ Reddit comments sourced from popular English-speaking subreddits, each labeled with one or more of 28 possible emotions, including "neutral". Additionally, we constructed a User Interface so that any other new short sentence written by the user can be predicted as one emotion.

This report serves as a comprehensive overview of our study, detailing the dataset and our exploration of it, previous research done on both the GoEmotions dataset and in sentiment and emotion analysis the transformations applied and task formulation, the model architectures and performance metrics, and the results of our experiments. In short, it describes the various issues we found with the dataset, how we solved them, including how we decided to incorporate human annotator disagreements, how we decided to model and evaluate the models, and the general conclusions of the study.
*/
/* COMMENTED BEFORE 1ST SUBMITION
These emotions can be broadly grouped into positive, negative, and ambiguous categories. During our initial exploration, we observed that some comments were duplicated but labeled differently by annotators, while others had unique texts with multiple assigned emotions or none at all. This subjectivity highlights a key challenge in emotion classification: emotional interpretation often varies between individuals. To address this, we cleaned and standardized the dataset by relabeling unclear or empty tags as "emotion_neutral," aggregating multiple emotion labels by selecting the most frequent one, and retaining only comments reviewed by at least three annotators to improve label reliability.


We also discovered a significant imbalance in the distribution of emotions—neutral labels were dominant, followed by positive emotions, while negative ones were the least represented. This reflects real world emotional expression, where certain feelings are simply more commonly conveyed than others. To train our model, we employed DistilBERT, a lightweight and efficient variant of BERT, particularly suited for faster inference and lower computational costs. This choice makes it feasible to deploy the model in real-time applications, such as chatbots, even on mobile devices. The Reddit comments were relatively short. They were fewer than 30 tokens on average, allowing for efficient tokenization and model processing. // All training was conducted using the Hugging Face Transformers library, which provided powerful tools for fine-tuning the model for emotion classification.
*/

= Dataset 
GoEmotions is the largest human-annotated emotion dataset, with multiple labels per comment to ensure quality @GoEmotionsDatasetOrigin. This section outlines how the dataset was collected, annotated, and processed, following the original scientific publication by its creators @GoEmotionsDatasetOrigin.

A key innovation is its 27-emotion taxonomy, illustrated in Figure 2, based on modern psychological research and going beyond Ekman's six basic emotions. The dataset includes English-only Reddit comments from subreddits containing at least 10k comments. 

#figure(
  image("images/goemotions.PNG"),
  caption: [Snippet of the GoEmotions Dataset.]
)

== Annotation Process
To ensure annotation quality, each comment was reviewed by multiple raters @GoEmotionsDatasetOrigin who categorized the comment into to emotions. Initially, three annotators assessed each comment. If there was no agreement on at least one emotion label, two additional annotators were assigned. All raters were native English speakers from India and they were presented with the comments without author or subreddit information @mislabeled. 


== Taxonomy
It is important for us to know how was the data set collected, put together and cleaned for us to be able to interpret the results correctly.

The 27-category emotion taxonomy of the dataset was inspired by modern psychological research which is far beyond the traditional six basic emotions — joy, anger, fear, sadness, disgust, and surprise — originally proposed by Ekman @GoEmotionsDatasetOrigin.

Comments containing offensive or adult language were removed, except for vulgar comments, which were kept to help study negative emotions. Comments with offensive content toward minorities were manually removed. Only comments with 3 to 30 tokens (including punctuation) were retained. Various techniques were applied to balance the dataset and reduce emotion overrepresentation. Additionally, personal names and religion terms were masked with [NAME] and [RELIGION] tokens, respectively. Note that raters saw the original, unmasked comments during annotation @GoEmotionsDatasetOrigin.

= Literature review // TODO review

The GoEmotions dataset was introduced by #cite(<GoEmotionsDatasetOrigin>, form: "prose"), which outlines 
the motivation, processes, and tools used to create the dataset we are using, along with experiments showcasing its effectiveness. The GoEmotions dataset was introduced to address the lack of sufficiently large datasets for language-based emotion classification and the limitations of existing emotion taxonomies, which typically use limited emotion taxonomies, such as Ekman's 6 emotions @ekman. #cite(<GoEmotionsDatasetOrigin>, form: "author") claims they created the largest human-annotated dataset of 58k carefully selected Reddit comments, labeled with 27 emotion categories or Neutral, as shown on @emotion_list, drawn from popular English subreddits. The dataset stands out for its richer taxonomy, which includes a more diverse range of positive, negative, and ambiguous emotions; in contrast, Ekman’s taxonomy includes only one positive emotion (joy).


The paper explains how the dataset was constructed and presented a baseline BERT-based model for emotion prediction, achieving a $F_1$ of $0.46$ over the proposed 27 emotions taxonomy, but performed better with a 0.64 score using an Ekman-style grouping into six emotion categories and 0.69 using a simple sentiment grouping (positive, neutral, negative) @GoEmotionsDatasetOrigin. These results suggest that the broader the emotion group, the better the accuracy. This new taxonomy proposal inspired us to explore different emotion categories and also confirmed that a BERT based model would be suitable for our aims. The Dataset has been used and analyzed in following studies, with #cite(<GoEmotionsUsedWithBert>, form: "prose") achieving comparable or better results, while comparing different models and a fine-tuned BERT model. For this study, we decided to use DistilBERT @distilbert as our BERT based model, as DistilBERT is a streamlined version of BERT developed by a Hugging Face team that achieves significant reductions in model size and inference time of BERT while maintaining most of its performance.

The paper also examines limitations of the dataset and ways to address them, such as the big class imbalance and biases present in the dataset. We intend to expand on this notion in Section 2. To help with class imbalance, #cite(<GoEmotionsUsedWithBert>, form: "prose") explores data augmentation methods such as Easy Data Augmentation, BERT Embeddings, and Bert-based _ProtAugment_; however, the improvement was marginal, with an increase of 0.027 from the F1 score of the original dataset. Nevertheless, they still achieved a significantly better performance on underrepresented emotion labels, which they attributed to using 10 training epochs instead of #cite(<GoEmotionsDatasetOrigin>, form: "author")'s @GoEmotionsDatasetOrigin 4 epochs.

Emotion Recognition in natural language processing is a complex field, with different nomenclatures, models and frameworks @EmotionAnalysisinNLP. For this report, we use emotion recognition, emotion prediction and emotion classification interchangeably as the classification of one or multiple emotions portrayed in a specific written text. Following #cite(<EmotionAnalysisinNLP>, form: "author")'s @EmotionAnalysisinNLP study, we also outline that we follow the discrete model of emotions, where each emotion is distinct between each other; however, opposed to other studies in emotion classification @disagreements, we will take advantage of the dataset's collection methodology @GoEmotionsDatasetOrigin and use both the multi-label facet and human label variation available to avoid masking the degree to which annotators disagree @disagreements. We will elaborate this further in the sections that follow.



/* 
Other papers that talk about emotion classificaiton (only add if you feel like it)

- https://aclanthology.org/2021.naacl-main.375.pdf (uses goemotions)
- 
*/

#figure(
  image("images/emotions.PNG"),
  caption: [Emotions Comprised in the Dataset.]) <emotion_list>

= Methodology
In this section, we outline the approach and tools used to carry out our study, from data preprocessing to model training and evaluation. The entire workflow consists of Data exploration, preprocessing from data driven decisions, modeling, and evaluating. Additionally, we developed a user interface to visualize the output of these models. The entire process, including model implementations, is available in #link("https://github.com/yanazlatanova/emotion-recognition").

== Data Exploration and Preprocessing

=== *Dataset Review*

Our initial data exploration, along with a review of the paper by the dataset creators @GoEmotionsDatasetOrigin, helped us better understand the dataset and informed our pre-processing strategy.

We made a graph of the distribution of the length of the comments as we can see on @textdistrib. We found that the distribution was normal and that most of the comment length was around 60 characters. When we selected the comments of length inferior to 4 characters, it showed "yes/no" comments or emoticons eg. such as _:^)_ and _XD_.

#figure(
  image("images/textdistrib.PNG"),
caption: [Comment Length Distribution]) <textdistrib>

=== *Investigating Labeling Disagreement*
#v(1em)
We found that each comment/text had multiple duplicates, each rated by different annotators. While there were no missing values, some rows were marked as either having no assigned emotion or labeled as "Neutral", often reflecting that the emotion of the comment was unclear. According to #cite(<GoEmotionsDatasetOrigin>, form: "prose"), _"If raters were not certain about any emotion being expressed, they were asked to select Neutral. We included a checkbox for raters to indicate if an example was particularly difficult to label, in which case they could select no emotions."_

However, in some cases, raters still assigned an emotion to a comment that another rater found unclear or believed expressed no emotion. This introduced challenges in interpreting such instances.

To further analyze the multiple raters per text, we plotted the number of raters per comment, which can be seen in @unique-raters-histogram. We could see that most texts were rated by three annotators, but some had only one or two.

#figure(
  image("images/unique-raters-histogram.png", width: 75%),
  caption: [ Number of unique raters per comment/text.] // TODO image shows 1 - 6 instead of categories
)<unique-raters-histogram>

Additionally, these raters often disagree in the emotions they assign, even in seemingly contradictory emotions. For example, the comment _"Definitely was a nonononoyes#footnote[nonononoyes is a name of a subreddit dedicated to sharing videos that depict situations initially appearing to go wrong ("no, no, no...") but ultimately result in a surprisingly positive or successful outcome ("yes!").] for me there lol, I'm a horrible person"_  got as fear, amusement, approval and disgust, from 5 different annotators. This noise, while problematic, is expected, since emotion classification is fairly subjective and there's multiple plausible answers @variation, annotators' lived experiences influence their interpretations @disagreements, and context is often necessary to access correctly the emotion (which the annotators did not have access to @GoEmotionsDatasetOrigin), and domain knowledge (which most annotators might've not had, because of the "ever-evolving ethos" of Reddit's culture, both site-wide and within each subreddit @reddit). On top of this, this dataset has been criticized before for its reliability @unknown_reliability @mislabeled, with specific examples from #cite(<mislabeled>, form: "prose") outlying issues on comments containing profanity, sarcasm, internet style conventions, and culturally specific references. From, this we realize that our model may struggle to accurately predict emotions in future inputs, especially on inputs that contain these.
#v(1em)
=== *Applying Labeling Disagreement Strategy*
#v(1em)
Based on these facts, we decided to:
- *Label unclear cases consistently*: We treated both "Neutral" and empty emotion labels as "Unclear", united under the _unclear_ label, since in both situations raters were unable to confidently identify an emotion.

- *Filter by rater count:* The dataset was filtered to include only comments with at least three raters, ensuring more reliable and confident annotations. As shown in @unique-raters-histogram, multiple comments had only one or two raters.

- *Soft Label Aggregation*: Since identical comments were often assigned different labels by different raters, we chose to aggregate the emotion ratings for each unique comment, embracing human label variation @variation. For instance, if the comment "this is adorable" received the following annotations:
  - *Rater 1:* [admiration, joy];
  - *Rater 2:* [admiration];
  - *Rater 3:* [amusement].

  The aggregated label distribution would be:
    - *admiration:* 0.67 (2 out of 3 raters);
    - *amusement:* 0.33 (1 out of 3 raters).
    - *joy*: 0.33 (1 out of 3 raters)

This aggregation gives more weight to emotions confirmed by multiple raters while still capturing the presence of less-agreed-upon emotions. It preserves the richness of the label diversity without resorting to semi-supervised learning. Essentially, it gives us a confidence level of the emotions per text. We found this preferable to majority vote aggregation since it preserves different annotators' perspectives @disagreements, while avoiding the different issues that arise with assigning a "ground truth" label to a text @disagreements @alm @variation

While annotator disagreement and emotion uncertainty has been explored before @label_quality @unknown_reliability @black_white @disagreements, this way of transforming the dataset seems to be a novel approach in treating annotator disagreement in both the GoEmotions dataset and emotion recognition, as it seems quite more to aggregate label disagreements into a single one @black_white since it introduces uncertainty into emotion recognition, transforming our multi label classification task into multi label regression problem, using "soft" labels instead. Arguably, this also gives a hierarchical label structure to the emotions @are_we_really, enabling this dataset to be used for point-wise learning to rank tasks. While this is not our priority, our metrics and models will take this into account as well.

// During our data exploration, we also reviewed relevant literature and discovered that the GoEmotions dataset contains some mislabeling issues. Since the dataset is primarily composed of comments from English-speaking subreddits, many of the texts include references to U.S. culture that may be unfamiliar or misinterpreted by the annotators, who are primarily based in India. Additionally, sarcastic remarks are often mislabeled, introducing bias into the data. Another major limitation is the lack of contextual metadata. Many comments are replies to images or other posts, but the raters are do not take the context into consideration, probably due to an overload of comments and a lack of time. As a result, we realized that our model may struggle to accurately predict emotions in future inputs, especially when dealing with sarcasm or culturally specific references.
#v(1em)
=== *Applying Emotion Taxonomies*
#v(1em)
Since emotions are complex and multidimensional, researchers have created a number of taxonomies to better classify and analyze them.

Three different emotion classification schemes were used in this study:

- Ekman's Basic Emotions taxonomy @ekman;
- A sentiment-based taxonomy;
- The original taxonomy of GoEmotions.

To each taxonomy, a seventh category label “unclear” was added to account for comments that are either ambiguous, emotionally neutral, or inconsistently labeled by human raters. This category helped to manage noise in the dataset, especially given the subjectivity involved in interpreting emotions from short texts.

By narrowing our classification to these seven categories, we aim to strike a balance between theoretical soundness and practical model performance, while acknowledging the limitations of emotional ambiguity in natural language.

The dataset was duplicated into three separate versions, remapping the labels according to the structure of each taxonomy. The same process outlined in #cite(<GoEmotionsDatasetOrigin>, form: "prose") was used for the label aggregation. These versions were used in the subsequent experiments and reflected in the results.


#v(3em) // more spacing just bcs of the page
=== *Exploring GoEmotions taxonomy*
#v(1em)
The GoEmotions dataset has an uneven number of comments across emotion categories, as shown in @examples-per-emotion-count. This makes some emotions underrepresented in the dataset, potentially resulting in a worse model performance on those categories.  Because of this imbalance, our performance metrics must account for it to better evaluate how the model performs on underrepresented emotions—especially important in the context of single emotion prediction. This is also something #cite(<GoEmotionsUsedWithBert>, form: "prose") noticed while fine-tuning a BERT model on the GoEmotions dataset, where the "grief" label, which has the least sample size in their training set, achieved the worst performance across different evaluation metrics @GoEmotionsUsedWithBert.

// == Emotion groups Correlation // TODO consider writing abt taxonomies here instead
As shown in @conf_mat some emotions are correlated as  indicated by darker shades in the confusion matrix. Annoyance and anger are very much linked, as well as nervousness and fear, sadness and disappointment or joy and excitement to cite a few examples. It can be explained by the fact that some emotions are verbally implicit and need more context to be interpreted.

In their work on the GoEmotions taxonomy, #cite(<GoEmotionsDatasetOrigin>, form: "prose") used a number of methods to investigate the consistency of emotion labeling. By using hierarchical clustering, they discovered that emotions naturally form groups based on sentiment polarity and intensity; for example, "ambiguous" emotions like surprise tended to cluster more closely with positive emotions. They used Principal Preserved Component Analysis (PPCA) to evaluate rater agreement and uncover deeper patterns, and the results indicated that all 27 emotion categories were highly distinct, which is an exceptionally strong finding in emotion research and motivated our choice of GoEmotion's taxonomy.

#figure(
image("diagramNew.png"),
caption: [Model architecture diagram.]
)<diagram>

#figure(
  image("images/examples-per-emotion-count.png", width: 60%),
  caption: [ Distribution emotion categories.]
)<examples-per-emotion-count>

#figure(image("images/confution-matrix.png"), caption: [Confusion matrix])<conf_mat> // TODO write caption


== Modeling and Tokenization
The Hugging Face Transformers library was used to implement DistilBERT. It is well suited for emotion classification tasks due to its ability to preserve much of BERT's performance while being more efficient. Before feeding the text into the model, we used Hugging Face’s tokenizer to convert sentences into token IDs. The tokenizer handles out-of-vocabulary words by breaking them into word units, ensuring even rare or misspelled terms are processed effectively.

Our models consist on fine-tuning DistilBERT to a multi-label regression task. Specifically, the text is tokenized (using DistilBert's tokenizer) and then the tokens and attention mask are fed into DistilBert's; the output of that model is fed into 2 dense layers, separated by a dropout layer, and concludes with a sigmoid activation layer over the output logits, to draw predictions per emotion#footnote[We find important to note that this is preferable over a softmax activation layer because the texts can have multiple emotions.]. To train these models, the DistilBert layers are frozen, to not only reduce computation times but also to leverage the power of the pre trained transformer. An overview of the model is shown on @diagram. 



== Training Setup
Training was conducted using PyTorch as the backend. The dataset was split into training, validation, and test sets with 70%/15%/15% ration, without any stratification, to ensure a fair assessment of the model’s generalization capability. 

Each model was trained for a fixed duration of 10 minutes on the same machine, using the Adam optimizer and a constant learning rate. The model was trained and evaluated using standard classification metrics that gave us information on the performance of the model, the general error and check for possible overconfidence.

== Model Variants
To train our models, we decided to employ different loss functions as to optimize the modeling for different objectives. These loss functions are closely related to our performance metrics, which will be introduced better in section 5.1.

#let Mbce = [_M#sub[BCE]_]
#let Mdcg = [_M#sub[DCG]_]
#let Msme = [_M#sub[MSE]_]
#let ndcg = [nDCG]

- Our first model #Mbce use the Mean Binary Cross Entropy (MBCE) as the loss function, to analyze the general potential of the model.

- The second model #Mdcg uses a SoftRank-style @softrank differentiable approximation of the Normalized Discounted Cumulative Gain (#ndcg) for the loss function, to optimize for the expected #ndcg. This model in theory should be better in ranking the emotions, while still giving relevance/confidence scores.

- The third model #Msme will use the Weighted Mean Squared Error (WSME) as our loss function, giving it the power to make more bold predictions while still penalizing aggressively wrong predictions.

In parallel, every model was trained on the 3 different emotion taxonomies previously discussed, denoted in this report as $M^3$, $M^7$, and $M^28$.
We generally expect for the $M^3$ models to have a higher performance, due to the generalized nature of the taxonomy, followed by the $M^7$ models for the same reason. With that being said, we still found interesting to share the performance of the $M^28$ models, as the unique taxonomy associated can share more specific emotions associated with the texts beyond sentiment and the simpler taxonomies. Additionally, as we're using annotator disagreement for the prediction directly, which is fairly uncommon (and potentially novel in emotion recognition) as discussed before, the results cannot be directly compared with ones from other articles that use this dataset, due to the big difference in the annotator aggregation. Finally, we don't expect great results in general, not only due to rig and time and constraints, but also due to the dataset quality, as discussed before.

== Output Visualization
Streamlit was used to create a basic user interface to present our models' predictions in an interactive way. Users can enter custom text into the interface and see the predicted emotion label and related confidence score right away. This makes it easier to understand how different emotion categories are assigned. The interface can be seen in @ui.

#figure(
  image("ui2.png"),
  caption: [User interface for emotion recognition of input text.])<ui>

/* LAST SUBMITION
We used the Hugging Face Transformers library to implement DistilBERT. It is well suited for emotion classification tasks due to its ability to preserve much of BERT's performance while being more efficient. Before feeding the text into the model, we used Hugging Face’s tokenizer to convert sentences into token IDs. The tokenizer handles out-of-vocabulary words by breaking them into word units, ensuring even rare or misspelled terms are processed effectively.

#figure(
  image("ui.png"),
  caption: [User interface for emotion recognition of input text build using Streamlit.])<ui>
  
Training was conducted using PyTorch as the backend. We split the dataset into training, validation, and test sets to ensure a fair assessment of the model’s generalization capability. We trained the model and evaluated it using standard classification metrics that gave us information on the performance of the model, the general error and check for possible overconfidence.

== Data Exploration and Pre-processing Decisions
Our initial data exploration, along with a review of the paper by the dataset creators @GoEmotionsDatasetOrigin, helped us better understand the dataset and informed our pre-processing strategy.

We made a graph of the distribution of the length of the comments as we can see on @textdistrib. We found that the distribution was normal and that most of the comment length was around 60 characters. When we selected the comments of length inferior to 4 characters, it showed "yes/no" comments or emoticons eg. such as _:^)_ and _XD_.

#figure(
  image("images/textdistrib.PNG"),
caption: [Comment Length Distribution]) <textdistrib>

We found that each comment/text had multiple duplicates, each rated by different annotators. While there were no missing values, some rows were marked as either having no assigned emotion or labeled as "Neutral", often reflecting that the emotion of the comment was unclear. According to #cite(<GoEmotionsDatasetOrigin>, form: "prose"), "If raters were not certain about any emotion being expressed, they were asked to select Neutral. We included a checkbox for raters to indicate if an example was particularly difficult to label, in which case they could select no emotions."However, in some cases, raters still assigned an emotion to a comment that another rater found unclear or believed expressed no emotion. This introduced challenges in interpreting such instances.

To further analyze the multiple raters per text, we plotted the number of raters per comment, which can be seen in @unique-raters-histogram. We could see that most texts were rated by three annotators, but some had only one or two.

#figure(
  image("images/unique-raters-histogram.png", width: 80%),
  caption: [ Distribution of the number of unique raters per comment/text.] // TODO image shows 1 - 6 instead of categories
)<unique-raters-histogram>

Additionally, these raters often disagree in the emotions they assign, even in seemingly contradictory emotions. For example, the comment _"Definitely was a nonononoyes#footnote[nonononoyes is a name of a subreddit dedicated to sharing videos that depict situations initially appearing to go wrong ("no, no, no...") but ultimately result in a surprisingly positive or successful outcome ("yes!").] for me there lol, I'm a horrible person"_  got as fear, amusement, approval and disgust, from 5 different annotators. This noise, while problematic, is expected, since emotion classification is fairly subjective and there's multiple plausible answers @variation, annotators' lived experiences influence their interpretations @disagreements, and context is often necessary to access correctly the emotion (which the annotators did not have access to @GoEmotionsDatasetOrigin), and domain knowledge (which most annotators might've not had, because of the "ever-evolving ethos" of Reddit's culture, both site-wide and within each subreddit @reddit). On top of this, this dataset has been criticized before for its reliability @unknown_reliability @mislabeled, with specific examples from #cite(<mislabeled>, form: "prose") outlying issues on comments containing profanity, sarcasm, internet style conventions, and culturally specific references. From, this we realize that our model may struggle to accurately predict emotions in future inputs, especially on inputs that contain these.

Based on these facts, we decided to:
- *Label unclear cases consistently*: We treated both "Neutral" and empty emotion labels as "Unclear", united under the _unclear_ label, since in both situations raters were unable to confidently identify an emotion.

- *Filter by rater count:* The dataset was filtered to include only comments with at least three raters, ensuring more reliable and confident annotations. As shown in @unique-raters-histogram, multiple comments had only one or two raters.

- *Aggregated Ratings*: Since identical comments were often assigned different labels by different raters, we chose to aggregate the emotion ratings for each unique comment, embracing human label variation @variation. For instance, if the comment "this is adorable" received the following annotations:
  - *Rater 1:* [admiration, joy];
  - *Rater 2:* [admiration];
  - *Rater 3:* [amusement].

  The aggregated label distribution would be:
    - *admiration:* 0.67 (2 out of 3 raters);
    - *amusement:* 0.33 (1 out of 3 raters).
    - *joy*: 0.33 (1 out of 3 raters)

This aggregation gives more weight to emotions confirmed by multiple raters while still capturing the presence of less-agreed-upon emotions. It preserves the richness of the label diversity without resorting to semi-supervised learning. Essentially, it gives us a confidence level of the emotions per text. We found this preferable to majority vote aggregation since it preserves different annotators' perspectives @disagreements, while avoiding the different issues that arise with assigning a "ground truth" label to a text @disagreements @alm @variation

While annotator disagreement and emotion uncertainty has been explored before @label_quality @unknown_reliability @black_white @disagreements, this way of transforming the dataset seems to be a novel approach in treating annotator disagreement in both the GoEmotions dataset and emotion recognition, as it seems quite more to aggregate label disagreements into a single one @black_white since it introduces uncertainty into emotion recognition, transforming our multi label classification task into multi label regression problem, using "soft" labels instead. Arguably, this also gives a hierarchical label structure to the emotions @are_we_really, enabling this dataset to be used for point-wise learning to rank tasks. While this is not our priority, our metrics and models will take this into account as well.

// During our data exploration, we also reviewed relevant literature and discovered that the GoEmotions dataset contains some mislabeling issues. Since the dataset is primarily composed of comments from English-speaking subreddits, many of the texts include references to U.S. culture that may be unfamiliar or misinterpreted by the annotators, who are primarily based in India. Additionally, sarcastic remarks are often mislabeled, introducing bias into the data. Another major limitation is the lack of contextual metadata. Many comments are replies to images or other posts, but the raters are do not take the context into consideration, probably due to an overload of comments and a lack of time. As a result, we realized that our model may struggle to accurately predict emotions in future inputs, especially when dealing with sarcasm or culturally specific references.

Finally, we also replicated the dataset into 3 separate datasets, where the emotion taxonomies previously mentioned were recreated using the same aggregation used by #cite(<GoEmotionsDatasetOrigin>, form: "prose"). These will be reflected in the results.

== Uneven emotion categories
The GoEmotions dataset has an uneven number of comments for each emotion category, as shown in @examples-per-emotion-count, which makes some emotions underrepresented in the dataset, potentially resulting in a worse performance on underrepresented emotions. Because of this unevenness, our performance metrics need to take imbalance into account, to make us undertand the performance of the model in underrepresented emotions, since this is especially important in predicting emotions reliably in the context of single emotion prediction. This is also something #cite(<GoEmotionsUsedWithBert>, form: "prose") noticed while fine-tuning a BERT model on the GoEmotions dataset, where the "grief" label, which has the least sample size in their training set, achieved the worst performance across different evaluation metrics @GoEmotionsUsedWithBert.

#figure(
  image("images/examples-per-emotion-count.png", width: 50%),
  caption: [ Distribution emotion categories.]
)<examples-per-emotion-count>


== Emotion groups Correlation // TODO consider writing abt taxonomies here instead
We can see in @conf_mat that some emotions are correlated as the dark color tell us. Annoyance and anger are very much linked, as well as nervousness and fear, sadness and disappointment or joy and excitement to cite a few examples. It can be explained by the fact that some emotions are verbally implicit and need more context to be interpreted.
#figure(image("images/confution-matrix.png"), caption: [Confusion matrix])<conf_mat> // TODO write caption

In analyzing the GoEmotions dataset, Alba Curry et al. @EmotionAnalysisinNLP previously employed several techniques to better understand the consistency of emotion labeling. Through hierarchical clustering, they discovered that emotions naturally group by intensity and sentiment polarity, for example “ambiguous” emotions like surprise, cluster closer to positive emotions. To evaluate rater agreement and uncover deeper patterns, they applied Principal Preserved Component Analysis (PPCA), which showed all 27 emotion categories to be highly distinct, which is an unusually strong result in emotion research. To further explore how emotions are organized, they used t-SNE, a dimensionality reduction method, to visualize how emotion labels relate in space. Lastly, they analyzed the linguistic features tied to each emotion by examining which words were statistically most associated with each category. They found that emotions with clear lexical markers—like gratitude being linked to “thanks” showed higher inter-rater agreement, while more context dependent emotions such as grief or nervousness were harder to label consistently. These findings highlight both the richness and the limitations of text-based emotion annotation.


== Emotion Taxonomies/Grouping emotions

Emotions are complex and multifaceted, and researchers have proposed various taxonomies to categorize and study them effectively. One of the most influential models is Paul Ekman’s basic emotion theory (Ekman, 1992), which identifies six universal emotions—anger, disgust, fear, happiness, sadness, and surprise—based on cross-cultural facial expression studies. Ekman’s framework is widely used in psychology and computational emotion analysis for its simplicity and empirical grounding.

Another widely cited taxonomy is Plutchik’s Wheel of Emotions (Plutchik, 1980), which organizes emotions in a circular structure based on intensity and similarity. Plutchik identifies eight primary emotions—joy, trust, fear, surprise, sadness, disgust, anger, and anticipation—each with varying degrees and opposites. This model is particularly useful in visualizing relationships between emotions and understanding how complex emotions arise from combinations of more basic ones.

Beyond these foundational models, more recent work by Bostan and Klinger (2018) aggregates 14 commonly used emotion classification schemes to analyze and unify emotion annotation practices in NLP. Their comparative study emphasizes how emotion categories vary across datasets, revealing discrepancies in granularity, terminology, and theoretical underpinnings. This work underscores the importance of standardizing emotional labels, especially in machine learning contexts, where inconsistent categorization can lead to ambiguous or biased model predictions.

Together, these taxonomies reflect the diversity in how emotions can be defined, labeled, and interpreted—highlighting the challenges and considerations in building accurate emotion recognition systems.

In our study, we chose to adopt Ekman’s six basic emotions as the foundation for our classification task. This decision was made to simplify the emotion space while maintaining a strong grounding in psychological theory. Additionally, we introduced a seventh category labeled as “unclear” to account for comments that are either ambiguous, emotionally neutral, or inconsistently labeled by human raters. This category helps manage noise in the dataset, especially given the subjectivity involved in interpreting emotions from short texts.

By narrowing our classification to these seven categories, we aim to strike a balance between theoretical soundness and practical model performance, while acknowledging the limitations of emotional ambiguity in natural language.
*/

= Evaluation

Due to the nature of uncertainty prediction and emotion complexity, it is critical that we don't focus on single metrics to evaluate the performance of our models, as that "gives no indication on how reasonable a model is, yet alone how confident and trustworthy it is" @variation. For that end, we implemented multiple different performance metrics, to measure the performance of our models in different aspects.

== Performance metrics

Overall, we considered and used 5 performance metrics, all of them to evaluate and compare different things. These were the following:

- Mean Binary Cross Entropy (MBCE), designed to give us information on the general error of the model, as well as its overconfidence;
/*

$
  "BCE"(y_e, hat(y_e)) &= -(y_e log(hat(y_e)) + (1-y_e)log(1-hat(y_e))) \
  "MBCE"(bold(y), bold(hat(y))) &= (sum_(e = 1)^(\#bold(y))"BCE"(y_e, hat(y_e)))/(\#y)
$
*/

- Weighted Mean Squared Error (WMSE, $w(y,hat(y)) = e^y$), designed to check the under-confidence of the model, incentivizing it to be more bold with its predictions, while still taking into account high errors;

- Normalized Discounted Cumulative Gain (#ndcg), commonly metric in information retrieval and in learning to rank tasks, to measure the model in a raking scenario, by evaluation the actual actual ranking to the ideal one, rewarding highly relevant items that appear earlier in the list, enabling us to understand if the model is correctly giving higher confidence scores to higher confidence emotions even in low confidence scenarios
- 2 macro-averaged $F_1$ metrics, differentiating on the true label definitions we use, designed to evaluate our model in a classification scenario, while giving us more insight on the ability of the model to predict less common emotions.
  
For these $F_1$ metrics, as the predicted labels, we decided on using 0.5 as the threshold for a positive or negative prediction; for the ground truth, for the metric $F_1^("any")$, we say a label is positive if any of the annotators rated as such, and for the metric $F_1^("conf")$, we say that the emotion with the most confidence, and every emotion with more than 0.8 confidence, are positive. // TODO add justification?

/*
As regular multi label classification metrics, we're going to use 2 different macro-averaged $F_1$ metrics, differentiating on the true label definitions we use. As the predicted labels, we decided on using 0.5 as the threshold for a positive or negative prediction; for the ground truth, for the metric $F_1^("any")$, we say a label is positive if any of the annotators rated as such, and for the metric $F_1^("conf")$, we say that the emotion with the most confidence, and every emotion with more than 0.8 confidence, are positive. The macro averaged $F_1$ is a useful classification metric in this case because it takes into account the imbalance of the dataset, giving us more insight on the ability of the model to predict less common emotions. // TODO make example maybe

Additionally, we employ a weighted mean squared error (WMSE), with a weight function designed to penalize errors on higher ground truth confidences. This metric is designed to check the under-confidence of the model, incentivizing the model to be more bold with their predictions, while still taking into account high errors. It can be calculated as the following:


$ "WMSE"(y,hat(y)) = e^(y) (y-hat(y))^2 $
This weight function was considered because of its exponential increase, since higher confidence means more annotators agreed on the emotion it seemed appropriate. @weights shows other considered weight functions.
#figure(table(
  columns:4, stroke: none, align: center,
  $y+1$,   $e^y$,  $2^y$, $e^(y/2)$,
  table.hline(),
  $1.00$, $1.00$, $1.00$, $1.00$,
  $1.20$, $1.22$, $1.15$, $1.11$,
  $1.40$, $1.49$, $1.32$, $1.22$,
  $1.60$, $1.82$, $1.52$, $1.35$,
  $1.80$, $2.22$, $1.78$, $1.49$,
  $2.00$, $2.72$, $2.00$, $1.65$
), caption: [Different weight functions]) <weights>

Finally, we employ the , which is a common metric in information retrieval and in learning to rank tasks. The metric measures the quality of a ranked list by comparing the actual ranking to the ideal one, rewarding highly relevant items that appear earlier in the list, while still taking ground truth scores/confidences into account. While our task is not strictly a learning to rank task, the metric enables us to understand if the model is correctly giving higher confidence scores to higher confidence emotions even in low confidence scenarios, while still taking into account the ground truth confidence scores. While making sure that $bold(hat(y))$ is sorted before calculating the DCG, this metric is calculated as the following:

$ 
  "DCG"(bold(y),bold(hat(y))) &= sum_(i=1)^(\#bold(hat(y))) (2^(y_i) - 1)/log(i+1) \
  ndcg(bold(y), bold(hat(y))) &= "DCG"(bold(y),bold(hat(y)))/"DCG"(bold(y),bold(y))
$
*/
== Results and Discussion

#import table: cell
#import calc: round
#let results = json("results.json")
#figure(placement: auto, scope: "parent", table(columns: 10, stroke: none, align: (x,y) => {if (x == 0){right} else {center}},
  [], table.vline(),cell(colspan: 3)[3 emotions], table.vline(),cell(colspan: 3)[7 emotions],table.vline(), cell(colspan: 3)[28 emotions],
  [], $Mbce^3$, $Mdcg^3$, $Msme^3$, $Mbce^7$, $Mdcg^7$, $Msme^7$, $Mbce^28$, $Mdcg^28$, $Msme^28$,
  table.hline(),
  [MBCE #sym.arrow.b], 
    [*#round(results.BCE_loss.at(0).test_cross_entropy, digits:3)*], 
    [#round(results.nDCG_loss.at(0).test_cross_entropy, digits:3)], 
    [#round(results.WMSE_loss.at(0).test_cross_entropy, digits:3)],
    [*#round(results.BCE_loss.at(1).test_cross_entropy, digits:3)*], 
    [#round(results.nDCG_loss.at(1).test_cross_entropy, digits:3)], 
    [#round(results.WMSE_loss.at(1).test_cross_entropy, digits:3)],
    [*#round(results.BCE_loss.at(2).test_cross_entropy, digits:4)*], 
    [#round(results.nDCG_loss.at(2).test_cross_entropy, digits:3)], 
    [#round(results.WMSE_loss.at(2).test_cross_entropy, digits:3)],
  [$F_1^"any"$ #sym.arrow.t],
    [#round(results.BCE_loss.at(0).test_f1_standard, digits:3)],
    [#round(results.nDCG_loss.at(0).test_f1_standard, digits:3)],
    [*#round(results.WMSE_loss.at(0).test_f1_standard, digits:3)*],
    [#round(results.BCE_loss.at(1).test_f1_standard, digits:3)],
    [*#round(results.nDCG_loss.at(1).test_f1_standard, digits:3)*],
    [#round(results.WMSE_loss.at(1).test_f1_standard, digits:3)],
    [#round(results.BCE_loss.at(2).test_f1_standard, digits:3)],
    [*#round(results.nDCG_loss.at(2).test_f1_standard, digits:3)*],
    [#round(results.WMSE_loss.at(2).test_f1_standard, digits:3)],
  [$F_1^"conf"$ #sym.arrow.t],
    [*#round(results.BCE_loss.at(0).test_f1_interesting, digits:3)*],
    [#round(results.nDCG_loss.at(0).test_f1_interesting, digits:3)],
    [#round(results.WMSE_loss.at(0).test_f1_interesting, digits:3)],
    [*#round(results.BCE_loss.at(1).test_f1_interesting, digits:3)*],
    [#round(results.nDCG_loss.at(1).test_f1_interesting, digits:3)],
    [#round(results.WMSE_loss.at(1).test_f1_interesting, digits:3)],
    [#round(results.BCE_loss.at(2).test_f1_interesting, digits:3)],
    [#round(results.nDCG_loss.at(2).test_f1_interesting, digits:3)],
    [*#round(results.WMSE_loss.at(2).test_f1_interesting, digits:3)*],
  [WMSE #sym.arrow.b],
    [#round(results.BCE_loss.at(0).test_weighted_mse, digits:3)],
    [#round(results.nDCG_loss.at(0).test_weighted_mse, digits:3)],
    [*#round(results.WMSE_loss.at(0).test_weighted_mse, digits:3)*],
    [#round(results.BCE_loss.at(1).test_weighted_mse, digits:4)],
    [#round(results.nDCG_loss.at(1).test_weighted_mse, digits:3)],
    [*#round(results.WMSE_loss.at(1).test_weighted_mse, digits:4)*],
    [#round(results.BCE_loss.at(2).test_weighted_mse, digits:3)],
    [#round(results.nDCG_loss.at(2).test_weighted_mse, digits:3)],
    [*#round(results.WMSE_loss.at(2).test_weighted_mse, digits:3)*],
  [nDCG #sym.arrow.t],
    [#round(results.BCE_loss.at(0).test_ndcg, digits:3)],
    [#round(results.nDCG_loss.at(0).test_ndcg, digits:3)],
    [*#round(results.WMSE_loss.at(0).test_ndcg, digits:3)*],
    [*#round(results.BCE_loss.at(1).test_ndcg, digits:3)*],
    [#round(results.nDCG_loss.at(1).test_ndcg, digits:3)],
    [#round(results.WMSE_loss.at(1).test_ndcg, digits:3)],
    [*#round(results.BCE_loss.at(2).test_ndcg, digits:3)*],
    [#round(results.nDCG_loss.at(2).test_ndcg, digits:3)],
    [#round(results.WMSE_loss.at(2).test_ndcg, digits:3)],
), caption: [The results from our models]) <results>

After training and testing the various models using the workflow described in the previous section, we obtained the results in @results. From them, we can make several conclusions: 

- As expected, the taxonomies with more emotions had a harder time in performing better, except when comparing the WMSE metric results; this probably means that the weight chosen wasn't penalizing lower ground truths hard enough, or this low values might be a reflection of class imbalance.

- All models are generally good at ranking the emotions, as the nDCG is high across all models. While this is not surprising in the 3 and 7 emotion taxonomies, the 28 emotion taxonomy having such high values suggest that the model is great at predicting the top emotions of the comments.

- Unexpectedly, the #Mdcg models aren't the best ones on the nDCG metric (even though they're all similar values between them), which suggests either an issue in implementation, the unsuitability of the approximated nDCG constructed, or that the model might be overfitting if optimizing for that metric. The overall performance observed in the rest of the metrics (except in the $F_1^"any"$) might suggest this overfitting hypothesis.

- Surprisingly, it seems that taxonomies with lower emotions have higher MBCE and WMSE than the higher emotion number taxonomy. This seems counter intuitive, but it's probably because of the big imbalance of labels per text (as most emotions should be 0). This suggests that alternative ways to model to be aware of this imbalance might be more appropriate, as well the importance of choosing a more appropriate weight function for the WMSE.

- The classification metrics pretty unstable, showing fairly different values between the different models. Regardless, from them, we can see that the models aren't as good in predicting the multiple labels directly. The #Mdcg, weirdly enough, seems to be able to handle this better than the other emotions, when looking at the $F_1^"any"$ metric. This might be because the model gives more confidence in the output for the emotions, as the order in the metric is more important, and there are more positive labels for the threshold we decided. This would also explain the fairly low $F_1^"conf"$ throughout the models (including #Mdcg), possibly suggesting that a better threshold for it should've been chosen.

- Both the #Mbce and #Msme models perform very similarly according to the regression metrics, and both seem to be capable of predicting the emotion of the texts close to the real predictions.

Overall, the results we're in line with our expectations, and from them it seems that using and fine-tuning Distilbert is suitable for emotion recognition when taking into account annotator disagreement. 

= Conclusion
In this report, we analyzed the GoEmotions dataset with the objective of doing emotion recognition on different emotion taxonomies, these comprised of 3, 6 and 27 emotions, with the latter ones including a label for unclear. While researching and analyzing, we found some issues of the dataset, including annotator disagreement, data quality issues, and label imbalance. After aggregating the annotators based on confidence per label, we fine-tuned DistilBERT using different loss functions on those taxonomies. We could conclude that for most loss functions, the models mostly performed fairly similar between loss functions, but the different taxonomies had a huge impact on model performance. Ultimately, we were able create models that somewhat classify and predict the emotions in Reddit comments accurately, even with heavy annotator disagreement. 

In future work, there are several key directions we should explore to improve both the accuracy and reliability of emotion recognition in our models. First, we should address the label noise and cultural bias in the GoEmotions dataset. As noted during our analysis, many labels appear to be inconsistent due to annotator misunderstanding of cultural references or sarcasm. A valuable improvement would be to refine or relabel parts of the dataset using more culturally diverse annotators or by adding contextual information (e.g., surrounding conversation or media references) to help raters make more accurate judgments.

Second, although we used DistilBERT for its efficiency, experimenting with larger or more specialized language models like RoBERTa, DeBERTa, or emotion-specific transformers could help us capture more nuanced emotional signals, especially for complex cases like sarcasm or mixed emotions. We could also explore prompt-tuned or instruction-following models, which might generalize better with less fine-tuning. Additionally, data augmentation techniques has shown to improve model performance and lift the burden somewhat from label imbalance @GoEmotionsUsedWithBert @data_aug @small_imb.

Third, to improve the model’s robustness and calibration, we should consider applying calibration techniques such as temperature scaling or isotonic regression before training, especially since our evaluation showed that while confidence scores were often reasonable, some configurations still showed signs of over- or under-confidence.

Finally, we should perform a more thorough error analysis. For instance, looking into which specific emotions are most often misclassified, and under what linguistic conditions, would help us fine-tune the model and dataset further. Another option is to use model explainability techniques like the use of shapley values @shap or LIME @LIME. Future work could also expand on the hard emotion distinctions possibly incorporating label correction @noise_corr, or to incorporate fuzzy probability theory into the mix.