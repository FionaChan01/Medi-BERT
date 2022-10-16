# Medi-BERT

This repository contains the code for our [*National Innovation Project*](http://gjcxcy.bjtu.edu.cn/NewLXItemListForStudentDetail.aspx?ItemNo=785167), which focuses on the problem of how to accurately triage the patients in the hospital to a specific department; the Project was rated as Good.

* [Medi-BERT](#medi-bert)
   * [<g-emoji class="g-emoji" alias="one" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/0031-20e3.png">1⃣</g-emoji>️ Background](#1⃣️-background)
   * [<g-emoji class="g-emoji" alias="two" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/0032-20e3.png">2⃣</g-emoji>️ Algorithm Design](#2⃣️-algorithm-design)
      * [1) Model Structure](#1-model-structure)
      * [2) Data Augmentation](#2-data-augmentation)
      * [3) Inference](#3-inference)
   * [<g-emoji class="g-emoji" alias="three" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/0033-20e3.png">3⃣</g-emoji>️ Result](#3⃣️-result)
   * [<g-emoji class="g-emoji" alias="four" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/0034-20e3.png">4⃣</g-emoji>️ Resources](#4⃣️-resources)

## 1⃣️ Background

> When first entering a hospital, patients are not sure which department they should visit, so hospital guidance station is prone to congestion during peak periods, especially in the context of the current epidemic, which may cause a series of problems such as the transmission of diseases, bringing many inconveniences to patients and hospitals. 

The frequently model used for now as the solution has several problems:

- Not efficient, the algorithm used to be trained with the large model, and this may slow down the feedback
- Not accurate, and this may lead to an increased risk of patients wrongly visiting and burdens the medical stress
- Not mapping precisely to the existing departments in a hospital, it usually maps the result to a very  general department and some hospital may not contains or have a different name

## 2⃣️ Algorithm Design

### 1) Model Structure

<div align=center>
	<img src="https://raw.githubusercontent.com/FionaChan01/Medi-BERT/main/images/Structure.png">
  </div>

### 2) Data Augmentation

- Adding Knowledge Graph
  - After obtaining the user consultation text, the text is firstly **divided** into words using the Chinese word separation library: jieba, and then each word in the text is **searched for all its triples** in the knowledge graph, and **one triple is randomly selected** and put into brackets, and **added** into the original word. Although simple, the table in the results section shows that such an embedding method can bring some improvement to the model.

  - Knowledge Embedding Algorithm

  	<div align=center>
  		<img src="https://raw.githubusercontent.com/FionaChan01/Medi-BERT/main/images/alg1.jpg" width="60%">
  		</div>

- Adding Adversarial Samples

	> It is difficult to distinguish the symptom descriptions in medical texts, and the similarity between texts is easy to generate, therefore,  identifying similar texts is the key to improving the accuracy of the model

	- Process of noise embedding

		[TODO]

		- The unsupervised learning of **TF-IDF features** is performed on the symptom description texts of each department in the training corpus, words with less than 500 occurrences are filtered out, and the **cosine similarity** of TF-IDF features of symptom description texts of two departments is calculated and normalized with respect to the **conversion probability**.
		- In the training reading samples, each sample can be replaced with the label of another department with a 20% probability based on the conversion probability and keep the original label unchanged with an 80% probability
		- This method takes into account the special characteristics of the medical text to introduce noise, which effectively **enhances the noise resistance and robustness** of the model, **prevents the model from overfitting**, and makes the accuracy rise further.

- Semantic Enrichment

	【TODO】

	- As the text described by the patients may be too vague, and concise as well as carry little information, the knowledge graph is needed to guide the patients in an appropriate way to enrich the semantics and inject the knowledge used in training to further improve the accuracy of the model.

### 3) Inference 

- Precisely Mapping

	[TODO]

	- After the model returns the result, the department corresponding to the hospital selected by the user and the department result returned by the model is queried in the database, and its name is returned to the user to ensure that the department exists in the hospital where the user is located.



## 3⃣️ Result



## 4⃣️ Resources

