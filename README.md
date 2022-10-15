# Medi-BERT

This repository contains the code for our [*National Innovation Project*](http://gjcxcy.bjtu.edu.cn/NewLXItemListForStudentDetail.aspx?ItemNo=785167), which focuses on the problem of how to accurately triage the patients in the hospital to a specific department; the Project was rated as Good.

## 1⃣️ Background

> When first entering a hospital, patients are not sure which department they should visit, so hospital guidance station is prone to congestion during peak periods, especially in the context of the current epidemic, which may cause a series of problems such as the transmission of diseases, bringing many inconveniences to patients and hospitals. 

The frequently model used for now as the solution has several problems:

- Not efficient, the algorithm used to be trained with the large model, and this may slow down the feedback
- Not accurate, and this may lead to an increased risk of patients wrongly visiting and burdens the medical stress
- Not mapping precisely to the existing departments in a hospital, it usually maps the result to a very  general department and some hospital may not contains or have a different name

## 2⃣️ Algorithm Design

**Structure**

<div align=center>
	<img src="https://raw.githubusercontent.com/FionaChan01/Medi-BERT/main/images/Structure.png">
  </div>

### 1) Data Augmentation

- Knowledge Graph
	- After obtaining the user consultation text, the text is firstly **divided** into words using the Chinese word separation library: jieba, and then each word in the text is **searched for all its triples** in the knowledge graph, and **one triple is randomly selected** and put into brackets, and **added** into the original word. Although simple, the table in the results section shows that such an embedding method can bring some improvement to the model.
	- Knowledge Embedding Algorithm

	<div align=center>
		<img src="https://raw.githubusercontent.com/FionaChan01/Medi-BERT/main/images/alg1.jpg">
  	</div>

- Adding Adversarial Samples

### 2) Model Training



## 3⃣️ Result

