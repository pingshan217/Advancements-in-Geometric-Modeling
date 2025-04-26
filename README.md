# Advancements in Geometric Modeling: An Exploratory Analysis of Patent Trends Using Latent Dirichlet Allocation

**Abstract**  
Geometric modeling stands as a cornerstone of industrial software, enabling reduced development and production costs, accelerated market entry, and enhanced functionality and aesthetics in the manufacturing sector.  This study leverages Latent Dirichlet Allocation (LDA) topic modeling to analyze the evolution of geometric modeling techniques from 1976 to 2024, as evident in patent literature.  Eight pivotal technological topics are identified and grouped into five overarching themes: surface modeling, parametric modeling, intelligent CAD, structural modeling, and solid modeling.  Our findings reveal a progression from traditional solid modeling to advanced forms such as parametric and surface modeling, culminating in sophisticated structural modeling and intelligent CAD.  Here, we show that parametric modeling emerged as a dominant technology in the 1990s, followed by surface modeling in the 2000s, and recent advancements in intelligent CAD post-2010.  This study provides insights into the technological landscape of geometric modeling, aiding policymakers and industry stakeholders in optimizing patent strategies and driving innovation.

---

## 📂 Data Availability
**Original Dataset**  
> "The dataset is not hosted publicly due to GitHub's file size restrictions. Researchers may request access for reproducibility purposes by contacting the corresponding author."

---

## 📊 Code Overview

### 1. Perplexity Analysis
**Script**: [Perplexity for numbers of topics.py](https://github.com/pingshan217/Advancements-in-Geometric-Modeling/blob/230683a5c1e9830ee1ff2683caef388fe6600f7d/Perplexity%20for%20numbers%20of%20topics.py)  

### 2. Coherence Score Optimization 
**Script**: [Topic coherence score.py](https://github.com/pingshan217/Advancements-in-Geometric-Modeling/blob/230683a5c1e9830ee1ff2683caef388fe6600f7d/Topic%20coherence%20score.py)  

### 3. Topic Visualization & Diagnostics
**Script**: [Word clouds of 8 Topics.py](https://github.com/pingshan217/Advancements-in-Geometric-Modeling/blob/230683a5c1e9830ee1ff2683caef388fe6600f7d/Word%20clouds%20of%208%20Topics.py)  

### Implementation Components
#### Text Preprocessing
- Custom stopword filtering  
- Term frequency analysis  
- Data cleaning pipeline

#### LDA Modeling
- Gensim-based implementation  
- Batch learning with 50 iterations  
- Automated hyperparameter tuning

---

## 📈 Results

### Topic Selection Methodology
The optimal topic count was determined through cross-validation using:
- Perplexity metrics (model generalization)  
- Coherence scores (semantic validity)  
- Visual diagnostics (topic distinctness)

### Final Output
**Selected Configuration**: K=8 topics  
**Validation Documentation**:  
- Topic separation analysis reports
- Lexical specificity metrics

---

