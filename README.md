# Evolutionary Trajectories in Geometric Modeling: Unveiling Patent Trends via Latent Dirichlet Allocation

**Abstract**  
Geometric modeling serves as a cornerstone of industrial software, facilitating reduced development costs, accelerated market entry, and enhanced functionality in manufacturing. This study leverages Latent Dirichlet Allocation (LDA) to analyze the evolution of geometric modeling techniques from 1976 to 2024, as evidenced in patent literature. Through LDA, we identify eight pivotal technological topics clustered into five overarching themes: surface modeling, parametric modeling, intelligent CAD, structural modeling, and solid modeling. Our findings reveal a progression from traditional solid modeling to advanced forms such as parametric and surface modeling, culminating in sophisticated structural modeling and intelligent CAD. Parametric modeling emerged as a dominant technology in the 1990s, followed by surface modeling in the 2000s, and recent advancements in intelligent CAD post-2010. This study provides insights into the technological landscape of geometric modeling, offering valuable guidance for policymakers and industry stakeholders in optimizing patent strategies and driving innovation.

---

## 📖 Citation 

If you use this code, please cite the following article: Published in ***The Visual Computer***.

---

## 🚀 Key Features
- **Memory Efficiency**: Gensim's streaming I/O for large corpora
- **Adaptive Priors**: Asymmetric Dirichlet distributions (`α`, `β` in `auto` mode)
- **Multi-Metric Validation**: 
  - Perplexity analysis
  - UMass coherence scores
  - PyLDAvis diagnostics
- **Optimal Granularity**: Balance between complexity and interpretability

---

## ⚙️ Core Hyperparameters
| Parameter          | Configuration                  |
|--------------------|--------------------------------|
| Training Iterations| 50 cycles (ensured convergence)|
| α (Document-Topic) | Asymmetric Dirichlet (`auto`)  |
| β (Topic-Word)     | Asymmetric Dirichlet (`auto`)  |
| Learning Method    | Online Variational Bayes       |

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

## 📂 Data Availability
**Original Dataset**  
> "The dataset is not hosted publicly due to GitHub's file size restrictions. Researchers may request access for reproducibility purposes by contacting the corresponding author."




