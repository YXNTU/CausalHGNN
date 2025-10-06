# 🧩 Are Heterogeneous Graph Neural Networks Truly Effective?  
### A Causal Perspective

> 📄 *Official implementation of our paper*  
> **"Are Heterogeneous Graph Neural Networks Truly Effective? A Causal Perspective"**  
> *(under review at **Knowledge-Based Systems (KBS)**).*

---

### 🧠 Abstract (shortened)

Heterogeneous Graph Neural Networks (HGNNs) extend traditional GNNs by integrating multiple relation types and semantic information.  
However, whether HGNNs are **intrinsically effective** remains unclear, as most studies assume rather than establish their causal validity.  

This work disentangles the effects of **model architecture** and **heterogeneous information** through large-scale reproduction across 21 datasets and 20 baselines.  
We introduce a **causal effect estimation framework** that evaluates candidate factors under factual and counterfactual analyses,  
with robustness validated via minimal sufficient adjustment sets, cross-method consistency, and sensitivity analyses.  

Our findings reveal that architectural complexity has **no causal impact** on performance,  
while heterogeneous information improves accuracy by enhancing **homophily** and **local–global distribution discrepancy**,  
making node classes more distinguishable.  

---

### 🌟 Highlights
1. 📊 A comprehensive **benchmark for HGNNs** with 21 datasets and 20 baselines  
2. 🧱 **Model architecture** and complexity show *no causal effect* on performance  
3. 🔗 **Homophily** and **distribution discrepancy** are the *key causal factors* underlying HGNN effectiveness  

---

📦 **Repository Structure**
