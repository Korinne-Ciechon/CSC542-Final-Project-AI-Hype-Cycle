# CSC542-Final-Project-AI-Hype-Cycle

## Project Overview
This project investigates whether public discourse surrounding emerging AI technologies follows the predictable phases of the **Gartner Hype Cycle**. We analyzed YouTube discourse patterns for five major technologies: **ChatGPT, GPT-4, Gemini, GitHub Copilot, and Sora.**

By combining YouTube metadata with **Germini 2.5 Flash Lite** for feature extraction, we identified linguistic "signatures" (sentiment, hyperbole, and technical complexity) that characterize different stages of technology maturity. Our results empirically validate the Hype Cycle with high statistical significance (p<0.001).

## Key Findings
- **Predictive Accuracy:** Our Linear SVM model achieved a 92.3% test accuracy in classifying technology maturity phases based soley on discourse features.
- **Temporal Validation:** One-Way ANOVA and Tukey HSD test confirmed that our sunsupervised clusters are significantly distinct temporal events (p = 6.26 x 10^-6).
- **Feature Importance:** Sentiment, hyperbole, and technical complexity were identified as the most discriminative predictors of a technology's position on the hype curve.

## Repository Structure
- /docs:
 -  CSC_542_Final_Report.pdf: The complete academic paper.
 - Presentation_slides.pdf: Visual summary of the project.
- /data:
  - youtube_hype_all_aggregated.csv: The primary aggregated dataset used for R analysis.
  - raw/: Sub-folder containing the 1,264 individual video records and original API pulls.
- /scr:
  - analysis/Hype_Cycle_Analysis.R: The final R script used for statistical testing, PCA visualization, and training the 92.3% accurate SVM model.
  - data_scraping/: Sub-folder of the python script for YouTube API data acquisition and python script for LLM feature extraction and window-based aggregation.
 
## How to Run
1. **Data Processing:** The Python notebooks in /scr/colab are designed for Google Colab. They require a YouTube Data API v3 key and a Google AI Studio (Gemini) API key.
2. **Statistical Analysis:** The R script requires the tidyverse, cluster, MASS, class, tree, randomForest, corrplot, gridExtra, ggpubr libraries.

## Authors
- Yoomee Shin
- Nikhita Guhan
- Korinne Ciechon
