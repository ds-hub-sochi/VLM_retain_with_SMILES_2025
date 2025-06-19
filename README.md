# VLM Retain: Preserving Pre-trained Capabilities in Vision-Language Models

## Overview
This repository provides a framework to explore techniques for fine-tuning Vision-Language Models (VLMs) while preserving their pre-existing capabilities. We investigate strategies to mitigate the phenomenon of catastrophic forgetting—where models lose or overwrite previously acquired knowledge—ensuring that adaptation to new tasks does not degrade fundamental skills.

## Problem Statement
Historical document recognition plays a crucial role in helping researchers understand our past, yet it remains a challenging task. Documents from different eras—even those authored by the same individual—can vary significantly in vocabulary, grammar, and style. Moreover, the limited availability of annotated data across many languages further complicates this endeavor.

At the same time, large pretrained models (LLM/VLM) offer extensive general capabilities but are susceptible to catastrophic forgetting when fine-tuned on new tasks. This often results in the loss of valuable skills such as logical reasoning and effective response generation.

Our project aims to develop training methodologies that enable VLMs to learn new tasks, such as historical document recognition, while preserving their original strengths. This approach provides a cost-effective and efficient solution by eliminating the need for custom models for each language.

## Project Objectives and Data
During the project, participants will pursue the following objectives:
1. **Analyze Current Fine-tuning Approaches:**
   - Conduct a comprehensive review of existing methodologies for fine-tuning Vision-Language Models (VLMs), emphasizing techniques that retain previously acquired knowledge and mitigate catastrophic forgetting.
2. **Examine Fine-tuning Frameworks:**
   - Evaluate state-of-the-art frameworks for efficient and user-friendly VLM fine-tuning, such as llama-factory, to understand their capabilities and limitations.
3. **Select and Fine-tune a Suitable Model:**
   - Identify an appropriate VLM and fine-tune it for the task of historical document recognition, ensuring that new skills are acquired without compromising existing competencies.
4. **Evaluate Model Degradation:**
   - Assess the impact of fine-tuning on model performance by utilizing established benchmarks for VLM evaluation, and conduct a detailed analysis of various training strategies.
5. **Develop a Multilingual VLM:**
   - Train a multilingual VLM capable of recognizing historical documents in multiple languages, ensuring that its performance remains at or above XX% of the original model's quality.

Participants will leverage available datasets of historical documents and apply rigorous quantitative and qualitative evaluation metrics to measure both knowledge retention and task-specific performance.

## Data Disclaimer
The interview data is sensitive and provided only to team members through secure channels. The data remains confidential and must not be distributed publicly. All source data should be deleted after project completion.

### Datasets
- **Digital Peter**: A digital collection of historical handwritten texts curated to support handwriting recognition research. This dataset comprises approximately 300 annotated pages at the transcription level, featuring manuscripts primarily from the late 19th century (circa 1870–1900) that highlight the transition from traditional calligraphy to early modern handwriting styles. More information can be found [here](https://github.com/MarkPotanin/DigitalPeter).

- **IAM Handwriting Database (IAM Lines)**: A widely-used dataset for handwritten text recognition, containing data from 657 writers, 1,539 document pages, and 13,353 annotated text lines. The documents, predominantly written during the late 20th century (from the 1970s to the 1990s), capture a diverse range of handwriting styles and serve as a benchmark for modern HTR systems. Check details [here](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

- **LAM**: The Ludovico Antonio Muratori (LAM) dataset is the largest line-level HTR dataset available, comprising over 25,823 annotated lines extracted from Italian ancient manuscripts. These manuscripts span more than 60 years, approximately from 1850 to 1910, and showcase the evolution of handwriting styles and orthographic conventions over time. More details are available [here](https://github.com/aimagelab/LAM).

- **VMLHD**: The Historical Arabic Documents dataset is designed to support recognition systems for handwritten Arabic texts. It features 668 fully annotated pages with 159,149 subword segments and 326,289 characters (with a vocabulary of 5,509 unique forms) annotated at the subword level. The manuscripts in this dataset were created between 1088 and 1451, reflecting a broad historical range and significant evolution in script and stylistic features. Learn more [here](https://majeek.github.io/tutorials/vmlHD/).

## Evaluation of Outcome
Successful outcomes will be measured by:
- Qualitative improvement in feedback.
- Quantitative increase in favorability ratings among skeptical agents.
- Demonstrated enhancement of communication sentiment post-iteration.

## About
This project was developed as part of a hackathon to explore innovative methods using LLM-based synthetic focus groups for central bank communication evaluation. It serves as a blueprint for future research and practical applications in economic communication strategy.

## Resources
- Relevant literature on focus groups and synthetic agent simulation.
- Hackathon documentation and submission guidelines.

## License
This project is licensed under the [MIT License](LICENSE.txt).

---

*Repository maintained by evstalina.* 