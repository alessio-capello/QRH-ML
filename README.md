# AI/ML Operations QRH

A tactical quick-reference handbook for machine learning and AI engineering, designed for fast triage, model selection, and production troubleshooting.

## What this project is

This repository is not a textbook. It is a compact operational guide inspired by aviation QRHs: short, structured, decision-oriented, and built for high-pressure situations.

It helps you:
- choose the right model family quickly
- apply GO / NO-GO rules before building or deploying
- find the likely failure mode when training or inference breaks
- navigate from a high-level decision tree to the exact model card and troubleshooting flow

## Project structure

- [en](en): English version of the handbook
- [it](it): Italian version of the handbook
- [pictures](pictures): shared diagrams and visual assets
- [README.md](README.md): project overview and entry point

## Start here

Choose the language you want to use:

- [English entry point](en/0%20-%20hot%20to%20use.md)
- [Italian entry point](it/0%20-%20hot%20to%20use.md)

## Handbook map

### English
- [0. How to use this QRH](en/0%20-%20hot%20to%20use.md)
- [2.1 Baseline ML & Linear Methods](en/2.1%20-%20baseline_lineari.md)
- [2.2 Ensemble Methods](en/2.2%20-%20ensemble.md)
- [2.3 Unsupervised & Representation](en/2.3%20-%20unsupervised.md)
- [2.4 Deep Learning Fundamentals](en/2.4%20-%20deep%20learning.md)
- [2.5 Time Series & Forecasting](en/2.5%20-%20time%20series.md)
- [2.6 Computer Vision & Spatial Data](en/2.6%20-%20computer%20vision.md)
- [2.7 Natural Language Processing](en/2.7%20-%20NLP.md)
- [2.8 Reinforcement Learning](en/2.8%20-%20reinforcement%20learning.md)
- [3. Troubleshooting Checklists](en/3%20-%20troubleshooting.md)

### Italiano
- [0. Come usare questo QRH](it/0%20-%20hot%20to%20use.md)
- [2.1 Baseline ML & Metodi Lineari](it/2.1%20-%20baseline_lineari.md)
- [2.2 Ensemble Methods](it/2.2%20-%20ensemble.md)
- [2.3 Unsupervised & Rappresentazione](it/2.3%20-%20unsupervised.md)
- [2.4 Deep Learning Fundamentals](it/2.4%20-%20deep%20learning.md)
- [2.5 Time Series & Forecasting](it/2.5%20-%20time%20series.md)
- [2.6 Computer Vision & Spatial Data](it/2.6%20-%20computer%20vision.md)
- [2.7 Natural Language Processing](it/2.7%20-%20NLP.md)
- [2.8 Reinforcement Learning](it/2.8%20-%20reinforcement%20learning.md)
- [3. Checklist di Troubleshooting](it/3%20-%20troubleshooting.md)

## Typical usage

1. Open the relevant language-specific entry file.
2. Use the decision matrix to identify the task family.
3. Follow the model card that matches the problem.
4. Validate the GO / NO-GO constraints and then move to implementation or troubleshooting.

## Scope

The handbook focuses on practical model selection and operational failure diagnosis across:
- tabular data
- computer vision
- time series
- NLP
- reinforcement learning
- production and deployment constraints

It is intended as a practical companion for practitioners, ML engineers, data scientists, and technical teams making rapid model decisions under real operating constraints.