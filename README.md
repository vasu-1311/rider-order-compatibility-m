# Rider–Order Compatibility ML System

## Overview
This project implements an end-to-end machine learning–based backend system designed to support delivery operations.  
The system evaluates multiple available riders for a given order, calculates rider fatigue using workload indicators, predicts assignment risk using a trained ML model, and recommends the safest rider for assignment.

The goal of the project is **decision support**, not automated assignment.

---

## Problem Statement
In delivery platforms, assigning an order to an overloaded rider often leads to delays, poor customer experience, and rider burnout.  
Simple rule-based assignment systems fail to capture the combined effect of workload, distance, and operational conditions.

This project addresses that gap by using machine learning to assess assignment risk and recommend safer rider–order matches.

---

## Solution Approach
The system follows a structured ML pipeline:

1. **Data Analysis & Feature Engineering**
   - Rider workload features such as working hours, completed orders, and past delays
   - A composite *fatigue score* engineered to represent rider strain

2. **Modeling**
   - Logistic Regression model trained to predict assignment risk
   - Focus on explainability and operational interpretability

3. **Backend Decision Logic**
   - Multiple riders evaluated for the same order
   - Rider-specific fatigue calculated
   - Risk probability predicted for each rider
   - Riders ranked by risk
   - Safest rider recommended

4. **Deployment**
   - Trained model serialized and reused
   - Backend logic exposed via a Streamlit-based application

---

## Fatigue Score Logic
Rider fatigue is computed using a weighted combination of workload indicators:

