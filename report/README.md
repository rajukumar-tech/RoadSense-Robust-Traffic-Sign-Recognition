# RoadSense: Group 4 Final Technical Report

This directory contains the final deliverables for the Capstone Project per Section 8 of the project brief:

1. **Official PDF Submission:** [`RoadSense_Group4_Final_Report.pdf`](RoadSense_Group4_Final_Report.pdf) (12-page comprehensive report, PDF ready for grading).
2. **Online Markdown Version:** [`FINAL_REPORT.md`](FINAL_REPORT.md) (fully rendered with embedded high-resolution figures for reading directly on GitHub).
3. **Interactive Demo Interface:** [`../app.py`](../app.py) (Gradio perception and Grad-CAM explainability app).

### Report Outline (Compliant with Section 8):
1. **Title Page & Contributions:** Author metadata, leader designation, repository link, ownership table.
2. **Abstract:** 190 words summarizing motivations, track-split methodology, 15-condition benchmark, and Grad-CAM findings.
3. **Introduction and Motivation:** Real-world driving sensor corruptions vs. clean benchmark limitations.
4. **Related Work:** 5 key foundational papers (Stallkamp 2012, Sermanet 2011, Hendrycks 2019, He 2016, Selvaraju 2017).
5. **Dataset, EDA & Splitting:** 43 classes, 10.7× imbalance, atomic track-level disjoint split without leakage.
6. **Methodology:** Scratch CNN vs. transfer models, label-safe augmentations, two-stage fine-tuning.
7. **Experiments and Benchmark Results:** Clean evaluation, training curves, clean vs. mCA comparison, 15-condition robustness table, degradation profiles.
8. **Error Analysis and Explainability:** Top-confused classes matrix with real numbers, native Grad-CAM attention shift, Indian road sign cross-jurisdiction check.
9. **Limitations, Ethics and Future Work:** Resolution constraints, confidence calibration, safety considerations.
10. **Conclusion:** Core takeaway that clean accuracy hides large robustness gaps.
11. **Contribution Table, AI Declaration & References:** Detailed ownership, viva focus, ethical AI disclosure, IEEE citations.
