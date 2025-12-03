Overview:
The backend of MindScan is the core engine where all the AI processing takes place. It receives MRI images from the frontend, preprocesses them, runs deep learning inference, and returns detailed analytical results such as tumor detection, segmentation maps, and confidence scores.

This backend transforms raw medical images into interpretable visual insights — built for accuracy, reliability, and research-grade extensibility.

Core Responsibilities:

Receive Input: Accepts MRI scans uploaded via the frontend interface.

Preprocessing:

Converts input images into the correct format (grayscale or 3D volume).

Normalizes intensity values and removes noise.

Applies optional enhancement techniques such as CLAHE or histogram equalization.

Model Inference:

Passes the processed image through a trained deep learning model (e.g., CNN, UNet).

Detects or segments abnormal regions such as brain tumors.

Generates classification labels or heatmaps based on model predictions.

Postprocessing & Response:

Converts output predictions into an interpretable form (overlay mask, confidence percentage).

Sends structured JSON data and/or processed image back to the frontend for display.

Architecture:

Frontend → Backend: Receives image files via REST API (POST /analyze).

Backend → Model: Loads the trained model once for inference efficiency.

Backend → Frontend: Returns prediction results and generated images for visualization.