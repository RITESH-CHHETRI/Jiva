# Try-tech-Nexus-24

# JIVA: RAG Trained Language Model for Hospital Purposes

## Overview
JIVA is a specially trained Language Model (LLM) designed to assist hospitals in managing patient-doctor interactions efficiently. With its advanced capabilities, JIVA serves as a mediator between doctors and patients, streamlining the communication process and enhancing patient care.

## Features

### 1. Storage of Doctor Details
JIVA maintains a comprehensive database of all doctors associated with the hospital. This includes their specialties, contact information, and availability, ensuring quick access and efficient doctor-patient matching.

### 3. Doctor Allocation
Upon analyzing the patient's health information, JIVA automatically assigns the most suitable doctor to the patient. This ensures timely and appropriate medical attention, optimizing patient care.

### 4. Abstract Generation
JIVA generates abstracts summarizing patients' details, symptoms, and other relevant information. These abstracts are provided to the assigned doctor, helping them quickly grasp the patient's condition and make informed decisions.

## Objective

The objective of JIVA is to enhance patient-doctor communication, streamline healthcare workflows, and improve overall efficiency and quality of care in hospitals. By leveraging advanced AI technology, JIVA aims to revolutionize the way healthcare providers interact with patients, ultimately leading to better health outcomes and patient satisfaction.

## Target Audience

- Hospital Administrators
- Healthcare Providers (Doctors, Nurses, etc.)
- Patients

## Background

In today's healthcare landscape, effective communication between patients and doctors is paramount for delivering high-quality care. However, traditional methods of patient-doctor interaction often suffer from inefficiencies, leading to delays in diagnosis, treatment, and patient dissatisfaction. There is a pressing need for a solution that streamlines this communication process, ensuring timely access to medical expertise and personalized care.

## Problem

Hospitals often face challenges in managing patient-doctor interactions efficiently. These challenges include:

1. *Information Overload*: Doctors are inundated with vast amounts of patient data, making it difficult to prioritize and address each case promptly.

2. *Communication Barriers*: Patients may struggle to convey their health issues accurately, leading to misunderstandings and misdiagnoses.

3. *Resource Allocation*: Hospitals need a system to allocate doctors effectively, matching patient needs with the expertise and availability of medical professionals.

4. *Documentation Burden*: Healthcare providers spend significant time documenting patient details, detracting from direct patient care.

## Solution

JIVA is a sophisticated Language Model designed to address these challenges by serving as a mediator between patients and doctors. It offers the following key features:

- *Centralized Doctor Database*: Stores comprehensive details of all doctors associated with the hospital, facilitating quick and efficient doctor allocation.

- *Automated Doctor Allocation*: Matches patients with the most suitable doctor based on their medical needs, ensuring timely access to appropriate care.

- *Abstract Generation*: Summarizes patient details and symptoms for doctors, streamlining the diagnosis and treatment process.

## Installation

1. **Clone this repository:**
   ```sh
   git clone https://github.com/RITESH-CHHETRI/JIVA
   ```
2. **Choose the LLM**  
   -`main.py` uses OpenAI (Requires OpenAI api key)  
   -`repl.py` uses llama-13b (Requires Replicate and Pinecone API keys)


2. **Install required Python packages:**
   ```sh
   pip install -U -r mainrequirements.txt
   ```
   Or
      ```sh
   pip install -U -r replrequirements.txt
   ```

3. **Set up environment variables for OpenAI or Pinecone and Replicate**
   - Set up either in code or a `.env` file

4. **Run the program**
   ```py
   py main.py
   ```
   Or
   ```py
   py repl.py
   ```

## Contributors  

|            |                          |
| -------------- | ------------------------------------- |
| Ritesh Chhetri       | [@RITESH-CHHETRI](https://github.com/RITESH-CHHETRI) |
| Jeeva A Johney     | [@JeevaAJohney](https://github.com/JeevaAJohney) |
| Fathimathu Swafa   | [@Fathimathu-swafa](https://github.com/Fathimathu-swafa) |
| Soumya Annie Thomas   | [@S-A-T-07](https://github.com/S-A-T-07) |
|            |                          |
