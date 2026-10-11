# AI Resume Screening Assistant 📄🤖

An AI-powered resume screening application built using **LangChain, Google Gemini, Retrieval-Augmented Generation (RAG), FAISS, and Streamlit**. It helps recruiters and hiring teams evaluate resumes against a job description by identifying relevant skills, missing qualifications, candidate strengths, and areas for improvement.

## 🚀 Project Overview

Screening multiple resumes manually can be time-consuming. This project simplifies the initial screening process by allowing users to upload PDF resumes and compare candidates against a given job description.

The application uses a RAG pipeline to retrieve relevant information from each resume and generate structured, evidence-based evaluations using a Large Language Model (LLM).

## ✨ Key Features

- **PDF Resume Upload:** Upload and evaluate multiple resumes in one session.
- **Job Description Analysis:** Compare candidate qualifications against job requirements.
- **AI-Powered Match Score:** Generate an estimated resume-to-job match score from 0–100.
- **Skill Analysis:** Identify matching skills, partially matching skills, and missing or unverified skills.
- **Candidate Summary:** Generate a concise summary of each candidate's relevant experience and qualifications.
- **Strengths and Weaknesses:** Highlight relevant strengths and potential gaps against the job description.
- **Hiring Recommendation:** Generate a preliminary recommendation with supporting justification.
- **Evidence-Based Evaluation:** Retrieve relevant resume passages to support the evaluation.
- **Candidate Comparison:** Compare evaluation results across multiple candidates.
- **CSV Export:** Export screening results for further analysis.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| LangChain | LLM application framework |
| Google Gemini 2.5 Flash | Resume evaluation and analysis |
| Google Gemini Embeddings | Convert resume text into vector embeddings |
| FAISS | Vector storage and similarity search |
| PyPDFLoader | Extract text from PDF resumes |
| RecursiveCharacterTextSplitter | Split resume text into manageable chunks |
| Pydantic | Structured output validation |
| Pandas | Tabular data processing and CSV export |
| python-dotenv | Environment variable management |

## 🧠 How It Works

1. **Upload Resumes:** Users upload one or more PDF resumes.
2. **Extract Text:** PyPDFLoader extracts text from each resume.
3. **Split Documents:** The extracted text is divided into smaller chunks using RecursiveCharacterTextSplitter.
4. **Generate Embeddings:** Google Gemini embeddings convert text chunks into vector representations.
5. **Build the Vector Store:** FAISS stores the vectors and enables similarity-based retrieval.
6. **Retrieve Relevant Context:** The retriever finds resume passages relevant to the job description and evaluation criteria.
7. **Evaluate with Gemini:** The retrieved context and job description are passed to the LLM through a LangChain prompt template.
8. **Validate Results:** Pydantic validates the structured evaluation output.
9. **Display and Export:** Streamlit presents the scores, skills, candidate summaries, recommendations, and downloadable results.

## 🏗️ Project Structure

```text
AI_Resume_Screening_Assistant/
│
├── app.py                 # Streamlit application
├── resume_screening.ipynb # Jupyter notebook for experimentation
├── requirements.txt       # Python dependencies
├── .env                   # API key and model configuration
├── .gitignore             # Files excluded from Git
│
└── sample_resumes/        # Sample PDF resumes for testing
    ├── resume_A.pdf
    ├── resume_B.pdf
    └── resume_C.pdf
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11 or 3.12
- Conda or another Python environment manager
- A Google Gemini API key
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Resume-Screening-Assistant.git
cd AI-Resume-Screening-Assistant
```

Replace `YOUR_USERNAME` with your GitHub username and adjust the repository URL if you choose a different name.

### 2. Create a Conda environment

```bash
conda create -n resume_ai python=3.12 -y
conda activate resume_ai
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Obtain an API key from [Google AI Studio](https://aistudio.google.com/apikey).

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=models/gemini-embedding-001
```

Replace the placeholder with your actual API key. Never commit your API key to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

```text
http://localhost:8501
```

## 📊 Example Evaluation Output

For each uploaded resume, the application provides:

- Estimated job match score (0–100)
- Matching and partially matching skills
- Missing or unverified skills
- Candidate summary
- Strengths and potential weaknesses
- Preliminary hiring recommendation
- Relevant resume excerpts and supporting evidence

The application can also compare multiple candidates and export evaluation results as CSV files.

## 🔍 Example Use Case

**Job Description:** Junior Data Scientist

**Required Skills:** Python, SQL, Pandas, NumPy, Machine Learning, Data Visualization

The system retrieves relevant sections from a candidate's resume and evaluates how well the documented qualifications align with the job requirements.

The evaluation is based on available resume evidence. A skill not found in the retrieved content should not automatically be interpreted as a skill the candidate lacks.

## 🌐 Deployment

The application can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/).

General deployment steps:

1. Push the project to GitHub.
2. Connect the repository to Streamlit Community Cloud.
3. Select `app.py` as the application entry point.
4. Configure `GOOGLE_API_KEY` and any required model settings in the deployment's Secrets configuration.
5. Deploy and test the application.

Ensure that the API key, private resumes, and other confidential information are not committed to the repository.

## ⚠️ Limitations and Responsible Use

- Evaluation quality depends on resume text extraction, retrieval relevance, and LLM output.
- Scoring is an approximate measure of alignment with a particular job description, not a validated prediction of job performance.
- Missing evidence does not necessarily mean a candidate lacks a qualification.
- The model may occasionally produce inaccurate or incomplete evaluations.
- The system should support, not replace, human review and hiring decisions.
- Candidate evaluations should focus on job-related qualifications and avoid protected or sensitive personal attributes.
- Obtain appropriate permission before processing candidate resumes and handle personal data securely.

## 🔮 Future Improvements

- OCR support for scanned resumes.
- Improved skill extraction and experience matching.
- Configurable evaluation criteria and scoring weights.
- Support for additional document formats.
- Persistent candidate history and evaluation reports.
- Enhanced evaluation validation and model error handling.
- Improved multilingual resume support.

## 👨‍💻 Learning Outcomes

This project demonstrates practical implementation of:

- Large Language Model (LLM) applications
- Retrieval-Augmented Generation (RAG)
- LangChain document processing and retrieval
- Text embeddings and vector databases
- Prompt engineering and structured output parsing
- PDF information extraction
- Streamlit application development
- AI-assisted candidate evaluation

## 📄 License

This project is intended for educational and demonstration purposes. Add a license file, such as the MIT License, if you want to permit others to use, modify, and distribute the code under its terms.

---

**If you find this project useful, consider giving the repository a ⭐ on GitHub!**
