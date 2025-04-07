![image](https://github.com/user-attachments/assets/b40f9669-0ba4-4b68-b318-42e18974f6a6)# HireIQ: Multi-Agent Job Screening System with Adaptive Intelligence

HireIQ revolutionizes recruitment through an on-premise AI system that automates the entire hiring pipeline. The solution deploys specialized agents for JD analysis, CV processing, candidate matching, and interview scheduling—all powered by local LLMs for data security. HireIQ reduces screening time by 80-90% while ensuring consistent, bias-free candidate evaluation across any recruitment scale.

## Key Features

- **On-Prem LLM Integration**: Uses Ollama to run Gemma3:4b locally for privacy and control
- **Custom Multi-Agent Framework**: Coordinates specialized agents for different parts of the screening process
- **Vector Embeddings**: Uses Ollama-based embeddings for semantic search and matching
- **SQLite Database**: Stores job descriptions, candidate data, and matching results
- **Optimized Performance**: Parallel processing, caching, and efficient matching algorithms

## Agents

The system uses a multi-agent architecture where specialized agents work together:

1. **CV Processor Agent**: Extracts data from CVs including skills, experience, qualifications
2. **JD Summarizer Agent**: Processes job descriptions to extract requirements
3. **Efficient Matcher**: Compares jobs and candidates using embeddings and LLM analysis
4. **Scheduler Agent**: Handles interview scheduling for shortlisted candidates

## Setting Up

### Prerequisites

- Python 3.8+ 
- [Ollama](https://ollama.ai/) installed on your system
- Basic dependencies (see requirements.txt)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install Ollama and make sure it's running (follow instructions at [ollama.ai](https://ollama.ai/))
4. Pull necessary models (optional, will be done automatically when needed):
   ```
   ollama pull gemma3:4b
   ollama pull nomic-embed-text
   ```

## Usage

### Running the System

The system runs the full job screening pipeline using Gemma3:4b model:

#### Windows:
```
run_pipeline.bat
```

#### Unix/Mac:
```
./run_pipeline.sh
```

Alternatively, you can run with Ollama checking:

#### Windows:
```
check_ollama_and_run.bat
```

#### Unix/Mac:
```
./check_ollama_and_run.sh
```

## **[Dataset](https://drive.google.com/drive/folders/12xWcKoy15ph9XH_VfVCXU9Yd7SByLb2-?usp=drive_link)**

### What the Pipeline Does

1. **Reset Database**: Clears previous data and prepares for a new run
2. **Process Job Descriptions**: Extracts requirements from job postings
3. **Process CVs**: Extracts skills and qualifications from candidate resumes
4. **Match Candidates**: Uses AI to match candidates to jobs
5. **Shortlist Candidates**: Selects the best candidates for each job
6. **Schedule Interviews**: Creates interview schedules for shortlisted candidates

## Architecture

### BaseAgent and Multi-Agent Framework

The system uses a custom multi-agent framework with a BaseAgent class that provides:

- Ollama LLM integration
- Tool registration and usage
- Agent messaging
- State management

### Ollama Integration

The system uses Ollama for:

- **LLM Processing**: Using Gemma3:4b for analyzing text, summarization, and generating insights
- **Embeddings**: For semantic search and matching
- **Skills Extraction**: Identifying skills and requirements from text

## Database Schema

The system uses SQLite with the following key tables:

- `job_descriptions`: Stores job postings
- `candidates`: Stores candidate information from CVs
- `match_results`: Stores matching scores and results
- `shortlists`: Tracks shortlisted candidates
- `interviews`: Manages interview schedules

## Project Structure

```
├── optimized_app.py    # Main application
├── config.py           # Configuration settings
├── efficient_matcher.py# Optimized matching system
├── reset_db.py         # Database reset utility
├── run_pipeline.bat    # Windows run script
├── run_pipeline.sh     # Unix run script
├── check_ollama_and_run.bat  # Windows Ollama check script
├── check_ollama_and_run.sh   # Unix Ollama check script
├── agents/             # Agent modules
│   ├── cv_processor.py # CV processing agent
│   ├── jd_summarizer.py # Job description summarizer
│   └── scheduler_agent.py # Interview scheduler
├── database/           # Database modules
├── Dataset/            # Dataset directory for job descriptions and CVs
├── generated_emails/   # Generated interview emails
├── utils/              # Utility modules
│   ├── ollama_client.py # Ollama LLM client
│   ├── embeddings.py   # Embedding utilities
│   └── pdf_extractor.py # PDF data extraction
└── requirements.txt    # Dependencies
```

## License

MIT License 
