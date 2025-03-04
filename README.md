# AI Research Idea Finder

A minimalistic Streamlit web application that helps researchers discover new AI research paper ideas based on the latest publications from arXiv.

## Features

- Search for the latest papers by topic, conference, or both
- Specify the number of papers to retrieve (20-500)
- Generate novel research ideas using Claude AI
- Download generated research ideas as text files
- No sign-in required - just provide your Claude API key for the session

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run main.py
   ```

## Usage

1. Open the web app in your browser
2. Enter your Claude API key in the sidebar (this is only stored for the current session)
3. Provide a nickname for your session
4. Set your search parameters:
   - Choose to search by topic, conference, or both
   - Enter your search terms
   - Set the number of papers to retrieve
   - Set the number of research ideas to generate
5. Click "Search Papers" to retrieve relevant papers
6. Click "Generate Research Ideas" to get AI-generated research ideas based on the retrieved papers
7. Download the generated ideas as a text file if desired

## Requirements

- Python 3.7+
- Streamlit
- arXiv API
- Anthropic Claude API key

## Note

This application requires a valid Claude API key to generate research ideas. Your API key is only used during the current session and is not stored anywhere.

## License

MIT