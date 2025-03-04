import streamlit as st
import requests
import json
import anthropic
import arxiv
import os
import tempfile
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI Research Idea Finder",
    page_icon="🔍",
    layout="wide"
)

# CSS for a clean, minimalist UI
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
    }
    .reportview-container {
        margin-top: -2em;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3 {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for API key and configuration
with st.sidebar:
    st.title("Settings")
    claude_api_key = st.text_input("Claude API Key", type="password")
    user_nickname = st.text_input("Your Nickname (for session)", value="Researcher")
    
    st.subheader("Search Parameters")
    search_type = st.radio("Search by:", ["Topic", "Conference", "Both"])
    
    topic = None
    conference = None
    if search_type == "Topic" or search_type == "Both":
        topic = st.text_input("Research Topic")
    if search_type == "Conference" or search_type == "Both":
        conference = st.text_input("Conference Name (e.g., CVPR, ICLR, NeurIPS)")
    
    max_results = st.slider("Number of papers to retrieve", 
                           min_value=20, max_value=500, value=100, step=10)
    
    ideas_to_generate = st.slider("Number of research ideas to generate", 
                                min_value=3, max_value=15, value=5, step=1)

# Main content
st.title("AI Research Idea Finder 🔍")
st.markdown("### Get inspiring research paper ideas based on latest publications")

# Function to search arxiv for papers
def search_arxiv(topic=None, conference=None, max_results=100):
    query = ""
    
    if topic and conference:
        query = f"{topic} AND {conference}"
    elif topic:
        query = topic
    elif conference:
        query = conference
    else:
        return []
    
    # Sort by submission date, newest first
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    for result in search.results():
        paper = {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "categories": result.categories
        }
        papers.append(paper)
    
    return papers

# Function to generate research ideas using Claude
def generate_ideas_with_claude(papers, api_key, nickname, num_ideas=5):
    if not api_key:
        return "Please provide a Claude API key to generate ideas."
    
    # Create context from papers
    paper_summaries = []
    for i, paper in enumerate(papers[:50]):  # Limit to 50 papers to fit in context window
        summary = f"Paper {i+1}:\nTitle: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\n"
        summary += f"Published: {paper['published']}\nSummary: {paper['summary']}\n\n"
        paper_summaries.append(summary)
    
    context = "\n".join(paper_summaries)
    
    # Create prompt for Claude
    prompt = f"""You are a research advisor for {nickname}. Based on the following recent research papers, generate {num_ideas} novel and promising research ideas.

For each idea:
1. Provide a clear title
2. Explain the research gap or opportunity
3. Suggest a high-level methodology approach
4. Explain why this research would be impactful
5. List 2-3 key papers from the provided list that would serve as a foundation

Recent papers:
{context}

Generate {num_ideas} research ideas that build upon these papers but explore new directions or combine approaches in novel ways.
"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            temperature=0.7,
            system="You are a helpful AI research advisor tasked with generating novel research ideas based on recent publications.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content
    except Exception as e:
        return f"Error when calling Claude API: {str(e)}"

# Application flow
if st.button("Search Papers"):
    if not (topic or conference):
        st.error("Please provide at least a topic or conference name to search.")
    else:
        with st.spinner("Searching for papers..."):
            papers = search_arxiv(topic, conference, max_results)
            
            if not papers:
                st.warning("No papers found matching your criteria. Try adjusting your search terms.")
            else:
                st.session_state.papers = papers
                st.success(f"Found {len(papers)} papers on the topic.")
                
                # Display some sample papers
                st.subheader("Sample of Retrieved Papers")
                for i, paper in enumerate(papers[:5]):  # Show first 5 papers
                    with st.expander(f"{paper['title']}"):
                        st.write(f"**Authors:** {', '.join(paper['authors'])}")
                        st.write(f"**Published:** {paper['published']}")
                        st.write(f"**Summary:** {paper['summary']}")
                        st.write(f"[View on arXiv]({paper['url']})")

if st.button("Generate Research Ideas") and st.session_state.get("papers"):
    if not claude_api_key:
        st.error("Please provide your Claude API key to generate ideas.")
    else:
        with st.spinner("Generating research ideas with Claude... This may take a minute."):
            ideas = generate_ideas_with_claude(
                st.session_state.papers, 
                claude_api_key, 
                user_nickname,
                ideas_to_generate
            )
            
            st.subheader("Generated Research Ideas")
            st.markdown(ideas)
            
            # Add a download button for the ideas
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_ideas_{current_time}.txt"
            
            st.download_button(
                label="Download Ideas as Text",
                data=ideas,
                file_name=filename,
                mime="text/plain"
            )

# If no papers have been searched yet
if not st.session_state.get("papers"):
    st.info("Enter your search criteria and click 'Search Papers' to begin.")