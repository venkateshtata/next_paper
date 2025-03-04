import streamlit as st
import requests
import json
import anthropic
import arxiv
import os
import tempfile
from datetime import datetime

# Set the theme to light mode
os.environ['STREAMLIT_THEME'] = 'light'
os.environ['STREAMLIT_THEME_BASE'] = 'light'

# Set page configuration with explicit light theme
st.set_page_config(
    page_title="Next AI Paper",
    page_icon="research-paper.png",  # Use the research paper icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide theme selector and other default elements
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
section[data-testid="stSidebar"] div.stRadio {display: none;}
section[data-testid="stSidebar"] div span:has(button[kind="secondary"]) {display: none;}
/* Remove padding from top of sidebar */
section[data-testid="stSidebar"] > div {
    padding-top: 0rem;
}
/* Remove title from sidebar */
section[data-testid="stSidebar"] div:first-child h1 {
    display: none;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# CSS for a minimalist black and white UI
st.markdown("""
<style>
    /* Global styles */
    .main {
        padding: 1rem;
        font-family: 'Inter', sans-serif;
        color: #000000;
        background-color: #ffffff;
    }
    
    /* Button styling - smaller with curved borders */
    .stButton button {
        width: 48%;
        border-radius: 20px;
        background-color: #000000;
        color: white;
        border: none;
        padding: 0.25rem 0.5rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s;
        margin: 2px auto;
        display: inline-block;
    }
    .stButton button:hover {
        background-color: #333333;
    }
    
    /* Custom button class for the Generate Ideas button */
    .generate-button button {
        background-color: #d63384 !important;
        width: 60% !important;
    }
    .generate-button button:hover {
        background-color: #ae296c !important;
    }
    
    /* Custom search button styling */
    .search-button button {
        background-color: #000000 !important;
        width: 40% !important;
    }
    
    /* Clear button styling */
    .clear-button button {
        background-color: #6c757d !important;
        width: 30% !important;
    }
    
    /* Container adjustments */
    .reportview-container {
        margin-top: -2em;
    }
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    /* Reduce vertical spacing between elements */
    div.row-widget.stButton {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    .stButton > button {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    /* Custom container for buttons to reduce gap */
    .button-container {
        margin-top: -10px;
        margin-bottom: -10px;
        padding: 0;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-weight: 600;
        color: #000000;
    }
    
    /* Paper item styling */
    .paper-item {
        border-bottom: 1px solid #f0f0f0;
        padding: 12px 0;
        margin-bottom: 5px;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        display: flex;
        align-items: center;
    }
    
    /* Scrollable containers */
    .scrollable-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 15px;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        background-color: #fafafa;
    }
    
    /* Expander styling */
    .stExpander {
        border: none !important;
    }
    
    /* Info box styling */
    .stAlert {
        background-color: #f8f8f8;
        border-left: 3px solid #000000;
        padding: 1rem;
    }
    
    /* Input fields and selects */
    .stTextInput input, .stSelectbox > div > div {
        border-radius: 3px;
        border: 1px solid #e0e0e0;
    }
    
    /* Style the dropdown items */
    .stSelectbox [data-baseweb="select"] > div:first-child {
        border-radius: 3px;
        background-color: #ffffff;
    }
    
    /* For the ideas container */
    .ideas-container {
        max-height: 800px;
        overflow-y: auto;
        padding: 20px;
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        margin-top: 20px;
    }
    
    /* Logo styling */
    .logo-text {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    
    .logo-tagline {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for API key and configuration
with st.sidebar:
    # Add some spacing at the top
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    
    # Add the research paper icon at the top of the sidebar
    from PIL import Image
    try:
        # Center the icon in the sidebar
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open("research-paper.png")
            st.image(image, width=100, use_column_width=True)
    except Exception as e:
        print(f"Error loading icon: {e}")
    
    # Add divider below the icon
    st.divider()
    
    # API Section
    st.markdown('<p style="font-weight: 600; margin-bottom: 5px;">Account</p>', unsafe_allow_html=True)
    claude_api_key = st.text_input("Claude API Key", type="password", 
                                  help="Required to generate research ideas")
    user_nickname = st.text_input("Your Nickname", value="Researcher",
                                 help="Used to personalize generated ideas")
    
    st.divider()
    
    # Search Section
    st.markdown('<p style="font-weight: 600; margin-bottom: 5px;">Search Parameters</p>', unsafe_allow_html=True)
    search_type = st.radio("Search by:", ["Topic", "Conference", "Both"])
    
    # Add search tips
    with st.expander("📋 Search Tips"):
        st.markdown("""
        **Topic Search Tips:**
        - Use specific phrases like "diffusion models" or "graph neural networks"
        - Include application areas like "medical image segmentation" or "robot navigation"
        - Try trending topics like "Large Language Models" or "Reinforcement Learning from Human Feedback"
        
        **Conference Tips:**
        - Use conference acronyms: CVPR, ICLR, NeurIPS, ACL, ICML, etc.
        - For computer vision research, try: CVPR, ICCV, ECCV
        - For NLP research, try: ACL, EMNLP, NAACL
        - For machine learning research, try: NeurIPS, ICLR, ICML
        """)
    
    topic = None
    conference = None
    if search_type == "Topic" or search_type == "Both":
        topic = st.text_input("Research Topic", 
                            help="E.g., 'Reinforcement Learning', 'Diffusion Models', 'Graph Neural Networks'")
    if search_type == "Conference" or search_type == "Both":
        conference = st.text_input("Conference", 
                                 help="E.g., CVPR, ICLR, NeurIPS, ACL")
    
    st.divider()
    
    # Advanced Settings
    st.markdown('<p style="font-weight: 600; margin-bottom: 5px;">Advanced Settings</p>', unsafe_allow_html=True)
    max_results = st.slider("Papers to retrieve", 
                           min_value=20, max_value=500, value=100, step=10,
                           help="Number of papers to search for")
    
    ideas_to_generate = st.slider("Ideas to generate", 
                                min_value=3, max_value=15, value=5, step=1,
                                help="Number of research ideas to generate")
    
    model_choice = st.selectbox(
        "Claude model",
        options=["claude-2.0", "claude-2.1", "claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
        index=0,
        help="Select which Claude model to use. More advanced models may provide better results but could be slower or more likely to be overloaded."
    )
    
    # Research Environment Settings
    st.divider()
    st.markdown('<p style="font-weight: 600; margin-bottom: 5px;">Research Environment</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.8rem; color: #666;">Help Claude understand your research capabilities</p>', unsafe_allow_html=True)
    
    # Initialize session state for research settings if not exists
    if 'research_settings' not in st.session_state:
        st.session_state.research_settings = {
            'gpu_count': 1,
            'gpu_memory': 16,
            'gpu_type': 'NVIDIA RTX 3090',
            'time_constraint': 'Medium (3-6 months)',
            'coding_expertise': 'Intermediate',
            'collaboration': 'Solo researcher'
        }
    
    # GPU resources
    gpu_count = st.number_input("Number of GPUs", min_value=0, max_value=32, value=st.session_state.research_settings['gpu_count'], help="How many GPUs you have access to")
    gpu_memory = st.number_input("GPU Memory (GB per GPU)", min_value=4, max_value=128, value=st.session_state.research_settings['gpu_memory'], help="Memory size of each GPU in GB")
    
    gpu_type = st.selectbox(
        "GPU Type",
        options=["NVIDIA RTX 3090", "NVIDIA RTX 4090", "NVIDIA A100", "NVIDIA H100", "NVIDIA V100", "NVIDIA T4", "Other/Mixed"],
        index=0,
        help="Type of GPU hardware available"
    )
    
    # Research constraints
    time_constraint = st.selectbox(
        "Time Constraint",
        options=["Short (1-3 months)", "Medium (3-6 months)", "Long (6+ months)"],
        index=1,
        help="Available time to complete the research"
    )
    
    coding_expertise = st.selectbox(
        "Coding Expertise",
        options=["Beginner", "Intermediate", "Advanced"],
        index=1,
        help="Your level of programming expertise"
    )
    
    collaboration = st.selectbox(
        "Collaboration",
        options=["Solo researcher", "Small team (2-3)", "Larger team (4+)"],
        index=0,
        help="Whether you're working alone or in a team"
    )
    
    # Save research settings to session state
    st.session_state.research_settings = {
        'gpu_count': gpu_count,
        'gpu_memory': gpu_memory,
        'gpu_type': gpu_type,
        'time_constraint': time_constraint,
        'coding_expertise': coding_expertise,
        'collaboration': collaboration
    }

# Main content
st.markdown('<h1 class="logo-text">Next AI Paper</h1>', unsafe_allow_html=True)
st.markdown('<p class="logo-tagline">Beam search your next AI paper idea.</p>', unsafe_allow_html=True)

# Function to search arxiv for papers
def search_arxiv(topic=None, conference=None, max_results=100):
    query = ""
    
    # Map common conferences to their arXiv categories and keywords
    conference_mapping = {
        "CVPR": "cs.CV",
        "ICCV": "cs.CV",
        "ECCV": "cs.CV",
        "ICLR": "cs.LG",
        "NeurIPS": "cs.LG OR cs.AI",
        "ICML": "cs.LG",
        "ACL": "cs.CL",
        "EMNLP": "cs.CL",
        "NAACL": "cs.CL",
        "AAAI": "cs.AI",
        "IJCAI": "cs.AI",
        "SIGGRAPH": "cs.GR",
        "KDD": "cs.DB OR cs.LG",
        "SIGIR": "cs.IR",
        "ICRA": "cs.RO",
        "IROS": "cs.RO"
    }
    
    # AI/ML research categories to search within by default
    default_categories = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.RO", "cs.NE", "stat.ML"]
    
    # Build the query with quotes for exact phrase matching
    categories = default_categories.copy()  # Always start with a copy to avoid modifying the original
    
    if topic and conference:
        # Handle case where both topic and conference are specified
        conference_upper = conference.upper() if conference else ""
        conference_cat = conference_mapping.get(conference_upper, "")
        
        # If conference is mapped to a category, use it as filter and topic as search term
        if conference_cat and conference_cat in list(conference_mapping.values()):
            query = f'"{topic}"'
            if " OR " in conference_cat:
                categories = [cat.strip() for cat in conference_cat.split(" OR ")]
            else:
                categories = [conference_cat]
        else:
            # Otherwise treat conference as keyword
            query = f'"{topic}" AND "{conference}"'
    elif topic:
        # Only topic specified
        query = f'"{topic}"'
    elif conference:
        # Only conference specified
        conference_upper = conference.upper() if conference else ""
        conference_cat = conference_mapping.get(conference_upper, "")
        
        if conference_cat and conference_cat in list(conference_mapping.values()):
            query = conference
            if " OR " in conference_cat:
                categories = [cat.strip() for cat in conference_cat.split(" OR ")]
            else:
                categories = [conference_cat]
        else:
            query = f'"{conference}"'
    else:
        return []
    
    print(f"Searching arXiv with query: {query}, categories: {categories}")
    
    # Current arxiv library doesn't support categories parameter directly
    # So we need to include it in the query string
    if categories and isinstance(categories, list) and len(categories) > 0:
        # Add category filtering to the query
        category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
        if query:
            # Combine with the existing query using AND
            query = f"({query}) AND ({category_filter})"
        else:
            query = category_filter
    
    print(f"Final arXiv query: {query}")
    
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
def generate_ideas_with_claude(papers, api_key, nickname, num_ideas=5, model="claude-2.0"):
    if not api_key:
        return "Please provide a Claude API key to generate ideas."
    
    # Create context from papers
    paper_summaries = []
    for i, paper in enumerate(papers[:50]):  # Limit to 50 papers to fit in context window
        summary = f"Paper {i+1}:\nTitle: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\n"
        summary += f"Published: {paper['published']}\nSummary: {paper['summary']}\n\n"
        paper_summaries.append(summary)
    
    context = "\n".join(paper_summaries)
    
    # Get research environment settings
    research_env = st.session_state.research_settings
    
    # Create a description of computational resources
    compute_description = f"{research_env['gpu_count']} {research_env['gpu_type']} GPU(s) with {research_env['gpu_count'] * research_env['gpu_memory']} GB total memory"
    
    # Create prompt for Claude
    prompt = f"""You are a research advisor for {nickname}. Based on the following recent research papers, generate {num_ideas} novel and promising research ideas.

RESEARCH ENVIRONMENT INFORMATION:
- Computational Resources: {compute_description}
- Time Constraint: {research_env['time_constraint']}
- Coding Expertise: {research_env['coding_expertise']}
- Team Structure: {research_env['collaboration']}

For each idea, provide an in-depth analysis with the following structure:

## [TITLE OF RESEARCH IDEA]

### 🔍 Research Gap/Opportunity
[Detailed explanation of the research gap this addresses]

### 🧠 Methodology Approach
[Step-by-step methodology, specifically tailored to the available computational resources]

### 💻 Implementation Details
- Model architecture considerations given the GPU constraints
- Estimated training time on the available hardware
- Potential optimization techniques for the available resources

### 🌟 Potential Impact
[Why this research would be impactful for the field]

### 📚 Foundation Papers
[2-3 key papers from the provided list that would serve as a foundation, with brief explanation of how each relates]

### ⚠️ Potential Challenges
[Discussion of technical challenges and how to address them with the available resources]

Recent papers:
{context}

Generate {num_ideas} detailed research ideas that build upon these papers but explore new directions or combine approaches in novel ways. The ideas should be realistic given the researcher's computational resources, time constraints, and expertise level.
"""

    try:
        # Directly use requests to call the Anthropic API without using the client
        import requests
        from requests.auth import HTTPBasicAuth
        import time
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "prompt": f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
            "model": model,
            "max_tokens_to_sample": 4000,
            "temperature": 0.7
        }
        
        # Try up to 3 times with exponential backoff
        max_retries = 3
        retry_delay = 2  # start with 2 seconds
        
        # Determine which API endpoint to use based on the model
        if model.startswith("claude-3"):
            # Claude 3 models use the messages API
            api_endpoint = "https://api.anthropic.com/v1/messages"
            # Update data structure for Claude 3 API
            claude3_data = {
                "model": model,
                "max_tokens": 4000,
                "temperature": 0.7,
                "system": "You are a helpful AI research advisor tasked with generating novel research ideas based on recent publications.",
                "messages": [{"role": "user", "content": prompt}]
            }
            data = claude3_data
        else:
            # Claude 2 models use the complete API
            api_endpoint = "https://api.anthropic.com/v1/complete"
        
        for attempt in range(max_retries):
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                # Success!
                response_json = response.json()
                
                # Different response structure for Claude 3 vs Claude 2
                if model.startswith("claude-3"):
                    # Claude 3 models return content in a different format
                    try:
                        content = response_json.get("content", [])
                        if content and len(content) > 0:
                            return content[0].get("text", "No text content found in response")
                        else:
                            return "No content returned in response"
                    except Exception as e:
                        print(f"Error parsing Claude 3 response: {e}")
                        print(f"Response JSON: {response_json}")
                        return "Error parsing Claude 3 response"
                else:
                    # Claude 2 models return completion directly
                    return response_json.get("completion", "No completion in response")
            
            elif response.status_code == 529:  # Overloaded error
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    # Wait with exponential backoff before retrying
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the delay for next attempt
                    continue
                else:
                    return """
                    ### The Claude API is currently experiencing high traffic
                    
                    Please try again in a few minutes. The Anthropic servers are currently overloaded with requests.
                    
                    **Tips:**
                    - Try selecting fewer papers (3-5)
                    - Try reducing the number of ideas to generate
                    - Wait a minute and try again
                    """
            else:
                # Other error - don't retry
                return f"Error from Anthropic API: {response.status_code} - {response.text}"
        
        # If we exhausted all retries
        return "Failed to get a response from Claude after multiple attempts. Please try again later."
            
    except Exception as e:
        # Print full exception details for debugging
        import traceback
        error_details = f"Error when calling Claude API: {str(e)}\n{traceback.format_exc()}"
        print(error_details)
        return error_details

# Initialize session state for selected papers if not exists
if 'selected_papers' not in st.session_state:
    st.session_state.selected_papers = []

# Search Papers button with styling
st.markdown('<div class="button-container" style="margin-bottom: -15px;">', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="search-button">', unsafe_allow_html=True)
    
    search_clicked = st.button("Search Papers")
    
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Application flow
if search_clicked:
    if not (topic or conference):
        st.error("Please provide at least a topic or conference name to search.")
    else:
        with st.spinner("Searching for papers..."):
            papers = search_arxiv(topic, conference, max_results)
            
            if not papers:
                st.warning("No papers found matching your criteria. Try adjusting your search terms.")
            else:
                st.session_state.papers = papers
                # Save search parameters for display
                st.session_state.last_search = {
                    "topic": topic,
                    "conference": conference,
                    "count": len(papers)
                }
                # Reset selected papers when new search is performed
                st.session_state.selected_papers = []
                
                # Display more informative success message
                if topic and conference:
                    st.success(f"Found {len(papers)} papers on '{topic}' related to {conference}.")
                elif topic:
                    st.success(f"Found {len(papers)} papers on '{topic}'.")
                elif conference:
                    st.success(f"Found {len(papers)} papers from {conference}.")

# Display papers and selection checkboxes if papers exist
if st.session_state.get("papers"):
    # Display the search parameters in the header
    if st.session_state.get("last_search"):
        search_info = st.session_state.last_search
        search_description = ""
        if search_info["topic"] and search_info["conference"]:
            search_description = f"on '{search_info['topic']}' related to {search_info['conference']}"
        elif search_info["topic"]:
            search_description = f"on '{search_info['topic']}'"
        elif search_info["conference"]:
            search_description = f"from {search_info['conference']}"
            
        st.subheader(f"Retrieved Papers ({search_info['count']} papers {search_description})")
    else:
        st.subheader("Retrieved Papers")
        
    st.markdown("Select up to 10 papers that you want to use for generating research ideas.")
    
    # Display count of selected papers
    st.info(f"Selected {len(st.session_state.selected_papers)} of 10 maximum papers")
    
    # Add search/filter functionality
    search_term = st.text_input("🔍 Filter papers by title, author, or content", "")
    
    # Filter papers based on search term if provided
    displayed_papers = st.session_state.papers
    if search_term:
        search_term = search_term.lower()
        displayed_papers = [
            paper for paper in st.session_state.papers
            if search_term in paper['title'].lower() or
               search_term in paper['summary'].lower() or
               any(search_term in author.lower() for author in paper['authors'])
        ]
        st.write(f"Found {len(displayed_papers)} papers matching '{search_term}'")
    
    # Create a fixed-height container for scrolling
    papers_container = st.container()
    
    # Add papers to scrollable container
    with papers_container:
        # Create the scrollable container
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
        
        # Display all papers with selection checkboxes
        for i, paper in enumerate(displayed_papers):
            # Paper container
            st.markdown(f'<div class="paper-item">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 15])
            
            # Determine if this paper should be checked
            is_selected = paper in st.session_state.selected_papers
            
            # Disable checkbox if 10 papers already selected and this one isn't selected
            max_reached = len(st.session_state.selected_papers) >= 10
            should_disable = max_reached and not is_selected
            
            # Checkbox for selection
            if col1.checkbox("", value=is_selected, key=f"paper_{i}", 
                           disabled=should_disable):
                if paper not in st.session_state.selected_papers:
                    st.session_state.selected_papers.append(paper)
            else:
                if paper in st.session_state.selected_papers:
                    st.session_state.selected_papers.remove(paper)
            
            # Paper title and basic info
            col2.markdown(f"**{i+1}. {paper['title']}**")
            col2.markdown(f"*{', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}* | {paper['published']}")
            
            # Paper details in expander
            with col2.expander("View Abstract"):
                st.write(f"**Full Authors:** {', '.join(paper['authors'])}")
                st.write(f"**Published:** {paper['published']}")
                st.write(f"**Summary:** {paper['summary']}")
                st.write(f"**Categories:** {', '.join(paper['categories'])}")
                st.write(f"[View on arXiv]({paper['url']}) | [PDF]({paper['pdf_url']})")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Close the scrollable container
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show selected papers summary
    if st.session_state.selected_papers:
        with st.expander("View Selected Papers"):
            for i, paper in enumerate(st.session_state.selected_papers):
                st.markdown(f"**{i+1}. {paper['title']}**")
                st.markdown(f"*{', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}*")
    
    # Clear selection button
    if st.session_state.selected_papers:
        with st.container():
            st.markdown('<div class="clear-button">', unsafe_allow_html=True)
            
            if st.button("Clear Selection"):
                st.session_state.selected_papers = []
                st.experimental_rerun()
                
            st.markdown('</div>', unsafe_allow_html=True)

# Generate ideas button with custom styling
st.markdown('<div class="button-container" style="margin-bottom: -15px;">', unsafe_allow_html=True)
with st.container():
    # Apply custom class to this container
    st.markdown('<div class="generate-button">', unsafe_allow_html=True)
    
    # Generate button
    generate_clicked = st.button("Generate Research Ideas")
    
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if generate_clicked:
    if not claude_api_key:
        st.error("Please provide your Claude API key to generate ideas.")
    elif not st.session_state.get("selected_papers"):
        st.error("Please select at least one paper before generating ideas.")
    else:
        with st.spinner("Generating research ideas with Claude... This may take a minute."):
            ideas = generate_ideas_with_claude(
                st.session_state.selected_papers,  # Use selected papers instead of all papers
                claude_api_key, 
                user_nickname,
                ideas_to_generate,
                model_choice
            )
            
            st.subheader("Generated Research Ideas")
            
            # Add some styling for the research ideas
            st.markdown("""
            <style>
            .ideas-container {
                padding: 20px;
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #f0f0f0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .ideas-container h2 {
                color: #000;
                padding-bottom: 8px;
                border-bottom: 2px solid #d63384;
                margin-top: 30px;
            }
            .ideas-container h3 {
                color: #2c3e50;
                margin-top: 20px;
                font-size: 1.2rem;
            }
            .ideas-container ul, .ideas-container ol {
                padding-left: 25px;
            }
            .ideas-container blockquote {
                border-left: 3px solid #d63384;
                padding-left: 15px;
                color: #555;
                font-style: italic;
                margin: 15px 0;
            }
            /* Style for the emoji before list items */
            .ideas-container ul li::before {
                content: "•";
                color: #d63384;
                font-weight: bold;
                display: inline-block; 
                width: 1em;
                margin-left: -1em;
            }
            /* Style for code blocks */
            .ideas-container code {
                background-color: #f8f8f8;
                border-radius: 3px;
                padding: 2px 5px;
                font-family: monospace;
                font-size: 0.9em;
            }
            /* Style for the action buttons below the ideas */
            div.row-widget.stButton button {
                background-color: #f8f8f8;
                color: #333;
                border: 1px solid #ddd;
                transition: all 0.3s;
            }
            div.row-widget.stButton button:hover {
                background-color: #f0f0f0;
                border-color: #d63384;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create scrollable container for ideas
            st.markdown('<div class="ideas-container">', unsafe_allow_html=True)
            
            # Add a decorative header
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <p style="font-size: 0.9rem; color: #666;">Generated using {model_choice}</p>
                <p style="font-size: 0.8rem; color: #888;">Based on {len(st.session_state.selected_papers)} selected papers</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the ideas with Markdown rendering
            st.markdown(ideas)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add quick action buttons
            st.markdown('<div style="margin-top: 30px; margin-bottom: 10px; text-align: center;">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1,1,1])
            
            # Current time for filenames
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Text file download
            with col1:
                filename_txt = f"research_ideas_{current_time}.txt"
                st.download_button(
                    label="📄 Download as Text",
                    data=ideas,
                    file_name=filename_txt,
                    mime="text/plain"
                )
            
            # Markdown file download
            with col2:
                filename_md = f"research_ideas_{current_time}.md"
                st.download_button(
                    label="📝 Download as Markdown",
                    data=ideas,
                    file_name=filename_md,
                    mime="text/markdown"
                )
            
            # Generate new ideas button
            with col3:
                if st.button("🔄 Generate New Ideas"):
                    st.experimental_rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# If no papers have been searched yet
if not st.session_state.get("papers"):
    st.info("Enter your search criteria and click 'Search Papers' to begin.")