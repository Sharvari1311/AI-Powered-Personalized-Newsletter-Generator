import streamlit as st
import requests
import json
from collections import defaultdict
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

users = {
    "Alex Parker (Tech Enthusiast)": {
        "interests": ["AI", "cybersecurity", "blockchain", "startups", "programming"],
        "sources": ["TechCrunch", "Wired", "Ars Technica", "MIT Technology Review"]
    },
    "Priya Sharma (Finance & Business Guru)": {
        "interests": ["Global markets", "startups", "fintech", "cryptocurrency", "economics"],
        "sources": ["Bloomberg", "Financial Times", "Forbes", "CoinDesk"]
    },
    "Marco Rossi (Sports Journalist)": {
        "interests": ["Football", "F1", "NBA", "Olympic sports", "esports"],
        "sources": ["ESPN", "BBC Sport", "Sky Sports F1", "The Athletic"]
    },
    "Lisa Thompson (Entertainment Buff)": {
        "interests": ["Movies", "celebrity news", "TV shows", "music", "books"],
        "sources": ["Variety", "Rolling Stone", "Billboard", "Hollywood Reporter"]
    },
    "David Martinez (Science & Space Nerd)": {
        "interests": ["Space exploration", "AI", "biotech", "physics", "renewable energy"],
        "sources": ["NASA", "Science Daily", "Nature", "Ars Technica"]
    }
}

source_domains = {
    "TechCrunch": "techcrunch.com",
    "Wired": "wired.com",
    "Ars Technica": "arstechnica.com",
    "MIT Technology Review": "technologyreview.com",
    "Bloomberg": "bloomberg.com",
    "Financial Times": "ft.com",
    "Forbes": "forbes.com",
    "CoinDesk": "coindesk.com",
    "ESPN": "espn.com",
    "BBC Sport": "bbc.com/sport",
    "Sky Sports F1": "skysports.com/f1",
    "The Athletic": "theathletic.com",
    "Variety": "variety.com",
    "Rolling Stone": "rollingstone.com",
    "Billboard": "billboard.com",
    "Hollywood Reporter": "hollywoodreporter.com",
    "NASA": "nasa.gov",
    "Science Daily": "sciencedaily.com",
    "Nature": "nature.com"
}

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def get_domain_for_source(source):
    return source_domains.get(source, source.lower().replace(" ", "") + ".com")

def search_source_with_interest(source, interest, num_results=3):
    url = "https://google.serper.dev/search"
    
    source_domain = get_domain_for_source(source)
    query = f"{interest} site:{source_domain}"
    
    payload = json.dumps({
        "q": query,
        "num": num_results
    })
    
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json()
    except Exception as e:
        st.error(f"Error searching for {interest} on {source}: {e}")
        return {}

def source_in_url(url, domain):
    clean_url = url.replace("www.", "")
    clean_domain = domain.replace("www.", "")
    return clean_domain in clean_url

@st.cache_data(ttl=3600)
def fetch_articles_for_user(user_data):
    all_articles = []
    
    with st.spinner(f"Finding articles for {user_data['interests']}..."):
        for source in user_data["sources"]:
            for interest in user_data["interests"]:
                results = search_source_with_interest(source, interest)
                
                if "organic" in results:
                    for item in results["organic"]:
                        source_domain = get_domain_for_source(source)
                        if source_in_url(item.get("link", ""), source_domain):
                            article = {
                                "title": item.get("title", "No title"),
                                "link": item.get("link", "#"),
                                "summary": item.get("snippet", "No summary available"),
                                "source": source,
                                "interest": interest,
                                "content": f"{item.get('title', '')} {item.get('snippet', '')}"
                            }
                            all_articles.append(article)
    
    return categorize_articles(all_articles, user_data)

def categorize_articles(articles, user_data):
    if not articles:
        return []
    
    categories = user_data["interests"]
    vectorizer = TfidfVectorizer(stop_words='english')
    article_texts = [article["content"] for article in articles]

    tfidf_matrix = vectorizer.fit_transform(article_texts)

    for i, article in enumerate(articles):
        best_score = -1
        best_category = article["interest"]
        
        for category in categories:
            category_vector = vectorizer.transform([category])
            score = cosine_similarity(tfidf_matrix[i:i+1], category_vector)[0][0]
            
            if score > best_score:
                best_score = score
                best_category = category
        
        article["category"] = best_category
        article["article_score"] = best_score
    
    unique_articles = []
    seen_urls = set()
    
    for article in articles:
        if article["link"] not in seen_urls:
            seen_urls.add(article["link"])
            unique_articles.append(article)
    
    relevant_articles = [a for a in unique_articles if a["article_score"] > 0.0]
    sorted_articles = sorted(relevant_articles, key=lambda x: x["article_score"], reverse=True)
    
    return sorted_articles[:20]

st.set_page_config(page_title="Source-Based AI Newsletter", page_icon="📰")

st.title("📰 AI-Powered Personalized Newsletter Generator")

selected_user = st.selectbox("Select a user persona", list(users.keys()))

if selected_user:
    st.subheader(f"Newsletter for {selected_user}")
    st.markdown(f"**Interests:** {', '.join(users[selected_user]['interests'])}")
    st.markdown(f"**Sources:** {', '.join(users[selected_user]['sources'])}")

    if st.button("Generate Newsletter") or "articles" in st.session_state:
        with st.spinner("Generating personalized newsletter..."):
            if "articles" not in st.session_state:
                articles = fetch_articles_for_user(users[selected_user])
                st.session_state.articles = articles
            else:
                articles = st.session_state.articles
        
        if not articles:
            st.error("No articles found. Try again or select a different user.")
        else:
            grouped = defaultdict(list)
            for article in articles:
                grouped[article["category"]].append(article)

            for category, items in grouped.items():
                st.markdown(f"### - {category}")
                for art in items:
                    st.markdown(f"**[{art['title']}]({art['link']})**")
                    st.markdown(f"{art['summary'][:200]}...")
                    st.caption(f"Source: {art['source']}")
                    st.markdown("--------")

st.cache_data.clear()
if "articles" in st.session_state:
    del st.session_state.articles