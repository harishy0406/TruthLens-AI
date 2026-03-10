import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of rows to generate
N = 10000

# Possible values based on your original dataset structure
platforms = ['Reddit', 'Telegram', 'Twitter', 'Facebook']
countries = ['USA', 'Germany', 'India', 'UK', 'Brazil']
cities = {
    'USA': ['New York', 'Chicago', 'Los Angeles'],
    'Germany': ['Berlin', 'Hamburg', 'Munich'],
    'India': ['Delhi', 'Bangalore', 'Mumbai'],
    'UK': ['London', 'Manchester', 'Birmingham'],
    'Brazil': ['Sao Paulo', 'Brasilia', 'Rio de Janeiro']
}
timezones = ['EST', 'CET', 'IST', 'GMT', 'BRT']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
model_signatures = ['GPT-like', 'human', 'unknown']
factcheck_verdicts = ['TRUE', 'FALSE', 'PARTLY', 'UNVERIFIED']
author_verified_options = [0, 1]

# Fake news relevant text templates (AI + misinformation themed)
fake_news_templates = [
    "Breaking: New AI model from {company} can perfectly clone any voice - experts warn of massive deepfake threat!",
    "Alert: Governments using AI to spy on citizens through social media - leaked documents prove it!",
    "Shocking: AI chatbots are secretly manipulating elections in 2025 - whistleblower reveals all!",
    "Warning: This AI image generator is creating undetectable fake news photos - here's proof!",
    "Exclusive: OpenAI just admitted their model is spreading dangerous misinformation about climate change!",
    "Viral claim: AI will replace 80% of jobs by 2027 - but insiders say it's a hoax to sell more tech!",
    "Fact-check needed: Viral video shows AI robot attacking humans - is it real or deepfake propaganda?",
    "Urgent: Big Tech AI companies are censoring truth about vaccine side effects using advanced filters!",
    "Leaked: Elon Musk's xAI is training models to push political agendas - sources confirm!",
    "Scam alert: Fake AI investment apps promising 1000% returns are stealing user data worldwide!",
    "Conspiracy exposed: AI is behind the rise in global misinformation - controlled by shadow governments!",
    "New study: Using ChatGPT increases your chance of believing fake news by 300% - researchers shocked!",
    "Viral hoax: AI predicts world ends in 2026 - but it's just another synthetic content farm!",
    "Shocking report: Social media algorithms powered by AI are radicalizing users faster than ever!",
    "Truth revealed: All those 'AI-generated art' posts are actually stealing from real artists!",
    "Emergency: Deepfake videos of world leaders are being used to start wars - military on alert!",
    "Hidden: Your phone's AI assistant is recording everything and selling it to advertisers!",
    "Fake news epidemic: AI tools now generate entire news sites in seconds - journalism is dead!",
    "Whistleblower: Google AI is suppressing conservative views on purpose!",
    "Viral lie: AI will solve world hunger by 2030 - experts call it corporate propaganda!"
]

# Companies for templates
companies = ['OpenAI', 'Google', 'Meta', 'xAI', 'Microsoft', 'Anthropic', 'Tesla']

# Generate data
data = {
    'id': range(1, N + 1),
    'post_id': [f'P{str(i).zfill(4)}' for i in range(1001, 1001 + N)],
    'platform': np.random.choice(platforms, N),
}

# Generate dates (2024-2026)
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=random.randint(0, 900)) for _ in range(N)]
data['timestamp'] = [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates]
data['date'] = [d.strftime('%Y-%m-%d') for d in dates]
data['time'] = [d.strftime('%H:%M:%S') for d in dates]
data['month'] = [d.strftime('%B') for d in dates]
data['weekday'] = [d.strftime('%A') for d in dates]

data['country'] = np.random.choice(countries, N)
data['city'] = [random.choice(cities[c]) for c in data['country']]
data['timezone'] = [timezones[countries.index(c) % len(timezones)] for c in data['country']]

data['author_id'] = [f'A{random.randint(1000, 9999)}' for _ in range(N)]
data['author_followers'] = np.random.randint(500, 1000000, N)
data['author_verified'] = np.random.choice(author_verified_options, N)

# Generate fake-news relevant text
texts = []
for _ in range(N):
    template = random.choice(fake_news_templates)
    company = random.choice(companies)
    text = template.format(company=company)
    # Add some variation
    if random.random() > 0.7:
        text += f" {random.choice(['Sources say this is 100% real!', 'This changes everything!', 'Share before it gets deleted!'])}"
    texts.append(text)
data['text'] = texts

data['text_length'] = [len(t) for t in texts]
data['token_count'] = [len(t.split()) + random.randint(-5, 15) for t in texts]
data['readability_score'] = np.round(np.random.uniform(30, 80, N), 2)

data['num_urls'] = np.random.randint(0, 4, N)
data['num_mentions'] = np.random.randint(0, 6, N)
data['num_hashtags'] = np.random.randint(0, 6, N)

data['sentiment_score'] = np.round(np.random.uniform(-1.0, 1.0, N), 3)
data['toxicity_score'] = np.round(np.random.uniform(0.0, 1.0, N), 3)

data['model_signature'] = np.random.choice(model_signatures, N)
data['detected_synthetic_score'] = np.round(np.random.uniform(0.0, 1.0, N), 3)
data['embedding_sim_to_facts'] = np.round(np.random.uniform(0.0, 1.0, N), 3)
data['factcheck_verdict'] = np.random.choice(factcheck_verdicts, N)
data['external_factchecks_count'] = np.random.randint(0, 6, N)
data['source_domain_reliability'] = np.round(np.random.uniform(0.0, 1.0, N), 3)
data['engagement'] = np.random.randint(100, 10000, N)

# 50% misinformation rate (fake news focus)
data['is_misinformation'] = np.random.choice([0, 1], N, p=[0.5, 0.5])

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('ai_news_extended_10k.csv', index=False)

print(f"✅ Dataset successfully generated with {len(df)} rows!")
print("File saved as 'ai_news_extended_10k.csv'")
print("\nFirst 5 rows preview:")
print(df.head())
print("\nColumns match original + fake news content added in 'text' column.")