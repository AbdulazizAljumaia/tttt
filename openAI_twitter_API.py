import requests
from requests_oauthlib import OAuth1

# Twitter API credentials
consumer_key = "xxxxxxxx"
consumer_secret = "xxxxxxxx"
access_token = "xxxxxxxx"
access_token_secret = "xxxxxxxx"

#OpenAI API key
api_key = "xxxxxxxx"

# Example prompt
prompt1 = "Generate an n-gram sentence with the 2 words 'car', 'man'"
prompt2 = "Generate an uni-gram sentence with the  word 'car'"
prompt3 = "Generate a d-gram sentence with the  word 'car'"


def post_tweet(text):
    post_url = "https://api.twitter.com/2/tweets"
    auth = OAuth1(consumer_key, consumer_secret, access_token, access_token_secret)
    tweet_data = {"text": text}
    
    response = requests.post(post_url, auth=auth, json=tweet_data)
    if response.status_code not in [200, 201]:
        raise Exception(response.status_code, response.text)

    return response.json()

def generate_text(prompt, api_key):
    url = "https://api.openai.com/v1/engines/text-davinci-003/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "max_tokens": 150
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")

    return response.json()['choices'][0]['text']
try:
    # Example tweet text

    # Post a tweet
    response = post_tweet(generate_text(prompt1, api_key))
    print("Tweet posted successfully:", response)
except Exception as e:
    print("Error:", e)
