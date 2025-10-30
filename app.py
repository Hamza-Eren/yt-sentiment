# Kütüphaneler
import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import yt_dlp
from urllib.parse import urlparse, parse_qs
from transformers import pipeline
import plotly.express as px

# YouTube API
api_key = st.secrets["api_key"]
youtube = build("youtube", "v3", developerKey=api_key)

# Fonksiyonlar
def get_video_details(youtube_url):
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return {
            "id": info.get('id'),
            "thumbnail": info.get('thumbnail'),
            "title": info.get('title'),
            "channel": info.get('channel'),
            "like_count": info.get('like_count'),
            "comment_count": info.get('comment_count'),
            "formats": info.get("formats", [])
        }

def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query)["v"][0]
    return None

def get_video_details_with_api(video_id):
    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()
    if not response["items"]:
        return None

    item = response["items"][0]
    return {
        "id": item["id"],
        "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "like_count": item["statistics"].get("likeCount", 0),
        "comment_count": item["statistics"].get("commentCount", 0),
        "formats": []
    }

def get_comments(video_id, max_results=100):
    video_comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()

    for item in response["items"]:
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        video_comments.append({
            "author": snippet["authorDisplayName"],
            "text": snippet["textDisplay"],
            "likeCount": snippet["likeCount"],
            "publishedAt": snippet["publishedAt"],
            "replyCount": item["snippet"]["totalReplyCount"]
        })
    return pd.DataFrame(video_comments)

def download_video(youtube_url, format_id, file_name):
    ydl_opts = {
        "format": format_id,
        "outtmpl": f"downloads/{file_name}.%(ext)s",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

def convert_for_download(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="HamzaEren/sentiment_model_final",
        use_auth_token=st.secrets["HUGGINGFACE_TOKEN"]
    )
    
def get_chart(positive, neutral, negative):
    total = positive + neutral + negative
    data = {
        "Duygu": ["Olumlu", "Nötr", "Olumsuz"],
        "Yüzde": [
            round(positive / total * 100, 1),
            round(neutral / total * 100, 1),
            round(negative / total * 100, 1)
        ]
    }

    # Plotly pie chart oluştur
    fig = px.pie(
        data,
        names="Duygu",
        values="Yüzde",
        color="Duygu",
        color_discrete_map={
            "Olumlu": "green",
            "Nötr": "gray",
            "Olumsuz": "red"
        },
        title="Yorumların Duygu Dağılımı"
    )
    return fig

# Değişkenler
data = None
sentiment_model = load_model()

# Conteyner Tanımları
con1 = st.container()
con2 = st.container()
con3 = st.container()
con4 = st.container()
con5 = st.container()
con6 = st.container()
col1, col2 = con3.columns(2)

# Conteynerler
with con1:
    message = con1.text("URL bekleniyor...")
    progress = con1.progress(0)

with con2:
    url = st.text_input("Youtube Video Linki Giriniz :")
    if url:
        message.text("Bilgiler çekiliyor...")
        progress.progress(25)

with con3:
    if url:
        with st.spinner("Video bilgileri alınıyor..."):
            #data = get_video_details(url)
            video_id = extract_video_id(url)
            data = get_video_details_with_api(video_id)

        col1.image(data['thumbnail'])
        col2.text(f"{data['title']}\n{data['channel']}")
        col2.text(f"{data['like_count']} beğeni - {data['comment_count']} yorum")

        message.text("Yorumlar çekiliyor...")
        progress.progress(50)

with con4:
    if data:
        with st.spinner("Lütfen bekleyin..."):
            comments = get_comments(data['id'])
            csv = convert_for_download(comments)

        if "comments" not in st.session_state and data:
            st.session_state["comments"] = get_comments(data['id'])

        if "comments" in st.session_state:
            comments = st.session_state["comments"]
            csv = convert_for_download(comments)
            con4.download_button(
                label="Video yorumlarını indir",
                data=csv,
                file_name="yorumlar.csv",
                mime="text/csv",
                icon=":material/download:"
            )

        message.text("Yorumlar duyguya göre yorumlanıyor...")
        progress.progress(75)

with con5:
    # --- Yorumlara tahmin ekle
    if "comments" in st.session_state:
        df = st.session_state["comments"].copy()
        
        # Modeli yorumlara uygula
        def get_sentiment(text):
            res = sentiment_model(text[:512])[0]  # Uzun yorumlarda truncate
            return res['label'], res['score']
        
        df[['label', 'percent']] = df['text'].apply(lambda x: pd.Series(get_sentiment(x)))
        
        # DataFrame’i göster
        st.subheader("Yorumların duygu analizi")
        st.dataframe(df[['text', 'label', 'percent']])
        
        # CSV olarak indirme opsiyonu
        csv_sentiment = convert_for_download(df)
        st.download_button(
            label="Duygu analizli yorumları indir",
            data=csv_sentiment,
            file_name="yorumlar_duygu.csv",
            mime="text/csv",
            icon=":material/download:"
        )
        
        message.text("Duygu analizi tamamlandı...")
        progress.progress(100)
        
with con6:
    if "comments" in st.session_state:
        fig = get_chart(df['label'].value_counts()['olumlu'], df['label'].value_counts()['notr'], df['label'].value_counts()['olumsuz'])
        st.plotly_chart(fig, use_container_width=True)