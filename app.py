from flask import Flask, request
import joblib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("recommend.log", encoding="utf-8"), logging.StreamHandler()]
)

app = Flask(__name__)

# Load data
logging.info("🔁 Loading data...")
try:
    df = joblib.load('df_cleaned.pkl')
    cosine_sim = joblib.load('cosine_sim.pkl')
    logging.info("✅ Data loaded successfully.")
except Exception as e:
    logging.error("❌ Failed to load required files: %s", str(e))
    raise e

# Recommendation logic
def recommend_songs(song_name, top_n=5):
    logging.info("🎵 Recommending songs for: '%s'", song_name)
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        logging.warning("⚠️ Song not found in dataset.")
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]

    result_df = df[['artist', 'song', 'link']].iloc[song_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."
    return result_df

@app.route("/", methods=["GET"])
def index():
    return '''
🎶 Welcome to the Terminal-Based Music Recommender 🎶

Usage: 
http://localhost:5000/recommend?song=YourSongName
'''

@app.route("/recommend", methods=["GET"])
def recommend():
    song_name = request.args.get("song")
    if not song_name:
        return "❗ Please provide a song name using ?song=YourSongName"

    recommendations = recommend_songs(song_name)

    if recommendations is None:
        return f"❌ No recommendations found for '{song_name}'."

    output = f"🎧 Top recommendations for '{song_name}':\n\n"
    for i, row in recommendations.iterrows():
        output += f"{i}. {row['song']} by {row['artist']} - https://www.musixmatch.com{row['link']}\n"
    return output

if __name__ == "__main__":
    print("🎶 Flask Music Recommender is running...")
    print("Visit: http://localhost:5000/")
    app.run(host='0.0.0.0', port=5000, debug=False)
