import streamlit as st
import cv2
import os
from deepface import DeepFace
from collections import defaultdict
import tempfile
import json
import pandas as pd

st.set_page_config(page_title="Emotion Insight", layout="wide")

st.title("🎥 Emotion Analysis from Meeting Video")
st.write("Upload a video file and get a report on participant emotions like attentiveness and disengagement.")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.video(uploaded_video)

    st.info("Analyzing video... (this may take a minute ⏳)")

    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    user_data = defaultdict(list)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % 10 == 0:
            try:
                detections = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                if detections and isinstance(detections, list):
                    for idx, detection in enumerate(detections):
                        region = detection.get('region', {})
                        if 'emotion' in detection and region.get('x', 0) > 0:
                            emotions = detection['emotion']
                            user_data[f'user_{idx+1}'].append(emotions)
            except:
                pass
        frame_num += 1

    cap.release()

    if not user_data:
        st.error("🚫 No face detected in the video.")
    else:
        table_data = []

        for user_id, emotion_list in user_data.items():
            emotion_totals = defaultdict(float)

            for emotions in emotion_list:
                for emotion, value in emotions.items():
                    emotion_totals[emotion] += value

            num_frames = len(emotion_list)
            if num_frames == 0:
                continue

            emotion_avg = {e: v / num_frames for e, v in emotion_totals.items()}

            attentiveness = emotion_avg.get('neutral', 0) + emotion_avg.get('happy', 0) + emotion_avg.get('surprise', 0)
            disengagement = emotion_avg.get('sad', 0) + emotion_avg.get('fear', 0) + emotion_avg.get('disgust', 0)

            total_score = attentiveness + disengagement
            if total_score > 0:
                attentiveness = (attentiveness / total_score) * 100
                disengagement = (disengagement / total_score) * 100

            table_data.append([user_id, round(attentiveness, 2), round(disengagement, 2)])

        df = pd.DataFrame(table_data, columns=["User", "Attentiveness (%)", "Disengagement (%)"])
        st.success("✅ Analysis complete!")
        st.subheader("🎯 Emotion Summary Table")
        st.dataframe(df, use_container_width=True)

        # Optional: Download as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="emotion_summary.csv", mime="text/csv")
