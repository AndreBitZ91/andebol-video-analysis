import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Andebol IA", layout="wide")
st.title("⚽ Análise de Vídeo de Andebol com IA")
st.markdown("**Upload um vídeo** → deteta jogadores e bola automaticamente com tracking!")

# Carrega o modelo (podes trocar por o teu modelo custom mais tarde)
model = YOLO("yolov8n.pt")  # pré-treinado COCO (excelente para bola + pessoas)

video_file = st.file_uploader("Carrega o vídeo do jogo", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Guarda vídeo temporariamente
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output vídeo processado
    output_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detecção + tracking (ByteTrack interno do YOLOv8)
        results = model.track(frame, persist=True, conf=0.3)
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)
        frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    
    # Mostra o vídeo processado
    st.success("✅ Análise concluída! Jogadores e bola trackeados.")
    with open(output_path, "rb") as f:
        st.video(f.read())
    
    # Limpeza
    os.unlink(video_path)
    os.unlink(output_path)
    
    st.info("Dica: Para detectar golos automaticamente, depois treinamos um modelo custom com o dataset do repo Fast_final.")
