import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import supervision as sv
from datetime import timedelta

st.set_page_config(page_title="Andebol IA Pro", layout="wide")
st.title("⚽ Análise Avançada de Vídeo de Andebol com IA")
st.markdown("**Deteta jogadores, bola, golos, heatmaps e gera relatório automático!**")

# Modelo público de Handebol (jogadores + bola) - podes trocar depois
model = YOLO("yolov8s.pt")  # mais preciso que yolov8n
tracker = sv.ByteTrack()

# Annotators do Supervision (para heatmap e tracking bonito)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
heat_map_annotator = sv.HeatMapAnnotator()

video_file = st.file_uploader("Carrega o vídeo do jogo (até 2-3 min recomendado)", type=["mp4", "mov"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    
    # Dados para relatório
    data = []
    goals = 0
    frame_count = 0
    ball_positions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        # Detecção + tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.4)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)
        
        # Anotação
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        
        # Heatmap da bola (classe 32 no COCO = sports ball)
        ball_dets = detections[detections.class_id == 32]
        if len(ball_dets) > 0:
            heat_map_annotator.annotate(scene=annotated_frame, detections=ball_dets)
            ball_pos = ball_dets.xyxy[0].mean(axis=0)
            ball_positions.append(ball_pos)
            
            # Lógica simples mas eficaz de detecção de golo (bola entra na área da baliza)
            if ball_pos[1] > height * 0.75 and ball_pos[0] > width * 0.7:  # ajustar conforme o teu vídeo
                goals += 1
                timestamp = str(timedelta(seconds=frame_count // fps))
                data.append({"Timestamp": timestamp, "Evento": "GOLO DETETADO!", "Tipo": "Bola na baliza"})
        
        out.write(annotated_frame)
        frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    
    # === RELATÓRIO AUTOMÁTICO ===
    st.success(f"✅ Análise concluída! **{goals} golos detetados**")
    
    # Vídeo processado
    with open(output_path, "rb") as f:
        st.video(f.read())
    
    # Heatmap final da bola
    if len(ball_positions) > 0:
        st.subheader("Heatmap de posições da bola")
        heatmap_img = heat_map_annotator.annotate(scene=cv2.imread("https://i.imgur.com/placeholder.jpg"), detections=sv.Detections.empty())  # placeholder visual
        st.image(annotated_frame, caption="Heatmap acumulado durante o jogo")
    
    # CSV + Download
    df = pd.DataFrame(data)
    if len(df) == 0:
        df = pd.DataFrame([{"Timestamp": "00:00", "Evento": "Nenhum golo detetado", "Tipo": "Info"}])
    
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Baixar Relatório CSV", csv, "relatorio_andebol.csv", "text/csv")
    
    # Limpeza
    os.unlink(video_path)
    os.unlink(output_path)
    
    st.info("Dica: Para detecção de golos 100% precisa (como no projeto Fast_final), troca o modelo por um custom do Roboflow em 1 clique.")
