import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import tempfile
import os
from datetime import datetime
import io
import base64

try:
    import supervision as sv
    from ultralytics import YOLO
except ImportError:
    st.error("install required packages: pip install supervision ultralytics")
    st.stop()

class VehicleCounter:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        self.tracker = sv.ByteTrack(track_activation_threshold=0.25)
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8, color_lookup=sv.ColorLookup.TRACK)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50, color_lookup=sv.ColorLookup.TRACK)
        
        self.vehicle_counter = 1
        self.vehicle_id_map = {}
        self.vehicle_counts_per_minute = defaultdict(int)
        self.vehicle_timestamps = {}
        self.counted_vehicles = set()

    def get_minute_from_frame(self, frame_number: int, fps: float) -> int:
        seconds = frame_number / fps
        return int(seconds // 60)

    def process_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        results = self.model(frame, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(results)

        vehicle_classes = [2, 5, 7]  # car, bus, truck
        detections = detections[np.isin(detections.class_id, vehicle_classes)]
        detections = self.tracker.update_with_detections(detections)

        current_minute = self.get_minute_from_frame(frame_number, fps)

        for tracker_id in detections.tracker_id:
            if tracker_id is not None:
                if tracker_id not in self.vehicle_id_map:
                    self.vehicle_id_map[tracker_id] = self.vehicle_counter
                    self.vehicle_counter += 1

                    if tracker_id not in self.counted_vehicles:
                        self.vehicle_counts_per_minute[current_minute] += 1
                        self.vehicle_timestamps[tracker_id] = current_minute
                        self.counted_vehicles.add(tracker_id)

        labels = []
        for tracker_id in detections.tracker_id:
            if tracker_id is not None:
                simple_id = self.vehicle_id_map.get(tracker_id, tracker_id)
                labels.append(f"V{simple_id}")
            else:
                labels.append("")

        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        total_vehicles = len(self.counted_vehicles)
        current_minute_count = self.vehicle_counts_per_minute[current_minute]

        cv2.putText(annotated_frame, f"Total Vehicles: {total_vehicles}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Current Minute ({current_minute+1}): {current_minute_count} vehicles",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated_frame

    def get_counting_summary(self):
        total_vehicles = len(self.counted_vehicles)
        minutes_data = dict(self.vehicle_counts_per_minute)

        return {
            'total_vehicles': total_vehicles,
            'vehicle_counts_per_minute': minutes_data,
            'max_vehicles_per_minute': max(minutes_data.values()) if minutes_data else 0,
            'average_vehicles_per_minute': np.mean(list(minutes_data.values())) if minutes_data else 0
        }

def process_video_with_counting(video_path, progress_callback=None):
    counter = VehicleCounter("yolov8n.pt")
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file dengan ekstensi yang tepat
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        output_path = tmp_file.name
    
    #kompetibel codec
    fourcc = cv2.VideoWriter_fourcc(*'H264')  #ganti 'mp4v' > 'H264'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    #Check if VideoWriter 
    if not out.isOpened():
        #Fallback to other codecs
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace('.mp4', '.avi')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            #Final fallback
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = output_path.replace('.avi', '.mp4')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = counter.process_frame(frame, frame_count, fps)
            
            cv2.putText(processed_frame, "Vehicle Counting & Tracking",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}",
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(processed_frame)
            frame_count += 1
            
            if progress_callback and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                progress_callback(progress, frame_count, total_frames)
    
    finally:
        cap.release()
        out.release()
    
    #pastiin output file exists 
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise Exception("Failed to create output video file")
    
    return output_path, counter

def create_vehicle_count_charts(counter):
    summary = counter.get_counting_summary()
    counts_per_minute = summary['vehicle_counts_per_minute']
    
    if not counts_per_minute:
        st.warning("No vehicle data to display")
        return summary
    
    df_data = []
    for minute, count in counts_per_minute.items():
        df_data.append({
            'Minute': minute + 1,
            'Vehicle_Count': count
        })
    
    df = pd.DataFrame(df_data)
    
    #dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Vehicle Counting Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # bar chart/minute
    if not df.empty:
        ax1 = axes[0, 0]
        bars = ax1.bar(df['Minute'], df['Vehicle_Count'],
                      color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_title('Vehicles Counted Per Minute', fontweight='bold')
        ax1.set_xlabel('Minute')
        ax1.set_ylabel('Number of Vehicles')
        ax1.grid(True, alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    #bar chart
    if not df.empty:
        ax2 = axes[0, 1]
        bars2 = ax2.barh(df['Minute'], df['Vehicle_Count'],
                        color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax2.set_title('Vehicles Per Minute (Horizontal View)', fontweight='bold')
        ax2.set_ylabel('Minute')
        ax2.set_xlabel('Number of Vehicles')
        ax2.grid(True, alpha=0.3)
        
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center')
    
    #line chart
    if not df.empty:
        ax3 = axes[1, 0]
        ax3.plot(df['Minute'], df['Vehicle_Count'],
                marker='o', linewidth=2, markersize=8, color='green')
        ax3.fill_between(df['Minute'], df['Vehicle_Count'], alpha=0.3, color='green')
        ax3.set_title('Vehicle Count Trend Over Time', fontweight='bold')
        ax3.set_xlabel('Minute')
        ax3.set_ylabel('Number of Vehicles')
        ax3.grid(True, alpha=0.3)
    
    #Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    total_vehicles = summary['total_vehicles']
    max_per_minute = summary['max_vehicles_per_minute']
    avg_per_minute = summary['average_vehicles_per_minute']
    total_minutes = len(counts_per_minute)
    
    summary_text = f"""
    VEHICLE COUNTING SUMMARY

    Total: {total_vehicles}

    Total Duration: {total_minutes} minutes

    Most vehicles/minute: {max_per_minute} vehicles

    Average: {avg_per_minute:.1f} vehicles

    The highest vehicle activity is in minutes: {df.loc[df['Vehicle_Count'].idxmax(), 'Minute'] if not df.empty else 'N/A'}

    The least vehicle activity was in minutes: {df.loc[df['Vehicle_Count'].idxmin(), 'Minute'] if not df.empty else 'N/A'}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig, summary, df

def create_csv_report(counter):
    summary = counter.get_counting_summary()
    counts_per_minute = summary['vehicle_counts_per_minute']
    
    csv_data = []
    for minute, count in counts_per_minute.items():
        csv_data.append({
            'Minute': minute + 1,
            'Vehicle_Count': count,
            'Cumulative_Count': sum([counts_per_minute[m] for m in range(minute + 1)])
        })
    
    df = pd.DataFrame(csv_data)
    return df

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    return {
        'fps': fps,
        'resolution': f"{width}x{height}",
        'total_frames': total_frames,
        'duration': f"{duration:.1f}s"
    }

def main():
    st.set_page_config(
        page_title="Vehicle Counter & Tracker",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("Vehicle Counter & Tracker on Toll Road")
    st.markdown("Upload a video to count and track vehicles using YOLO and ByteTrack")
    
    #sidebar
    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox(
        "Choose YOLO Model",
        ["yolov8"],
        help="Larger models are more accurate but slower"
    )
    
    #user upload file
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to process"
    )
    
    if uploaded_file is not None:
        #save user vid
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        #user video info
        st.subheader("Video Information")
        video_info = get_video_info(video_path)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Resolution", video_info['resolution'])
        with col2:
            st.metric("FPS", video_info['fps'])
        with col3:
            st.metric("Total Frames", video_info['total_frames'])
        with col4:
            st.metric("Duration", video_info['duration'])
        
        #review user video
        st.subheader("Original Video")
        st.video(uploaded_file)
        
        if st.button("Process Video", type="primary"):
            st.subheader("Processing Video...")
            
            #progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress, frame_count, total_frames):
                progress_bar.progress(progress / 100)
                status_text.text(f"Processing: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            #Process video
            with st.spinner("Processing video... This may take a while."):
                try:
                    output_path, vehicle_counter = process_video_with_counting(
                        video_path, 
                        progress_callback=update_progress
                    )
                    
                    st.success("✅ Video processing completed!")
                    
                    #verify file, exist/not
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        
                        # Read video file for download
                        with open(output_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        
                        # Show download button instead of video player
                        if len(video_bytes) > 0:
                            st.subheader("Processing Complete!")
                            st.info("📥 Your processed video is ready for download!")
                            
                            # Download processed video button
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=video_bytes,
                                file_name=f"processed_{uploaded_file.name}",
                                mime="video/mp4",
                                use_container_width=True
                            )
                        else:
                            st.error("❌ Processed video file is empty")
                    else:
                        st.error("❌ Failed to create processed video file")
                        st.info("Try using a different video format or check if the video is corrupted")
                    
                    # Create and display charts
                    st.subheader("📊 Analysis Results")
                    
                    fig, summary, df = create_vehicle_count_charts(vehicle_counter)
                    st.pyplot(fig)
                    
                    #summary 
                    st.subheader("📈 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Vehicles", summary['total_vehicles'])
                    with col2:
                        st.metric("Max Vehicles/Minute", int(summary['max_vehicles_per_minute']))
                    with col3:
                        st.metric("Average Vehicles/Minute", f"{summary['average_vehicles_per_minute']:.1f}")
                    with col4:
                        st.metric("Total Duration", f"{len(summary['vehicle_counts_per_minute'])} minutes")
                    
                    #show detailed data
                    if not df.empty:
                        st.subheader("📋 Detailed Count Data")
                        st.dataframe(df, use_container_width=True)
                        
                        csv_report = create_csv_report(vehicle_counter)
                        csv_buffer = io.StringIO()
                        csv_report.to_csv(csv_buffer, index=False)
                        
                        st.subheader("📥 Download Files")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                with open(output_path, 'rb') as video_file:
                                    video_bytes = video_file.read()
                                if len(video_bytes) > 0:
                                    st.download_button(
                                        label="🎬 Download Processed Video",
                                        data=video_bytes,
                                        file_name=f"processed_{uploaded_file.name}",
                                        mime="video/mp4",
                                        use_container_width=True
                                    )
                        
                        with col2:
                            st.download_button(
                                label="📊 Download CSV Report",
                                data=csv_buffer.getvalue(),
                                file_name=f"vehicle_count_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    

                    try:
                        if os.path.exists(output_path):
                            os.unlink(output_path)
                    except Exception:
                        pass  
                    
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                    st.error("Please make sure the video format is supported and try again.")
                    

                    st.info("""
                    **Troubleshooting Tips:**
                    - Make sure the video file is not corrupted
                    - Try using MP4 format with H.264 encoding
                    - Check if the video duration is reasonable
                    - Ensure you have enough storage space
                    """)
        

        try:
            if os.path.exists(video_path):
                os.unlink(video_path)
        except Exception:
            pass  
    
    else:
        st.info("Upload a video file to get started")
        

        st.subheader("How it works(?)")
        st.markdown("""
        1. **Upload Video**: Choose a video file containing vehicles
        2. **Processing**: The app uses YOLOv8 for vehicle detection and ByteTrack for tracking
        3. **Counting**: Vehicles are counted and tracked with unique IDs
        4. **Analysis**: Get detailed statistics and visualizations
        5. **Export**: Download the processed video and CSV report
        
        **Supported vehicle types:** Cars, Trucks, Buses
        
        **Features:**
        - Real-time vehicle detection and tracking
        - Unique ID assignment for each vehicle
        - Per-minute counting statistics
        - Visual traces showing vehicle paths
        - Comprehensive analysis dashboard
        """)

if __name__ == "__main__":
    main()