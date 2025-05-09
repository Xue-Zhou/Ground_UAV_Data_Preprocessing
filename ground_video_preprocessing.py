import cv2
import os

def crop_video(input_video_path, cropped_video_path, crop_left_ratio=0.0, crop_right_ratio=0.0, crop_top_ratio=0.0):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read video: {input_video_path}")
        return False

    height, width = frame.shape[:2]
    left = int(width * crop_left_ratio)
    right = int(width * (1 - crop_right_ratio))
    top = int(height * crop_top_ratio)

    cropped_width = right - left
    cropped_height = height - top

    out = cv2.VideoWriter(cropped_video_path, fourcc, fps, (cropped_width, cropped_height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[top:, left:right]
        out.write(cropped_frame)

    cap.release()
    out.release()
    print(f"[✔] Cropped video saved to: {cropped_video_path}")
    return True

def split_video_half(cropped_video_path, left_video_path, right_video_path):
    cap = cv2.VideoCapture(cropped_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read cropped video: {cropped_video_path}")
        return False

    height, width = frame.shape[:2]
    mid = width // 2

    out_left = cv2.VideoWriter(left_video_path, fourcc, fps, (mid, height))
    out_right = cv2.VideoWriter(right_video_path, fourcc, fps, (width - mid, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_left.write(frame[:, :mid])
        out_right.write(frame[:, mid:])

    cap.release()
    out_left.release()
    out_right.release()
    print(f"[✔] Split videos saved: {left_video_path}, {right_video_path}")
    return True

def extract_frames(video_path, output_dir, frame_interval=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"[✔] Extracted {saved_count} frames to: {output_dir}")

def process_all_videos(input_dir, output_base_dir, crop_left, crop_right, crop_top, frame_interval=10):
    video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_base_dir, f"{video_name}_cropped")
        os.makedirs(output_dir, exist_ok=True)

        cropped_path = os.path.join(output_dir, f"cropped_{video_name}.mp4")
        left_path = os.path.join(output_dir, f"cropped_{video_name}_left.mp4")
        right_path = os.path.join(output_dir, f"cropped_{video_name}_right.mp4")

        # Step 1: Crop video
        if crop_video(input_path, cropped_path, crop_left, crop_right, crop_top):
            # Step 2: Split video
            if split_video_half(cropped_path, left_path, right_path):
                # Step 3: Extract frames from left and right halves
                left_frames_dir = os.path.join(output_base_dir, f"{video_name}_left_frames")
                right_frames_dir = os.path.join(output_base_dir, f"{video_name}_right_frames")

                extract_frames(left_path, left_frames_dir, frame_interval)
                extract_frames(right_path, right_frames_dir, frame_interval)

# ---------- Example usage ----------
if __name__ == "__main__":
    input_videos_folder = "/path/to/the/video"
    output_folder = "/path/to/save/the/processed_video"

    crop_left = 0.1
    crop_right = 0.22
    crop_top = 0.33
    frame_interval = 30

    process_all_videos(
        input_dir=input_videos_folder,
        output_base_dir=output_folder,
        crop_left=crop_left,
        crop_right=crop_right,
        crop_top=crop_top,
        frame_interval=frame_interval
    )
