# Ground_UAV_Data_Preprocessing
The scripts are used to process ground videos and UAV images for strawberry field

The UAV_image_preprocessing.py is used to process UAV images to segment out individual strawberry plants for annotation. The output images will contain one plant in each image. AI10_demo.jpg is a sample UAV image captured at 10 m above ground level. AI5_demo.jpg is a sample UAV image captured at 5 m above ground level.

The ground_video_preprocessing.py is used to trim the video footage to remove the background to retain a single plant in view, extract high-resolution frames from the video, and save images for annotation. ground_video_demo.mp4 is a demo video collected at 0.5 m above ground level.
