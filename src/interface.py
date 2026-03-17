    import tkinter as tk
    from tkinter import filedialog, messagebox
    import cv2
    import tensorflow as tf
    import numpy as np
    import threading
    from PIL import Image, ImageTk
    
    # Load the trained model
    model = tf.keras.models.load_model('model.h5')
    
    # Function to preprocess and predict the video frames
    def predict_video(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        fake_count = 0
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame: Resize and normalize it before feeding to the model
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = cv2.resize(frame, (224, 224))  # Resize to match the model's input
            frame = np.expand_dims(frame, axis=0)  # Add batch dimension
            frame = frame / 255.0  # Normalize
            
            # Predict using the model
            prediction = model.predict(frame)
            
            if prediction[0] > 0.5:  # Assuming binary classification
                fake_count += 1
            frame_count += 1
    
        # Show the result once all frames are processed
        if fake_count / frame_count > 0.5:
            return "Fake Video"
        else:
            return "Real Video"
    
    # Function to handle file upload and prediction in a new thread
    def upload_and_predict():
        video_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")])
        if not video_path:
            return
    
        # Disable the button to prevent multiple uploads during processing
        upload_button.config(state=tk.DISABLED)
        
        # Start video prediction in a new thread
        result = threading.Thread(target=run_prediction, args=(video_path,))
        result.start()
    
    # Function to run prediction and update the result on the GUI
    def run_prediction(video_path):
        result = predict_video(video_path)
        
        # Update the GUI with the result (running this in the main thread)
        result_label.config(text=result)
        upload_button.config(state=tk.NORMAL)
    
    # Initialize the Tkinter window
    root = tk.Tk()
    root.title("Deepfake Detection")
    
    # Create and pack GUI components
    upload_button = tk.Button(root, text="Upload Video", command=upload_and_predict)
    upload_button.pack(pady=10)
    
    result_label = tk.Label(root, text="Prediction: None")
    result_label.pack(pady=10)
    
    # Run the Tkinter event loop
    root.mainloop()
    
    
