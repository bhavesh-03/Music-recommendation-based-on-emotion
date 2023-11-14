import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random

# Load the pre-trained emotion detection model
emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Spotify API credentials
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="880361f78af640abb72b97bedef19595",
                                               client_secret="554b6006a61540f0b0eaaa37575833af",
                                               redirect_uri="http://localhost:8888/callback",
                                               scope="user-library-read"))

# Define emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(face_image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the input size of the model
    gray_image = cv2.resize(gray_image, (64, 64))
    
    # Normalize the pixel values to be in the range [0, 1]
    gray_image = gray_image / 255.0
    
    # Reshape the image to match the input shape of the model
    gray_image = np.reshape(gray_image, (1, 64, 64, 1))
    
    # Predict emotion
    emotion_prediction = emotion_model.predict(gray_image)
    
    # Get the index with the highest probability
    emotion_index = np.argmax(emotion_prediction)
    
    # Get the corresponding emotion label
    emotion_label = emotion_labels[emotion_index]
    
    return emotion_label

def recommend_music(emotion):
    # Map emotion to Spotify genres, artists, or moods
    emotion_mapping = {
        'Happy': 'https://open.spotify.com/playlist/37i9dQZF1EVJSvZp5AOML2?si=ba77863944f347b7',  # Replace with the actual Spotify URI
        'Sad': 'https://open.spotify.com/playlist/37i9dQZF1EIg6gLNLe52Bd?si=e8c21acb9ff843ac',      # Replace with the actual Spotify URI
        # Add more mappings as needed
    }

    # Get the recommended playlist URI based on the detected emotion
    playlist_uri = emotion_mapping.get(emotion)

    # Get the tracks from the recommended playlist
    if playlist_uri:
        playlist_tracks = sp.playlist_tracks(playlist_uri)

        # Print the response for debugging
        print("Playlist Tracks Response:")
        print(playlist_tracks)

        # Extract track names from the playlist response
        track_names = extract_track_names(playlist_tracks)

        return track_names
    else:
        return None

def extract_track_names(playlist_tracks):
    try:
        # Check if 'tracks' key is present in the response
        if 'tracks' in playlist_tracks:
            items = playlist_tracks['tracks']['items']
        else:
            # If 'tracks' key is not present, assume the response is directly a list of items
            items = playlist_tracks['items']

        # Extract track names from the playlist response
        if items:
            if 'track' in items[0]:
                # If 'track' key is present, assume it's a track object
                track_names = [track['track']['name'] for track in items]
            else:
                # If 'track' key is not present, assume it's already a list of tracks
                track_names = [track['name'] for track in items]

            # Shuffle the track names
            random.shuffle(track_names)

            return track_names
        else:
            print("No items in the playlist response.")
            return None

    except KeyError as e:
        print(f"Error extracting track names: {e}")
        return None
    
def main():
    # Streamlit UI
    st.set_page_config(page_title="Emotion-Based Music Recommendation", page_icon=":musical_note:")

    st.title("Emotion-Based Music Recommendation")

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Set the resolution of the webcam capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Button for capturing image and detecting emotion
    if st.button("Capture Image and Detect Emotion"):
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the faces and detect emotions
        for (x, y, w, h) in faces:
            face_image = frame[y:y + h, x:x + w]
            emotion = detect_emotion(face_image)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display emotion text above the face rectangle
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame with increased clarity
        st.image(frame, caption="Emotion Detection", use_column_width=True, width=800)

        # Get music recommendation based on detected emotion
        recommendations = recommend_music(emotion)

        # Display results
        st.subheader("Detected Emotion:")
        st.write(emotion)
        st.subheader("Music Recommendations:")

        # Display links to Spotify tracks
        if recommendations:
            for track_name in recommendations:
                track_url = sp.search(track_name, type='track')['tracks']['items'][0]['external_urls']['spotify']
                st.markdown(f"Listen to [{track_name}]({track_url}) on Spotify")

    # Release the webcam and close all windows
    cap.release()

if __name__ == "__main__":
    main()