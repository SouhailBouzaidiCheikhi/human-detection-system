import face_recognition
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle
import os

class FaceRecognitionSystem:
    def __init__(self, db, model="hog"):
        self.db = db
        self.model = model  # "hog" or "cnn"
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from database"""
        persons = self.db.get_all_persons()
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        for person in persons:
            encoding = person['face_encoding']
            if encoding:
                self.known_face_encodings.append(np.array(encoding))
                self.known_face_names.append(person['name'])
                self.known_face_ids.append(person['id'])
        
        print(f"Loaded {len(self.known_face_encodings)} known faces from database")
    
    def detect_faces(self, frame):
        """Detect faces in a frame using specified model"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame for faster processing if it's too large
        height, width = rgb_frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = 800
            new_height = int(height * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        
        # Scale locations back if we resized
        if width > 800:
            scale_factor = width / 800
            face_locations = [
                (int(top * scale_factor), int(right * scale_factor), 
                 int(bottom * scale_factor), int(left * scale_factor))
                for (top, right, bottom, left) in face_locations
            ]
        
        return face_locations
    
    def recognize_faces(self, frame) -> List[Dict]:
        """Recognize faces in a frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing if needed
        height, width = rgb_frame.shape[:2]
        original_size = (height, width)
        
        if width > 800:
            scale = 800 / width
            new_width = 800
            new_height = int(height * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
        
        # Find all faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back if we resized
            if width > 800:
                scale_factor = original_size[1] / 800
                top, right, bottom, left = [
                    int(coord * scale_factor) 
                    for coord in [top, right, bottom, left]
                ]
            
            name = "Unknown"
            person_id = None
            confidence = 0
            
            if self.known_face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding,
                    tolerance=0.6
                )
                
                if True in matches:
                    # Find the best match
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        person_id = self.known_face_ids[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
            
            recognized_faces.append({
                'location': (top, right, bottom, left),
                'name': name,
                'person_id': person_id,
                'confidence': confidence
            })
        
        return recognized_faces
    
    def register_new_face(self, frame, name: str, metadata: Dict = None) -> bool:
        """Register a new face from a frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for better performance
        height, width = rgb_frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = 800
            new_height = int(height * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        
        if not face_locations:
            print("No faces found for registration")
            return False
        
        print(f"Found {len(face_locations)} face(s) for registration")
        
        # Use the largest face found
        face_sizes = [(bottom - top) * (right - left) for (top, right, bottom, left) in face_locations]
        best_face_idx = np.argmax(face_sizes)
        
        try:
            face_encoding = face_recognition.face_encodings(
                rgb_frame, 
                [face_locations[best_face_idx]]
            )[0]
            
            # Add to database
            person_id = self.db.add_person(name, face_encoding.tolist(), metadata)
            
            # Update known faces
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            self.known_face_ids.append(person_id)
            
            print(f"Successfully registered {name}")
            return True
        except Exception as e:
            print(f"Error registering face: {e}")
            return False
    
    def draw_face_boxes(self, frame, recognized_faces: List[Dict]):
        """Draw bounding boxes and labels on frame"""
        for face in recognized_faces:
            top, right, bottom, left = face['location']
            name = face['name']
            confidence = face['confidence']
            
            # Draw box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            thickness = 2
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            
            # Draw label background
            label_height = 30
            cv2.rectangle(
                frame, 
                (left, bottom - label_height), 
                (right, bottom), 
                color, 
                cv2.FILLED
            )
            
            # Draw label text
            if confidence > 0:
                label = f"{name} ({confidence:.1%})"
            else:
                label = name
            
            font_scale = 0.6
            font_thickness = 1
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)[0]
            
            # Center text in label box
            text_x = left + (right - left - text_size[0]) // 2
            text_y = bottom - (label_height - text_size[1]) // 2
            
            cv2.putText(
                frame, 
                label, 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_DUPLEX, 
                font_scale, 
                (255, 255, 255), 
                font_thickness
            )
        
        return frame