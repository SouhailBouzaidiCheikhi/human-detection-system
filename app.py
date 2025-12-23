import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime
import face_recognition
import tempfile
import os

# Import local modules
from database import PersonDatabase
from face_utils import FaceRecognitionSystem

# Page configuration
st.set_page_config(
    page_title="Human Detection & Recognition System",
    page_icon="👤",
    layout="wide"
)

# Initialize database and face recognition
@st.cache_resource
def init_systems():
    db = PersonDatabase()
    face_system = FaceRecognitionSystem(db)
    return db, face_system

db, face_system = init_systems()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "📷 Real-time Detection", "👥 Manage Database", "⚙️ Settings"]
)

# Dashboard Page
if page == "🏠 Dashboard":
    st.title("👤 Human Detection & Recognition System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_persons = len(db.get_all_persons())
        st.metric("Total Registered Persons", total_persons)
    
    with col2:
        stats = db.get_detection_stats(days=1)
        st.metric("Today's Detections", stats['total_detections'])
    
    with col3:
        st.metric("Unique Visitors Today", stats['unique_persons'])
    
    st.divider()
    
    # Recent detections
    st.subheader("Recent Detections")
    persons = db.get_all_persons()
    
    if persons:
        # Show recent persons in a simple list
        for person in persons[:5]:  # Show first 5
            with st.expander(f"👤 {person['name']}"):
                st.write(f"**Role:** {person['metadata'].get('role', 'N/A')}")
                if person['last_seen']:
                    last_seen = datetime.datetime.fromisoformat(person['last_seen'])
                    st.write(f"**Last seen:** {last_seen.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Total detections:** {person['total_detections']}")
    else:
        st.info("No persons registered yet. Go to 'Real-time Detection' to register new faces.")

# Real-time Detection Page
elif page == "📷 Real-time Detection":
    st.title("📷 Real-time Face Detection & Recognition")
    
    tab1, tab2 = st.tabs(["🎥 Live Camera", "📤 Upload & Register"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Camera input
            camera_image = st.camera_input(
                "Take a picture for detection/registration",
                help="The system will detect and recognize faces in real-time"
            )
            
            if camera_image:
                # Convert PIL Image to OpenCV format
                image = Image.open(camera_image)
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Recognize faces
                recognized_faces = face_system.recognize_faces(frame)
                
                # Draw boxes on frame
                annotated_frame = face_system.draw_face_boxes(frame.copy(), recognized_faces)
                
                # Convert back to RGB for display
                annotated_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(annotated_image, caption="Detection Results", use_column_width=True)
                
                # Process recognized faces
                for face in recognized_faces:
                    if face['name'] == "Unknown":
                        st.warning(f"Unknown face detected! Register this person.")
                        
                        # Simple registration form
                        name = st.text_input("Enter name to register", key=f"name_{face['location']}")
                        if st.button("Register", key=f"btn_{face['location']}"):
                            if name:
                                metadata = {"registered_via": "camera"}
                                if face_system.register_new_face(frame, name, metadata):
                                    st.success(f"✅ {name} registered successfully!")
                                    st.rerun()
                                else:
                                    st.error("Failed to register face. Please try again.")
                    else:
                        # Log known person detection
                        db.log_detection(face['person_id'], face['confidence'])
                        st.success(f"✅ Recognized: {face['name']} (Confidence: {face['confidence']:.2f})")
        
        with col2:
            st.subheader("Detection Info")
            st.info("👆 Point camera at a face and take a picture")
            st.info("✅ Green boxes: Recognized faces")
            st.info("❌ Red boxes: Unknown faces (register them!)")
    
    with tab2:
        st.subheader("📤 Upload Image for Registration")
        
        uploaded_file = st.file_uploader(
            "Choose an image file with a clear face",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Make sure the face is clearly visible and well-lit"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            try:
                # Load image
                image = cv2.imread(temp_path)
                
                if image is None:
                    st.error("Could not read the image file. Please try another format.")
                else:
                    # Display original image
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                            caption="Uploaded Image", 
                            width=300)
                    
                    # Check for faces using multiple methods
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Method 1: face_recognition library
                    face_locations = face_recognition.face_locations(rgb_image)
                    
                    if len(face_locations) > 0:
                        st.success(f"✅ Found {len(face_locations)} face(s) in the image!")
                        
                        # Draw boxes on detected faces
                        for (top, right, bottom, left) in face_locations:
                            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                        
                        # Show image with detected faces
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                caption="Detected Faces", 
                                width=300)
                        
                        # Registration form
                        st.subheader("Register Person")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            name = st.text_input("Full Name", placeholder="Enter person's name")
                        
                        with col2:
                            role = st.selectbox(
                                "Role",
                                ["Visitor", "Employee", "Student", "Guest", "Family", "Other"]
                            )
                        
                        additional_info = st.text_area("Additional Information (optional)")
                        
                        if st.button("Register This Person", type="primary"):
                            if name and name.strip():
                                metadata = {
                                    "role": role,
                                    "additional_info": additional_info,
                                    "registered_via": "upload",
                                    "registration_date": datetime.datetime.now().isoformat()
                                }
                                
                                # Try to register
                                if face_system.register_new_face(image, name.strip(), metadata):
                                    st.success(f"✅ **{name}** has been successfully registered!")
                                    st.balloons()
                                else:
                                    st.error("❌ Registration failed. The person might already exist or there was an error.")
                            else:
                                st.warning("Please enter a name for the person.")
                    else:
                        st.warning("⚠️ No faces detected in the uploaded image.")
                        st.info("""
                        **Tips for better face detection:**
                        1. Make sure the face is clearly visible
                        2. Good lighting is important
                        3. Face should be facing forward
                        4. Avoid sunglasses or heavy shadows
                        5. Try a different image
                        """)
                        
                        # Try alternative method with different model
                        st.write("Trying alternative detection method...")
                        
                        # Resize image for faster processing
                        small_image = cv2.resize(rgb_image, (0, 0), fx=0.5, fy=0.5)
                        face_locations = face_recognition.face_locations(small_image, model="cnn")
                        
                        if len(face_locations) > 0:
                            st.success(f"Found {len(face_locations)} face(s) using alternative method!")
                            
                            # Scale back up the face locations
                            for (top, right, bottom, left) in face_locations:
                                top, right, bottom, left = [coord * 2 for coord in [top, right, bottom, left]]
                                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                            
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                    caption="Detected Faces (Alternative Method)", 
                                    width=300)
                            
                            # Show registration form
                            name = st.text_input("Enter name to register")
                            if st.button("Register This Person"):
                                if name:
                                    metadata = {"registered_via": "upload_cnn"}
                                    if face_system.register_new_face(image, name, metadata):
                                        st.success(f"✅ {name} registered successfully!")
                                    else:
                                        st.error("Registration failed.")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

# Manage Database Page
elif page == "👥 Manage Database":
    st.title("👥 Manage Person Database")
    
    tab1, tab2, tab3 = st.tabs(["📋 View All", "🔍 Search", "✏️ Edit/Delete"])
    
    with tab1:
        st.subheader("Registered Persons")
        
        persons = db.get_all_persons()
        
        if persons:
            # Create a table-like display
            for idx, person in enumerate(persons):
                with st.expander(f"👤 {person['name']} (ID: {person['id']})", expanded=idx==0):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.metric("Detections", person['total_detections'])
                    
                    with col2:
                        st.write(f"**Role:** {person['metadata'].get('role', 'N/A')}")
                        if person['last_seen']:
                            last_seen = datetime.datetime.fromisoformat(person['last_seen'])
                            st.write(f"**Last Seen:** {last_seen.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        st.write(f"**Registered:** {person['registration_date'][:10]}")
                        
                        if 'additional_info' in person['metadata']:
                            st.write(f"**Info:** {person['metadata']['additional_info']}")
        else:
            st.info("No persons registered yet.")
    
    with tab2:
        st.subheader("Search Persons")
        
        search_query = st.text_input("Search by name", placeholder="Enter name to search...")
        
        if search_query:
            results = db.search_persons(search_query)
            
            if results:
                st.success(f"Found {len(results)} matching person(s)")
                for person in results:
                    with st.expander(f"👤 {person['name']}"):
                        st.write(f"**ID:** {person['id']}")
                        st.write(f"**Role:** {person['metadata'].get('role', 'N/A')}")
                        st.write(f"**Detections:** {person['total_detections']}")
            else:
                st.warning("No matching persons found.")
    
    with tab3:
        st.subheader("Edit or Delete Persons")
        
        persons = db.get_all_persons()
        
        if persons:
            # Create selectbox for person selection
            person_names = [f"{p['id']}: {p['name']}" for p in persons]
            selected_person_str = st.selectbox("Select a person to edit", person_names)
            
            if selected_person_str:
                selected_id = int(selected_person_str.split(":")[0])
                person = db.get_person_by_id(selected_id)
                
                if person:
                    st.write(f"Editing: **{person['name']}**")
                    
                    new_name = st.text_input("Name", value=person['name'])
                    
                    current_role = person['metadata'].get('role', 'Visitor')
                    new_role = st.selectbox(
                        "Role",
                        ["Visitor", "Employee", "Student", "Guest", "Family", "Other"],
                        index=["Visitor", "Employee", "Student", "Guest", "Family", "Other"].index(current_role) 
                        if current_role in ["Visitor", "Employee", "Student", "Guest", "Family", "Other"] else 0
                    )
                    
                    current_info = person['metadata'].get('additional_info', '')
                    new_info = st.text_area("Additional Information", value=current_info)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("💾 Update Person", use_container_width=True):
                            metadata = person['metadata'].copy()
                            metadata.update({
                                'role': new_role,
                                'additional_info': new_info
                            })
                            
                            db.update_person(person['id'], new_name, metadata)
                            face_system.load_known_faces()
                            st.success(f"✅ {person['name']} updated successfully!")
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️ Delete Person", type="secondary", use_container_width=True):
                            st.warning(f"Are you sure you want to delete {person['name']}?")
                            if st.button("Yes, Delete Permanently", type="primary"):
                                db.delete_person(person['id'])
                                face_system.load_known_faces()
                                st.success(f"✅ {person['name']} deleted successfully!")
                                st.rerun()
        else:
            st.info("No persons to edit.")

# Settings Page
elif page == "⚙️ Settings":
    st.title("⚙️ System Settings")
    
    st.subheader("Face Detection Settings")
    
    # Create a session state for settings if not exists
    if 'detection_model' not in st.session_state:
        st.session_state.detection_model = "hog"  # or "cnn"
    
    detection_model = st.selectbox(
        "Face Detection Model",
        ["hog (faster, less accurate)", "cnn (slower, more accurate)"],
        index=0 if st.session_state.detection_model == "hog" else 1
    )
    
    if detection_model.startswith("hog"):
        st.session_state.detection_model = "hog"
    else:
        st.session_state.detection_model = "cnn"
    
    st.info(f"Current model: **{st.session_state.detection_model.upper()}**")
    st.write("""
    - **HOG**: Faster, works on CPU, good for real-time
    - **CNN**: More accurate, requires more processing power
    """)
    
    st.divider()
    
    st.subheader("Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Face Database", use_container_width=True):
            face_system.load_known_faces()
            st.success("Face database refreshed!")
    
    with col2:
        if st.button("📊 View Database Stats", use_container_width=True):
            persons = db.get_all_persons()
            if persons:
                st.write(f"**Total persons:** {len(persons)}")
                total_detections = sum(p['total_detections'] for p in persons)
                st.write(f"**Total detections:** {total_detections}")
                
                # Show most frequently detected
                sorted_persons = sorted(persons, key=lambda x: x['total_detections'], reverse=True)
                st.write("**Most frequently detected:**")
                for person in sorted_persons[:3]:
                    st.write(f"- {person['name']}: {person['total_detections']} detections")
            else:
                st.info("Database is empty.")
    
    st.divider()
    
    st.subheader("System Information")
    
    persons = db.get_all_persons()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Registered Persons", len(persons))
    
    with col2:
        total_detections = sum(p['total_detections'] for p in persons)
        st.metric("Total Detections", total_detections)
    
    with col3:
        if persons:
            # Find person with most recent detection
            persons_with_detections = [p for p in persons if p['last_seen']]
            if persons_with_detections:
                latest = max(persons_with_detections, 
                           key=lambda x: datetime.datetime.fromisoformat(x['last_seen']))
                st.metric("Last Detected", latest['name'])

# Footer
st.sidebar.divider()
st.sidebar.markdown("### Quick Stats")
persons = db.get_all_persons()
st.sidebar.write(f"👤 **Registered:** {len(persons)}")
total_detections = sum(p['total_detections'] for p in persons)
st.sidebar.write(f"📊 **Total Detections:** {total_detections}")
st.sidebar.divider()
st.sidebar.caption("System Status: 🟢 Online")