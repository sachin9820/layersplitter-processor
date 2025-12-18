import firebase_admin
from firebase_admin import credentials, firestore
import requests
import os
import json
import sys
from datetime import datetime
import base64

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Setup logging
log_file = f"logs/process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
def log_message(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(log_file, 'a') as f:
        f.write(full_msg + '\n')

log_message("=" * 60)
log_message("STARTING IMAGE PROCESSING")
log_message("=" * 60)

try:
    # Initialize Firebase
    log_message("Initializing Firebase...")
    firebase_creds_str = os.getenv('FIREBASE_CREDENTIALS')
    
    if not firebase_creds_str:
        raise Exception("FIREBASE_CREDENTIALS not found in environment")
    
    try:
        creds_dict = json.loads(firebase_creds_str)
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid Firebase credentials JSON: {str(e)}")
    
    creds = credentials.Certificate(creds_dict)
    firebase_admin.initialize_app(creds)
    db = firestore.client()
    log_message("✓ Firebase initialized")

    # Get environment variables
    HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    B2_KEY_ID = os.getenv('BACKBLAZE_KEY_ID')
    B2_APP_KEY = os.getenv('BACKBLAZE_APP_KEY')
    B2_BUCKET_ID = os.getenv('BACKBLAZE_BUCKET_ID')

    if not all([HF_API_KEY, B2_KEY_ID, B2_APP_KEY, B2_BUCKET_ID]):
        raise Exception("Missing required environment variables")

    log_message("✓ Environment variables loaded")

    # Get processing projects
    log_message("Fetching projects with status='processing'...")
    projects = []
    try:
        docs = db.collection('projects').where('status', '==', 'processing').stream()
        projects = [doc.to_dict() for doc in docs]
        log_message(f"✓ Found {len(projects)} projects to process")
    except Exception as e:
        log_message(f"✗ Error fetching projects: {str(e)}")
        sys.exit(1)

    if not projects:
        log_message("No projects to process. Exiting.")
        sys.exit(0)

    def get_b2_auth():
        """Get Backblaze authorization"""
        try:
            auth_string = base64.b64encode(
                f"{B2_KEY_ID}:{B2_APP_KEY}".encode()
            ).decode()
            
            response = requests.get(
                "https://api.backblazeb2.com/b2api/v3/b2_authorize_account",
                headers={"Authorization": f"Basic {auth_string}"},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log_message(f"✗ B2 auth error: {str(e)}")
            return None

    def remove_background(image_url):
        """Use Hugging Face to remove background"""
        try:
            log_message(f"  → Calling Hugging Face for background removal...")
            response = requests.post(
                "https://api-inference.huggingface.co/models/briaai/BRIA-2.0-RMBG-ViT-B-1024",
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                files={"data": ("image.png", requests.get(image_url, timeout=30).content)},
                timeout=30
            )
            response.raise_for_status()
            log_message(f"  ✓ Background removed")
            return response.content
        except Exception as e:
            log_message(f"  ✗ Background removal error: {str(e)}")
            return None

    def detect_objects(image_url):
        """Detect objects in image using Hugging Face"""
        try:
            log_message(f"  → Calling Hugging Face for object detection...")
            response = requests.post(
                "https://api-inference.huggingface.co/models/facebook/detr-resnet-50",
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                files={"data": ("image.png", requests.get(image_url, timeout=30).content)},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            log_message(f"  ✓ Detected objects: {len(data) if isinstance(data, list) else 'N/A'}")
            return data
        except Exception as e:
            log_message(f"  ✗ Object detection error: {str(e)}")
            return None

    def create_layers(objects):
        """Create individual layers from detected objects"""
        layers = []
        
        # Background layer
        layers.append({
            "name": "Background",
            "type": "background"
        })
        
        # Create layer for each detected object
        if objects and isinstance(objects, list) and len(objects) > 0:
            for i, obj in enumerate(objects[:10]):  # Max 10 layers
                label = obj.get('label', 'Object') if isinstance(obj, dict) else 'Object'
                score = obj.get('score', 0) if isinstance(obj, dict) else 0
                
                layers.append({
                    "name": f"Layer {i+1} - {label}",
                    "type": "object",
                    "confidence": score,
                })
        
        log_message(f"  ✓ Created {len(layers)} layers")
        return layers

    def upload_to_backblaze(file_content, file_name):
        """Upload processed file to Backblaze"""
        try:
            # Get auth
            auth_data = get_b2_auth()
            if not auth_data:
                raise Exception("Failed to authorize with Backblaze")
            
            auth_token = auth_data['authorizationToken']
            api_url = auth_data['apiUrl']
            
            # Get upload URL
            log_message(f"  → Getting Backblaze upload URL...")
            upload_url_response = requests.post(
                f"{api_url}/b2api/v3/b2_get_upload_url",
                headers={"Authorization": auth_token},
                json={"bucketId": B2_BUCKET_ID},
                timeout=10
            )
            upload_url_response.raise_for_status()
            upload_url = upload_url_response.json()['uploadUrl']
            
            # Upload file
            log_message(f"  → Uploading to Backblaze...")
            upload_response = requests.put(
                upload_url,
                headers={
                    "Authorization": auth_token,
                    "X-Bz-File-Name": file_name,
                    "Content-Type": "application/json"
                },
                data=file_content,
                timeout=30
            )
            upload_response.raise_for_status()
            
            result = upload_response.json()
            download_url = f"https://f000.backblazeb2.com/file/{B2_BUCKET_ID}/{file_name}"
            log_message(f"  ✓ Uploaded successfully")
            
            return download_url
        except Exception as e:
            log_message(f"  ✗ Upload error: {str(e)}")
            return None

    def update_firestore(project_id, status, layer_urls=None, error=None):
        """Update project status in Firestore"""
        try:
            data = {
                'status': status,
                'updatedAt': firestore.SERVER_TIMESTAMP
            }
            
            if layer_urls:
                data['layerUrls'] = layer_urls
                data['layersCount'] = len(layer_urls)
                data['completedAt'] = firestore.SERVER_TIMESTAMP
            
            if error:
                data['error'] = error
            
            db.collection('projects').document(project_id).update(data)
            log_message(f"  ✓ Firestore updated: status={status}")
        except Exception as e:
            log_message(f"  ✗ Firestore update error: {str(e)}")

    # Process each project
    for i, project in enumerate(projects, 1):
        project_id = project.get('projectId', 'unknown')
        image_url = project.get('originalImageUrl', '')
        
        log_message("")
        log_message(f"[{i}/{len(projects)}] Processing: {project_id}")
        log_message("-" * 60)
        
        try:
            if not image_url:
                raise Exception("No image URL found")
            
            # Step 1: Remove background
            log_message("Step 1: Removing background...")
            bg_removed = remove_background(image_url)
            if not bg_removed:
                raise Exception("Background removal failed")
            
            # Step 2: Detect objects
            log_message("Step 2: Detecting objects...")
            objects = detect_objects(image_url)
            
            # Step 3: Create layers
            log_message("Step 3: Creating layers...")
            layers = create_layers(objects)
            
            # Step 4: Create layer file
            log_message("Step 4: Creating layer file...")
            layer_file = json.dumps({
                "format": project.get('format', 'tiff'),
                "layersCount": len(layers),
                "layers": layers,
                "processedAt": datetime.now().isoformat()
            }).encode()
            
            # Step 5: Upload to Backblaze
            log_message("Step 5: Uploading to Backblaze...")
            b2_url = upload_to_backblaze(
                layer_file,
                f"layers/{project.get('userId', 'unknown')}/{project_id}/layers.json"
            )
            
            if not b2_url:
                raise Exception("Upload to Backblaze failed")
            
            layer_urls = [b2_url]
            
            # Step 6: Update Firestore
            log_message("Step 6: Updating database...")
            update_firestore(
                project_id,
                status='completed',
                layer_urls=layer_urls
            )
            
            log_message(f"✓ COMPLETED: {project_id}")
            
        except Exception as e:
            log_message(f"✗ ERROR: {str(e)}")
            update_firestore(project_id, status='failed', error=str(e))

    log_message("")
    log_message("=" * 60)
    log_message("PROCESSING COMPLETE")
    log_message("=" * 60)

except Exception as e:
    log_message(f"FATAL ERROR: {str(e)}")
    sys.exit(1)
