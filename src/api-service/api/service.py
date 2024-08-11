from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
#import asyncio
#from api.tracker import TrackerService
#import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi import BackgroundTasks


from tempfile import TemporaryDirectory
from api import load_models
from api import model

#tracker_service = TrackerService()

object_processor = None
object_model = None

embedding_processor = None 
embedding_model = None

# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    print("Startup tasks")
    #asyncio.create_task(tracker_service.track())
    load_models.download_object_detection()
    load_models.download_embedding_model()

    global object_processor
    global object_model
    object_processor, object_model = load_models.load_object_model()

    global embedding_processor
    global embedding_model            
    embedding_processor, embedding_model = load_models.load_embedding_model()




# Routes
@app.get("/")
async def get_index():
    return {"message": "Welcome to the API Service"}

@app.post("/match")
async def match(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...), 
    dog_status: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...)):

    image_data = await image.read()

    global object_processor
    global object_model
    global embedding_processor
    global embedding_model 


    match_results, normalized_embedding, original_image, bounded_image = model.match_image(dog_status, image_data, object_processor, object_model,
                                                                                           embedding_processor, embedding_model, similarity_threshold = .7)

    background_tasks.add_task(model.model_background_tasks, dog_status, normalized_embedding, original_image, bounded_image, name, email, phone)

    return match_results

@app.get("/status")
async def get_api_status():
    return {
        "version": "2.2",
        "tf_version": tf.__version__,
    }