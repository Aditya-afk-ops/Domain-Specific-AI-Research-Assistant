/domain-specific-ai-assistant
|
|-- data/
|   |-- my_resource.pdf  <-- (Your specific documents go here)
|
|-- main.py                     <-- (The FastAPI API)
|-- ingest.py                   <-- (Script to load data into Pinecone)
|-- requirements.txt            <-- (All the Python libraries)
|-- Dockerfile                  <-- (For containerizing the app)
|-- cloudbuild.yaml             <-- (For Google Cloud Run CI/CD)
|-- .env                        <-- (Your secret API keys)
|-- .gitignore                  <-- (To ignore .env, venv, etc.)


How to Run This Project

1. Prerequisites (One-time setup)
Install: Make sure you have Python 3.10+ and Docker installed.

OpenAI Key: Get your key from platform.openai.com.

Pinecone Key & Index:

Go to app.pinecone.io and get your API key.

Create a new Index.

Give it a name (e.g., research-assistant).

Set the Dimensions to 1536 (this is the required size for OpenAI's text-embedding-3-small model).

For Metric, select cosine.

Click "Create Index" and wait for it to initialize.

2. Fill in .env
Copy your API keys and the index name into the .env file you created.

3. Add Your Data
Find the .pdf or .txt research papers you want to use and place them inside the /data folder.
e.g - You can add a Medical Book pdf and make the chatbot work like a Medical Assistant.

4. Install Dependencies & Ingest Data
# Install all the libraries
pip install -r requirements.txt

# Run the ingestion script (This may take a few minutes)
python ingest.py

5. Run the API Locally
# This command starts your FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

(--reload makes the server auto-restart when you change the code.)

Open your browser to http://localhost:8080/docs. You will see the auto-generated FastAPI (Swagger) documentation, where you can test your API!

6. Deploy to Google Cloud Run
Make sure you have the gcloud CLI installed and configured.
Enable the Cloud Build, Cloud Run, and Artifact Registry APIs in your GCP project.

To securely handle API keys in deployment:
The best way is to use Secret Manager. Store your keys there and give your Cloud Run service permission to access them.
The quick way (for testing) is to pass them as build variables.

Run the build : 
# This command reads your cloudbuild.yaml and starts the CI/CD pipeline
gcloud builds submit --config cloudbuild.yaml .