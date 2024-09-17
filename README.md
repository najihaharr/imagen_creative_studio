# Creative Studio | Vertex AI

Original code cloned from: https://github.com/GoogleCloudPlatform/vertex-ai-creative-studio/tree/main

This app is built with [Mesop](https://google.github.io/mesop), a Python-based UI framework that allows you to rapidly build web apps like this demo and internal apps.

# How to run locally

## 1. Create a virtual environment
```
sudo apt install python3-venv
```
```
python3 -m venv ~/myvirtualenv
```

Run it
```
source myvirtualenv/bin/activate
```

```
cd imagen_creative_studio
```

## 2. Install dependencies
```
pip install -r requirements.txt
```

## 3. Prepare your environment (.env) file
Open your main.py and have a look at the loadenv comments.
Create your own .env file and input the following values:
- Google Cloud project
- Google Cloud Storage bucket
- Flux endpoint value (create your own)

Currently, the code is saving generated pictures in both local storage and google cloud storage.

However, Imagen models and displaying pic from the cloud storage whereas Flux is displaying through local via base64 url encoder

## 3. Run main.py
```
mesop main.py
```
