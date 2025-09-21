# Validador de imagenes

## Correr la aplicacion

se necesita tener las siguientes env vars definidas
GEMINI_API_KEY
GOOGLE_GENAI_USE_VERTEXAI=False
GOOGLE_CLOUD_PROJECT
GOOGLE_CLOUD_LOCATION

y tener el archivo serviceAccount.json


```sh
docker compose up --build
```

la aplicacion corre en el puerto 8000
