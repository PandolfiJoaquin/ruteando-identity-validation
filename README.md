<img width="1243" height="230" alt="image" src="https://github.com/user-attachments/assets/f87393dc-9b64-4338-b47b-8a125fcbc636" /># Validador de imagenes

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
