# Proyecto RIIB en Railway (prueba gratuita)

La interfaz y el backend se despliegan juntos con el Dockerfile de la raíz.
No hacen falta Docker Desktop ni un túnel en el PC. Esta configuración usa
PyTorch para CPU, un único worker de Gunicorn, caché local y lotes de ocho
textos al generar embeddings. Los tres modelos léxicos comparten el mismo
índice de frecuencias para reducir la RAM sin cambiar las puntuaciones.

## Condiciones verificadas el 22 de noviembre de 2026

La prueba concede US$5 durante un máximo de 30 días y limita cada servicio a
1 GB de RAM. Termina al agotarse el crédito o el plazo. No activar un plan
pagado ni añadir recursos fuera de la prueba si se quiere mantener coste cero.
El plan Free posterior tiene otros límites y no garantiza que este proyecto
completo pueda seguir ejecutándose.

Conectar GitHub para la verificación de Full Trial. Limited Trial restringe
la red saliente y puede impedir descargar FiQA o el modelo de Hugging Face.

La medición local de Windows, con los 57 638 documentos y los cuatro modelos,
obtuvo aproximadamente 930 MiB tras buscar y un pico de 1 010 MiB antes del
ajuste de lectura de vectores mediante mmap. Son mediciones locales: no una
garantía del consumo de Linux. Revisar las métricas reales de Railway y los
logs durante la importación y la primera generación de embeddings. Si aparece
OOM o exit 137, no resolverlo comprando un plan automáticamente; revisar la RAM.

## 1. Subir los archivos preparados a main

Aplicar el parche de Railway desde la raíz del repositorio y revisar `git status`.
Los archivos esenciales son:

- `Dockerfile`, `.dockerignore`, `requirements-docker.txt`, `gunicorn.conf.py`.
- `railway.json`: builder Dockerfile y comprobación de salud `/healthz`.
- `server.py`, `dense_retrieval.py`, `evaluation_analysis.py`, `requirements.txt`.
- La carpeta `front/`.

Las pruebas, Compose y los documentos ayudan al desarrollo; la imagen no los
incluye. No subir `.venv`, `.cache`, claves ni tokens.

## 2. Crear el servicio desde GitHub

1. Entrar en https://railway.com/ con GitHub y confirmar que la cuenta muestra
   la prueba activa y crédito disponible.
2. Seleccionar **New Project → Deploy from GitHub repo**.
3. Elegir `Boris-epn/ProyectoRIIB` y la rama `main`.
4. Mantener la raíz del repositorio como **Root Directory**.
5. Railway debe detectar `railway.json` y construir el `Dockerfile`.
   No usar Railpack para instalar el requirements genérico: el Dockerfile
   instala explícitamente PyTorch CPU y Gunicorn.
6. Dejar vacío el Start Command personalizado: se usa el CMD del Dockerfile.
7. Mantener una réplica. El servidor comparte corpus e índices en memoria.
8. Esperar a que Build y Deploy terminen correctamente.

El servidor escucha `0.0.0.0:$PORT`. No hace falta fijar un puerto manualmente.
El healthcheck confirma disponibilidad HTTP, aunque el corpus aún no esté cargado.

## 3. Dominio y primera prueba

En **Settings → Networking → Public Networking**, seleccionar **Generate Domain**.
Abrir el dominio HTTPS generado por Railway. El navegador iniciará la importación
de FiQA. Comprobar Jaccard, TF-IDF, BM25 y Embeddings, comparación, métricas,
cortes por consulta y los análisis por grupos y diferencias semánticas.

La primera preparación de embeddings puede tardar. Revisar los logs y
`/api/corpus/status`. La evaluación semántica se calcula en segundo plano.
Un healthcheck verde por sí solo no demuestra que los cuatro modelos estén listos.

## 4. Persistencia opcional

El despliegue inicial funciona sin volumen, pero la caché se pierde al recrear
el contenedor. Si la prueba permite volumen y hay capacidad suficiente, se
puede montar en `/app/.cache`, donde están modelos, datasets y vectores.
Consultar antes la cuota disponible y el uso de almacenamiento, que consume
crédito. El tamaño de los vectores por sí solo ronda 85 MiB; el corpus y los
archivos del modelo ocupan espacio adicional.

Railway monta volúmenes como root. Para esta imagen no-root, su documentación
indica definir `RAILWAY_RUN_UID=0` si se usa ese volumen. No hace falta esta
variable para el primer despliegue sin volumen. El volumen no conserva los
índices Python en RAM: esos se reconstruyen al reiniciar el servidor.

## Si el despliegue falla

- **OOM / exit 137**: comprobar la RAM del servicio y la etapa donde ocurre.
- **No matching distribution**: revisar las versiones y el builder Dockerfile.
- **Descarga bloqueada**: comprobar Full Trial, red saliente y logs de Hugging Face.
- **502 / puerto**: comprobar que se ejecuta Gunicorn y respeta la variable PORT.
- **Permission denied en caché**: revisar si se montó un volumen root sobre la
  ruta de caché de una imagen que usa el usuario app.

## Referencias oficiales

- https://docs.railway.com/pricing/free-plan
- https://docs.railway.com/builds/dockerfiles
- https://docs.railway.com/config-as-code/reference
- https://docs.railway.com/networking/public-networking
- https://docs.railway.com/volumes
