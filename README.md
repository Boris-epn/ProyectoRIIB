# Proyecto RIIB

Front de recuperación de información sobre el corpus FiQA.

## Ejecutar

```powershell
python -m pip install -r requirements.txt
python server.py
```

Abrir `http://127.0.0.1:5000`.

El servidor importa automáticamente el corpus `BeIR/fiqa/corpus` desde el front.
También puede importar un archivo `.json`, `.jsonl` o `.txt` desde la vista **Métricas**.

Los archivos JSON deben contener una lista de documentos con `_id`, `title` y `text`.
