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


## Evaluación de los modelos

La vista **Métricas** permite elegir de uno a tres cortes positivos distintos
(por ejemplo, `3, 15, 50`). El resumen global muestra MAP, Precision y Recall. Los cortes se aplican al análisis
por consulta para Jaccard, TF-IDF, BM25 y Embeddings.

- `Precision@k`: documentos relevantes en los primeros k resultados, dividido
  para k, incluso cuando hay menos de k resultados.
- `Recall@k`: relevantes recuperados en los primeros k resultados, dividido
  para el total de relevantes según los qrels oficiales de FiQA.
- `AP`: promedio de las precisiones en las posiciones relevantes de todo el
  ranking, dividido para el total de relevantes. `MAP` promedia AP por consulta.
- `Precision` y `Recall` generales: usan todos los resultados recuperados por
  cada modelo y luego se promedian por consulta, incluyendo consultas sin aciertos.
  Los modelos léxicos recuperan coincidencias; embeddings devuelve todo el corpus
  al evaluar el ranking completo. Por eso sus precisiones generales deben leerse
  junto con las métricas @k.

Se conservan las posiciones relevantes de los rankings en memoria para cambiar
los cortes sin repetir las búsquedas. Importar otro corpus invalida la evaluación.
Los qrels de FiQA se cargan al importar `BeIR/fiqa` con la opción predeterminada;
un archivo personalizado no recibe automáticamente esos juicios de relevancia.

Embeddings usa el recuperador de Michael (`all-MiniLM-L6-v2` y FAISS), ya integrado
en búsqueda y comparación. En Métricas se prepara y evalúa en segundo plano,
con progreso y errores independientes: los otros modelos siguen disponibles.
La primera carga requiere las dependencias de `requirements.txt` y acceso al
modelo; las siguientes reutilizan la caché. Opcionalmente puede prepararse con:

```powershell
python dense_retrieval.py --build
```

Pruebas:

```powershell
python -m unittest discover -v
node tests/test_frontend.cjs
```

Las pruebas reales de FAISS y del codificador se omiten cuando esas dependencias
no están instaladas. Las pruebas de evaluación usan datos controlados sin descargas.


## Estructura del proyecto

- `server.py`: servidor Flask, búsqueda y evaluación.
- `dense_retrieval.py`: embeddings, índice FAISS y preparación de su caché.
- `front/`: página, estilos, JavaScript y logos utilizados por la interfaz.
- `tests/`: pruebas de los modelos, las métricas y el frontend.
- `requirements.txt`: dependencias de Python.

La caché de embeddings se genera en `.cache/` y no se incluye en Git.


## Análisis de los resultados

Métricas incluye MAP por tipo de pregunta y por longitud, con el tamaño de cada
categoría, los modelos con mayor MAP (incluidos empates) y los tipos donde cada
modelo alcanza su mayor MAP observado. Las categorías se asignan con reglas
aproximadas; la comparación es descriptiva y debe interpretarse con su tamaño.

La comparación semántica permite elegir BM25 o TF-IDF. Cuenta consultas donde
Embeddings mejora, empeora o empata por AP y muestra hasta cinco ejemplos por
dirección, ordenados por diferencia de AP, con el texto, el tipo de consulta y
la posición del primer documento relevante. Cada ejemplo abre su análisis por
consulta. Se usan los mismos qrels y el mismo conjunto de consultas de prueba;
los resultados semánticos se publican cuando termina la evaluación completa.
Estos análisis usan AP/MAP sobre el ranking completo y no cambian al elegir k.
