
# Taller 02 — Transfer Learning en Diagnóstico Dermatológico

**Curso:** Aprendizaje Profundo · 32310019 · 2026-01
**Dataset:** DermaMNIST (7 clases, 28×28 RGB)
**Clase crítica:** `mel` (Melanoma) — prioridad en recall
**Framework:** TensorFlow / Keras
**Reproducibilidad:** `SEED = 161105`

---

## Resultados

> Los siguientes valores se completan automáticamente al ejecutar `main.ipynb`
> con `FAST_MODE = False`. La tabla canónica vive en
> `artifacts_taller_02/comparison_results.csv`.


| Modelo                        |   accuracy |   balanced_acc |   f1_macro |   mel_recall |       parametros |
|:------------------------------|-----------:|---------------:|-----------:|-------------:|-----------------:|
| Baseline CNN (5×5)            |     0.5102 |         0.5164 |     0.3695 |       0.6323 |  27783           |
| MobileNetV2 (FE)              |     0.5761 |         0.5094 |     0.3683 |       0.5291 |      2.26695e+06 |
| MobileNetV2 (FT)              |     0.6958 |         0.5334 |     0.5199 |       0.5605 |      2.26695e+06 |
| MobileNetV2 + embeddings + RF |     0.6943 |         0.2348 |     0.2727 |       0.0135 |    nan           |
| DeiT / Small ViT              |     0.4464 |         0.1917 |     0.11   |       0      | 411719           |

**Métricas mínimas esperadas (`FAST_MODE=False`):**

| Modelo            | F1 Macro objetivo | Mel Recall objetivo |
|-------------------|-------------------|---------------------|
| Baseline CNN      | > 0.45            | > 0.40              |
| MobileNetV2 FE    | > 0.55            | > 0.50              |
| MobileNetV2 FT    | > 0.60            | > 0.55              |
| Small ViT / DeiT  | > 0.50            | > 0.45              |

---

## Estructura del repositorio

```
Taller_02/
├── main.ipynb                  # Entregable narrativo (importa src/)
├── predict.py                  # CLI de inferencia standalone
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── config.py               # SEED, FAST_MODE, hiperparámetros, rutas
│   ├── data_loader.py          # Carga DermaMNIST + subset estratificado
│   ├── augmentation.py         # Pipelines RandomFlip/Rotation/Zoom
│   ├── preprocessing.py        # tf.data por familia + interpolación
│   ├── training.py             # class_weights, callbacks, timed_fit()
│   ├── evaluation.py           # métricas, latencia, tabla comparativa
│   ├── visualization.py        # curvas, confusión, scatter, FFT
│   └── models/
│       ├── baseline_cnn.py     # CNN aproximación Caso 01 (5×5)
│       ├── mobilenetv2.py      # FE + FT + snapshot + embedding extractor
│       ├── vit_fallback.py     # DeiT preset list + Small ViT fallback
│       └── custom_layers.py    # Patches + PatchEncoder con get_config()
│
├── artifacts_taller_02/        # generado al ejecutar (gitignored)
│   ├── best_model.keras
│   ├── best_model_meta.json
│   ├── rf_pipeline.joblib
│   └── comparison_results.csv
│
└── figures/                    # generado al ejecutar (gitignored)
    ├── interp_comparison.png
    ├── class_distribution.png
    ├── learning_curves_*.png
    ├── confusion_matrices.png
    └── tradeoff_scatter.png
```

---

## Instalación

### Local (Linux / macOS)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Local (Windows)

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### Google Colab

Subir el ZIP y ejecutar en una celda:

```python
import zipfile, os
with zipfile.ZipFile("Taller_02.zip") as z:
    z.extractall(".")
os.chdir("Taller_02")
!pip install -q -r requirements.txt
```

Luego reiniciar el runtime antes de ejecutar `main.ipynb`.

### Versiones probadas

- Python 3.10 – 3.12
- TensorFlow 2.15 – 2.17 (NumPy < 2.0 es importante)
- Keras 3.x
- `keras-hub >= 0.3` (opcional; si no está, ViT cae automáticamente a Small ViT desde cero)

---

## Ejecución

### Opción 1 — Notebook completo

```bash
jupyter lab main.ipynb
```

Modos:
- `FAST_MODE = True` (default): subset estratificado, < 15 min en CPU.
- `FAST_MODE = False`: dataset completo, requerido para la entrega final.

Se cambia editando `src/config.py`.

### Opción 2 — Inferencia sobre una imagen

```bash
python predict.py --image ruta/a/lesion.jpg
python predict.py --image lesion.png --top-k 5
python predict.py --image lesion.png --json
```

Salida típica:

```
Imagen:        lesion.png
Modelo:        mobilenetv2_fine_tuned (familia=mobilenet, input=224px, interp=bilinear)

Rank  Clase     Probabilidad
--------------------------------
1     mel             0.6832  <-- MEL
2     nv              0.2104
3     bkl             0.0541
```

`predict.py` requiere que existan en `artifacts_taller_02/`:
- `best_model.keras`
- `best_model_meta.json`

Ambos se generan en la Sección 14 del notebook.

---

## Reproducibilidad

- Semilla global única: `SEED = 161105` (declarada en `src/config.py` y propagada a NumPy, TensorFlow, `StratifiedShuffleSplit` y todas las capas `Random*` de augmentation).
- Subsetting `FAST_MODE` usa `StratifiedShuffleSplit` con la misma semilla → mismas muestras entre ejecuciones.
- `tf.data` se baraja con `seed=SEED` y `reshuffle_each_iteration=True`.
- Las pequeñas diferencias entre corridas en GPU son inevitables (operaciones cuDNN no deterministas), pero las métricas reportadas se mantienen dentro de ±1%.

---

## Decisiones técnicas relevantes

1. **Baseline con kernel 5×5 y GAP** — replica la arquitectura del Caso 01 (Taller 01) para que la comparación contra MobileNetV2 sea metodológicamente válida.
2. **Fine-tuning conservador** — solo se descongelan las últimas 30 capas de MobileNetV2 con `lr=1e-5`. Descongelar la base completa con un dataset de este tamaño tiende a destruir las representaciones de ImageNet.
3. **Snapshot del modelo FE antes del FT** — permite reportar FE y FT como entidades separadas en la tabla comparativa sin re-entrenar.
4. **Fallback automático para ViT** — `try_load_deit()` itera una lista de presets de `keras_hub`; si todos fallan, instancia un Small ViT desde cero (patch=16, projection=64, 4 transformer layers).
5. **Custom layers serializables** — `Patches` y `PatchEncoder` implementan `get_config()` y están registradas con `@keras.saving.register_keras_serializable`, permitiendo `keras.save()` / `load_model()` sin pasar `custom_objects` adicional.
6. **Class weights balanceados** — `compute_class_weight("balanced", ...)` durante `.fit()` mitiga el sesgo hacia la clase mayoritaria (`nv`) y mejora `mel_recall`.
7. **Augmentation por resolución** — pipelines distintos para 28×28 (solo flip + rotación leve) y 224×224 (flip + rotación + zoom), evitando artefactos por sobreaumentar imágenes ya degradadas.

---

## Limitaciones conocidas

- La comparación contra el Caso 01 usa la CNN baseline como _proxy_ del modelo `dermClass.keras` original (no incluido en este repo).
- `keras-hub` puede no exponer presets DeiT en todas las versiones; el fallback a Small ViT garantiza que la sección T6 corra siempre, pero un Small ViT desde cero no es comparable en condiciones a un DeiT pre-entrenado en ImageNet.
- DermaMNIST está derivado de HAM10000 mediante downsampling agresivo a 28×28. El upsampling a 224×224 (bilinear o bicubic) no recupera la información perdida; el ejercicio mide cuánto puede compensar transfer learning este _domain shift_.

