# Explicación de los resultados de la línea base

Resultados de `src/evaluate_masks.py`: SAM 2.1 (`hiera_small`) congelado, generador automático de máscaras
sin prompts humanos, evaluado contra los mapas de parcelas de Sen4AgriNet sobre 888 parches (148 por tesela,
6 teselas: 3 en España y 3 en Francia).

Configuración usada (ver `eval_baseline.log`): `points_per_side=16` (256 llamadas al decoder por imagen),
`pred_iou_thresh=0.5`, `stability_score_thresh=0.8`, dispositivo MPS.

**Concepto clave: IoU (Intersection over Union).** Mide el solape entre una máscara predicha A y una parcela real B:
`IoU = |A ∩ B| / |A ∪ B|`. Va de 0 (sin solape) a 1 (coincidencia exacta). Un IoU > 0.5 se suele considerar
"parcela encontrada".

---

## 1. `eval_patches.csv` (una fila por parche, 888 filas)

### Columnas de identificación

| Columna | Significado |
|---|---|
| `country` | País del parche: `ES` (España) o `FR` (Francia). |
| `tile` | Tesela Sentinel-2 a la que pertenece el parche (por ejemplo `31TBF`). |
| `patch` | Identificador del parche (por ejemplo `2019_31TBF_patch_28_08`). |
| `date` | Fecha de la adquisición de mayo usada como imagen de entrada. |

### Métricas

| Columna | Qué mide | Cómo se calcula | Cómo interpretarla |
|---|---|---|---|
| `n_masks` | Número de máscaras que el generador conserva. | Cantidad de máscaras que superan los filtros `pred_iou_thresh` y `stability_score_thresh` y la supresión de duplicados (NMS). | Pocas máscaras: el modelo propone poco. Muchas: segmentación más fragmentada. No es una medida de calidad por sí sola. |
| `coverage` | Fracción de la imagen cubierta por alguna máscara. | Píxeles que pertenecen a la unión de todas las máscaras, divididos por el total de píxeles del parche. | Valores cercanos a 1 significan que casi todo el parche está cubierto, pero no que los límites sean correctos. |
| `n_fields` | Número de parcelas reales evaluadas. | Parcelas del mapa de referencia con área ≥ 20 píxeles (`MIN_PARCEL_AREA`). El fondo (id 0) se excluye. | Es el denominador de las métricas por parcela. Parches con pocas parcelas dan métricas más ruidosas. |
| `mean_best_iou` | Calidad media con la que se encuentran las parcelas. | Para cada parcela se toma el IoU con la máscara que mejor la solapa (*best IoU*), y se promedian esos valores sobre las parcelas del parche. | Métrica principal. 1 = todas las parcelas segmentadas perfectamente; valores bajos = el modelo no encuentra las parcelas. |
| `median_best_iou` | Igual que la anterior, pero con la mediana. | Mediana del *best IoU* por parcela. | Menos sensible a unas pocas parcelas muy buenas o muy malas. Si es mucho menor que la media, la mayoría de las parcelas se encuentran mal y unas pocas muy bien. |
| `frac_iou_gt_0.5` | Proporción de parcelas "encontradas". | Fracción de parcelas cuyo *best IoU* es mayor que 0.5. | Interpretación práctica: ¿cuántas parcelas detecta el modelo de forma aceptable? |
| `small_mean_iou` | `mean_best_iou` de las parcelas pequeñas. | Media del *best IoU* en la mitad de parcelas más pequeñas del parche (corte en la mediana del área). Vacío si no se puede calcular. | Las parcelas pequeñas suelen dominar los efectos de borde y se segmentan peor. |
| `large_mean_iou` | `mean_best_iou` de las parcelas grandes. | Igual, para la mitad de parcelas más grandes. Vacío si no se puede calcular. | Comparar con `small_mean_iou` muestra si el fallo depende del tamaño de la parcela. |
| `mask_on_field_fraction` | Cuánto de lo que segmenta el modelo cae realmente sobre parcelas. | Para cada máscara: píxeles dentro de parcelas (`parcels != 0`) divididos por el área de la máscara; luego se promedia sobre las máscaras del parche. | 1 = todas las máscaras están sobre parcelas. Valores bajos = muchas máscaras cubren carreteras, bosque, edificios o terreno sin etiquetar. |
| `alignment` | Comprobación de que las etiquetas coinciden con la imagen. | Gradiente medio de la imagen sobre los límites de parcela dividido por el gradiente medio en el resto. | Valores > 1: los bordes de las parcelas coinciden con bordes visibles. Sirve para descartar que un IoU bajo se deba a un error de georreferenciación y no al modelo. |

---

## 2. `eval_summary.csv` (una fila por tesela y una fila `ALL`)

El resumen agrupa `eval_patches.csv` por tesela (`tile`) y añade una fila `ALL` con los 888 parches.
Cada valor es el **promedio de los valores por parche** (todos los parches pesan igual, tengan 5 o 400 parcelas).

| Columna | Significado | Se obtiene de |
|---|---|---|
| `tile` | Tesela (`31TBF`, `31TCG`, `31TDG` en España; `31TCL`, `31TDK`, `31UCP` en Francia) o `ALL` para el total. | Agrupación de `eval_patches.csv`. |
| `patches` | Número de parches del grupo (148 por tesela, 888 en `ALL`). | Conteo de filas. |
| `mean_n_masks` | Número medio de máscaras por parche. | Promedio de `n_masks`. |
| `mean_coverage` | Cobertura media de la imagen por las máscaras. | Promedio de `coverage`. |
| `mean_best_iou` | IoU medio de la mejor máscara por parcela. Métrica principal. | Promedio de `mean_best_iou`. |
| `mean_frac_iou_gt_0.5` | Proporción media de parcelas con IoU > 0.5. | Promedio de `frac_iou_gt_0.5`. |
| `mean_mask_on_field` | Fracción media de las máscaras que cae sobre parcelas. | Promedio de `mask_on_field_fraction`. |

### Resultados de la línea base

| tesela | país | `mean_best_iou` | `mean_frac_iou_gt_0.5` | `mean_mask_on_field` |
|---|---|---|---|---|
| 31TBF | ES | 0.068 | 0.048 | 0.445 |
| 31TCG | ES | 0.090 | 0.076 | 0.499 |
| 31TDG | ES | 0.105 | 0.083 | 0.302 |
| 31TCL | FR | 0.218 | 0.206 | 0.613 |
| 31TDK | FR | 0.166 | 0.143 | 0.741 |
| 31UCP | FR | 0.318 | 0.319 | 0.778 |
| **ALL** | | **0.161** | **0.146** | **0.563** |

Lectura:

- Solo el 14.6 % de las parcelas se encuentra con IoU > 0.5; el IoU medio es 0.161. La línea base es débil.
- Las teselas españolas rinden mucho peor que las francesas (IoU medio ≈ 0.09 frente a ≈ 0.23).
- Aproximadamente el 44 % de lo que cubren las máscaras no está etiquetado como parcela.
