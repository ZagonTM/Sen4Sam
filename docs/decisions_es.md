# Registro de decisiones — SAM 2 × Sen4AgriNet

Cada decisión de diseño, las opciones que sopesamos y la evidencia que respalda la elección.
Las cifras medidas provienen de `notebooks/00_es_data_availability.ipynb` (datos) y
`src/evaluate_masks.py` (línea base). Cada entrada termina con su advertencia, para que los
puntos débiles también queden registrados.

## El árbol de decisiones de un vistazo

```
Segmentar campos agrícolas individuales en imágenes Sentinel-2 con SAM 2, sin prompts humanos
│
├── 1. ALCANCE DE LOS DATOS
│   ├─ ¿qué conjunto?        → Sen4AgriNet PAD        (necesita etiquetas a nivel de píxel)
│   ├─ ¿qué año?             → 2019                   (único año con ES *y* FR)
│   ├─ ¿qué teselas?         → 3 por país, AL AZAR    (semilla 42)
│   ├─ ¿qué parches?         → 148 por tesela, al azar (semilla 42), sin filtro de cobertura
│   ├─ ¿qué mes?             → MAYO                   (mes más verde que cubre (casi) todos los parches)
│   ├─ ¿una fecha o mediana? → UNA adquisición real   (no un compuesto)
│   ├─ ¿regla de nubes?      → ≤5% (Hollstein), tomar la menos nubosa
│   └─ ¿imagen de entrada?   → B04/B03/B02, ÷10000, estirado 2–98% → uint8
│
├── 2. MODELO Y TAREA
│   ├─ ¿qué modelo?          → SAM 2.1 hiera_small, CONGELADO
│   ├─ ¿con prompts?         → NO — generador automático sin prompts (la tarea)
│   └─ ¿punto de operación?  → points_per_side 16, pred_iou 0.5, stability 0.8
│
├── 3. EVALUACIÓN
│   ├─ ¿métrica?             → por instancia: mejor IoU por campo real
│   ├─ ¿dónde se mide?       → solo píxeles de campo (parcels ≠ 0)
│   └─ ¿controles?           → estratificar por tamaño de campo + control de alineación GT
│
├── 4. FASE B (planificada)
│   └─ ¿intervención?        → ?
│
└── 5. FASE C
    └─ ¿cambio de dominio?   → SA-1B → Sen4AgriNet
```

## 1. Alcance de los datos

### 1.1 Qué conjunto → Sen4AgriNet, Patches Assembled Dataset (PAD)
- **Opciones:** Sen4AgriNet vs teselas genéricas de Sentinel-2 vs benchmarks sintéticos.
- **Decisión:** PAD. Es el único candidato que incluye verdad de referencia **a nivel de
  píxel** (`parcels`, `labels`) alineada con series temporales de Sentinel-2 — la segmentación
  necesita etiquetas espaciales.
- **Advertencia:** el otro subconjunto (OAD) son estadísticas por parcela, inservibles aquí.

### 1.2 Qué año → 2019
- **Opciones:** 2019; 2020; ambos.
- **Decisión:** 2019. 2020 contiene **solo Cataluña**, lo que eliminaría la comparación entre
  países.
- **Advertencia:** un solo año limita la diversidad estacional / de campaña agrícola.

### 1.3 Qué teselas → 3 por país, elegidas al azar (semilla 42)
- **Opciones:** (a) ordenar las teselas por número de campos/adquisiciones y quedarse con las
  mejores; (b) elegir al azar.
- **Decisión:** (b) al azar — ES {31TBF, 31TDG, 31TCG}, FR {31TCL, 31TDK, 31UCP}. Elegir las
  teselas que mejor aspecto tienen sesgaría los resultados a favor del modelo.
- **Evidencia:** la selección aleatoria es la opción honesta; el ordenamiento se descartó
  explícitamente. Como las teselas son aleatorias, no hizo falta un perfilado exhaustivo de
  cada una.
- **Advertencia:** el sorteo nos dio **31TDG, en gran parte no agrícola** — cobertura de campo
  mediana 0.061 frente a 0.73 en 31UCP. La mantenemos (eso es lo que hace un sorteo aleatorio)
  y lo reportamos en lugar de cambiarla en silencio.

### 1.4 Qué parches → 148 por tesela, al azar, sin filtro de cobertura de campo
- **Opciones:** (a) quedarse solo con parches con suficientes campos; (b) muestra aleatoria,
  conservar todos.
- **Decisión:** (b). La cobertura de campo se **registra y se reporta como distribución, nunca
  se usa para excluir parches**.
- **Evidencia:** la cobertura va de p10 0.018 / mediana 0.409 / p90 0.791 — una mezcla del
  mundo real. Filtrar haría las métricas optimistas y eliminaría negativos útiles ("aquí no hay
  campo").
- **Advertencia:** el leve filtro de nubes/fecha todavía sesga ligeramente la muestra hacia
  localizaciones más despejadas.
- **Nota:** se muestrean 150 candidatos por tesela, pero solo se conservan **148** — la ventana
  de mayo deja 2 parches (ambos `31TCL`) sin cubrir, así que cada tesela se recorta a los
  mismos 148 por equilibrio.

### 1.5 Qué mes → Mayo
- **Opciones:** abril; mayo; junio; julio; una ventana más amplia (mayo–agosto); una mediana
  mensual.
- **Decisión:** **Mayo** — el mes más verde que aún cubre (casi) todos los parches.
- **Evidencia** (`00_es_data_availability.ipynb`, §1 y §2): los meses que dan (casi) a todos
  los parches una fecha limpia son febrero, mayo, julio, agosto y septiembre (>98 %, notebook
  §1). Entre ellos **mayo** es el más verde — el NDVI de campo alcanza su máximo en mayo (0.58)
  y vuelve a la línea base invernal en julio (0.38) (notebook §2):

  | mes | parches con ≥1 fecha limpia | NDVI de campo medio |
  |---|---|---|
  | Febrero | 99.8% | 0.39 |
  | **Mayo** | **99.8%** | **0.58** |
  | Julio | 100% | 0.38 |
  | Agosto | 99.9% | 0.37 |
  | Septiembre | 98.4% | 0.36 |

  (Para comparar, junio solo cubre el 75 % de los parches y abril el 61 %.) Mayo tiene 1,934
  adquisiciones en total, el 82 % de ellas *exactamente* libres de nubes. Un único mes también
  mantiene la imaginería fenológicamente consistente entre parches, algo que una ventana de
  varios meses no puede.
- **Advertencia:** una mediana por píxel de mayo cubriría solo el 31 % de los parches (menos
  fechas limpias por parche que julio), y mayo es algo más nuboso que julio (96 % frente a
  98 % de adquisiciones con ≤5 % de nubes).

### 1.6 Una fecha o una mediana → una adquisición real
- **Opciones:** (a) mediana mensual; (b) una adquisición (la menos nubosa, al azar entre
  empates).
- **Decisión:** (b). SAM 2 consume una **sola imagen**; una mediana solo reemplazaría una
  observación limpia por una sintética.
- **Nota:** el propio filtro <10 % de Sen4AgriNet es a nivel de escena, y sus modelos de series
  temporales pueden compensar una fecha nubosa con las demás; un modelo de una sola imagen no
  tiene ninguna de las dos cosas, así que cada imagen debe estar limpia
  (`00_es_data_availability.ipynb`, §1).
- **Evidencia:** las nubes son **despreciables** — mediana de fracción de nubes 0.000, solo el
  4.6 % de las adquisiciones supera el 5 %. Una mediana es una herramienta de rechazo de nubes
  que simplemente no necesitamos. Una mediana por píxel entre fechas también mezcla fenología
  (distintos píxeles muestrean distintas fechas) y no es la convención del propio conjunto (la
  suya es una mediana *dentro* de un mes).
- **Advertencia:** los parches no provienen todos del mismo día (las fechas de mayo van del 1
  al 31).

### 1.7 Regla de nubes → Hollstein ≤5%, tomar la menos nubosa
- **Opciones:** ≤5%; ≤1%; exactamente 0.
- **Decisión:** ≤5%, usando la propia fórmula Hollstein del conjunto, y después tomar la
  adquisición menos nubosa (al azar entre empates).
- **Evidencia:** exigir *exactamente* 0 dejaría solo 848/900 parches cubiertos (94 %), y un
  único píxel falso positivo (el borde de un tejado, suelo desnudo) convierte una imagen
  visualmente limpia en "0.2 % de nubes". El 82 % de las adquisiciones de mayo son exactamente
  0 de todos modos.
- **Advertencia:** el filtro favorece levemente los sitios más despejados.

### 1.8 Imagen de entrada → B04/B03/B02, ÷10000, estirado de percentiles 2–98%
- **Opciones:** color real vs color falso (NIR) vs otros conjuntos de bandas.
- **Decisión:** color real `B04/B03/B02` (las bandas nativas de 10 m), divididas entre 10000
  (reflectancia L1C), estiradas por banda según su percentil 2.º–98.º a uint8.
- **Razonamiento:** SAM 2 espera RGB de 3 canales. Descartar las otras 10 bandas *es* parte de
  la historia del cambio de dominio. El estirado importa porque la reflectancia en bruto es
  oscura y de bajo contraste, y SAM 2 se entrenó con imágenes naturales.
- **Advertencia:** sin NIR perdemos información que ayudaría a la vegetación — es deliberado, y
  es justo el tipo de coste que examina el proyecto.

## 2. Modelo y tarea

### 2.1 Qué modelo → SAM 2.1 `hiera_small`, congelado
- **Opciones:** SAM 2 `tiny` / `small` / `base+` / `large`.
- **Decisión:** `hiera_small` (46.06 M de parámetros) como modelo principal; `tiny` (38.96 M)
  para iterar rápido. Ambos pesos son locales.
- **Evidencia:** los cuatro tamaños difieren **solo** en `image_encoder.trunk` — el
  decodificador de máscaras, el codificador de prompts y los módulos de memoria son idénticos —
  así que el análisis del decodificador es invariante al tamaño.
- **Advertencia:** el modelo congelado es la línea base, no uno adaptado; la adaptación es la
  Fase B.

### 2.2 ¿Con prompts o sin prompts? → sin prompts (generador automático)
- **Opciones:** prompts humanos/de puntos; el generador automático de máscaras.
- **Decisión:** sin prompts. Es literalmente la tarea asignada ("adaptar sin prompts humanos")
  y la fuente de la medida del "coste del prompt": `points_per_side²` llamadas al decodificador.

### 2.3 Punto de operación → umbrales relajados
- **Opciones:** los valores por defecto de SAM 2 (`pred_iou` 0.8 / stability 0.95) frente a un
  ajuste relajado.
- **Decisión:** `points_per_side`=16 (256 llamadas al decodificador), `pred_iou`=0.5,
  stability=0.8.
- **Evidencia:** con los **valores por defecto el modelo devuelve 0 máscaras** en nuestra
  imaginería — es en sí mismo evidencia directa del cambio de dominio, e inútil como línea base.
- **Advertencia:** el punto de operación cambia las cifras; un barrido de umbrales es un
  seguimiento natural.

## 3. Evaluación

### 3.1 Métrica → emparejamiento por instancia, mejor IoU por campo
- **Opciones:** precisión por píxel; IoU semántica; mejor emparejamiento por campo.
- **Decisión:** para cada campo real (`parcels`), buscar la máscara de SAM 2 que mejor se
  solapa y registrar su IoU. SAM 2 devuelve una *lista* de máscaras sin orden, no una partición,
  así que el emparejamiento por instancia es el planteamiento honesto. Se reportan: IoU
  media/mediana del mejor emparejamiento, proporción de campos con IoU>0.5, más el número de
  máscaras, la cobertura y la fracción de máscara sobre campo.
- **Advertencia:** las máscaras de SAM 2 pueden anidarse/solaparse, así que el "mejor
  emparejamiento" favorece a un modelo que sobre-segmenta.

### 3.2 Dónde se mide → solo píxeles de campo
- **Decisión:** restringir las métricas a `parcels ≠ 0`. Filtrar *qué parches* conservamos
  (nunca) es una decisión distinta de restringir *dónde* medimos (sí).

### 3.3 Controles → estratificación por tamaño + control de alineación
- **Decisión:** dividir los campos en la mediana de tamaño de cada parche, y registrar una
  puntuación de alineación de imagen (fuerza de borde en los límites de parcela frente al
  resto) con cada parche.
- **Evidencia:** la alineación es de 1.2–2.2× en todas partes y el desplazamiento de mejor
  ajuste es (0,0) en 8/12 parches (±1 px en el resto), así que un IoU bajo es **culpa del
  modelo**, no un artefacto de georreferenciación. Los campos grandes puntúan ~3× mejor que los
  pequeños — un hallazgo, no ruido.

## 4. Fase B — la intervención (planificada)

- **Decisión:** adaptar el **decodificador de máscaras** (candidato: descongelar sus últimas
  *k* capas), dejando el codificador de imagen congelado; la ablación controlada es congelado vs
  descongelado de las últimas *k* capas.
- **Por qué el decodificador:** es el núcleo calificado de la tarea y es pequeño (~9 % de los
  parámetros), así que es el lugar más barato para probar una hipótesis.
- **Advertencia:** una cabeza de clasificación añadida por encima evitaría el decodificador y
  sacrificaría el núcleo calificado — no es una opción.

## 5. Fase C — cambio de dominio y honestidad

- **Decisión:** el cambio de dominio **principal** es SA-1B (fotos naturales) → Sen4AgriNet
  (Sentinel-2); está presente en todos los experimentos, no es un paso aparte. La división
  **ES vs FR** se reetiqueta como una *comprobación de robustez entre regiones dentro del mismo
  dominio*.
- **Evidencia:** el modelo congelado devuelve 0 máscaras con los umbrales por defecto y empareja
  solo 1 de 30 campos reales con IoU>0.5 — el cambio está medido, no supuesto.

## Dónde se usó asistencia de IA (calificado explícitamente)

- **Se usó para:** redactar y refactorizar los scripts de `src/` (descarga, E/S, análisis de
  disponibilidad, evaluador) y el andamiaje de los notebooks; ejecutar las mediciones; redactar
  el texto de estos documentos.
- **Decidido por nosotros (el equipo):** el alcance de los datos, la elección del mes, la regla
  de una sola fecha, el diseño de la métrica y la intervención de la Fase B. Cada una de esas
  decisiones está listada arriba con la evidencia en que la basamos.
- **Verificación:** ejecuciones reales respaldan cada cifra (la descarga de 900 parches, los CSV
  de disponibilidad, la evaluación de la línea base).
