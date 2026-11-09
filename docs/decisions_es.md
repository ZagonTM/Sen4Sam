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
├── 4. FASE B
│   ├─ ¿qué se entrena?      → transformer.layers[1] + iou_prediction_head (1.71 M, ~3.7 %)
│   ├─ ¿pérdida?             → Dice (token 0) + MSE de la cabeza de IoU
│   ├─ ¿prompt de entrenam.? → centroide de la parcela
│   ├─ ¿criterio de parada?  → early stopping por pérdida de val (v1); por IoU del generador (v2)
│   └─ ¿resultado v1?        → val mejora, test real colapsa (0.156 → 0.064)
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

## 4. Fase B — la intervención

### 4.1 Decisiones tomadas (v1, notebook 07)
- **Qué se entrena:** solo `transformer.layers[1]` del decodificador de máscaras y
  `iou_prediction_head` (1.71 M de parámetros, ~3.7 % del modelo); el codificador de imagen y
  el resto del decodificador quedan congelados. Razón: el decodificador es el núcleo calificado
  de la tarea, es pequeño y es el lugar más barato para probar una hipótesis; una cabeza de
  clasificación externa lo evitaría y no es una opción.
- **Pérdida:** Dice sobre el token de salida 0 + MSE de la cabeza de IoU contra el IoU real de
  las 4 máscaras. Razón: Dice trata el desbalance de clases (campos pequeños); el MSE evita
  que la cabeza de IoU se descalibre al mover las máscaras.
- **Prompt de entrenamiento:** centroide de cada parcela (escalado al marco de 1024 px de SAM),
  200 parches de entrenamiento × hasta 30 parcelas. Razón: es el prompt más simple y
  determinista con etiqueta disponible.
- **Parada:** early stopping sobre la pérdida de validación; mejor época = 4.
- **Resultado v1 (honesto):** en validación con prompts de centroide mejoró: pérdida Dice
  0.704 → 0.342, IoU de máscara 0.312 → 0.530, MSE de IoU 0.099 → 0.027. Pero en el **test real
  con el generador automático** (132 parches) **empeoró mucho**: mejor IoU medio 0.156 → 0.064
  (Δ pareado −0.092, IC 95 % bootstrap [−0.103, −0.083]); campos con IoU>0.5: 14.2 % → 5.9 %;
  máscaras por parche 78.7 → 41.1; cobertura de imagen 0.59 → 0.10; máscara-sobre-campo sin
  cambio (0.58 → 0.59); 128 de 132 parches peores, en todas las teselas y ambos países.
- **Interpretación:** las máscaras desaparecen (colapso de recall) más que empeorar. La pérdida
  de validación no pudo avisar porque mide el mismo tipo de prompt con el que se entrenó.
  Hipótesis **no verificadas**: (1) la cabeza de IoU quedó pesimista para puntos de la rejilla
  fuera del centro de parcela y el filtro `pred_iou_thresh=0.5` los descarta; (2) los tokens
  1–3 (los que usa el generador) empeoraron porque solo se entrenó el token 0; (3) el filtro
  de estabilidad descarta más máscaras.
- **Advertencia:** el cuello de botella real podría ser la densidad del prompt, no el
  decodificador: la parcela mediana mide ~10×10 px y la rejilla 16×16 sobre 366 px deja un
  punto cada ~23 px, así que una parcela mediana recibe un clic ~20 % de las veces (aritmética,
  no medición).

### 4.2 Fase B v2 — ablación: expectativas escritas ANTES de los resultados (marca de tiempo 2026-11-08 22:15 UTC-5)

Todas las corridas se evalúan en los 132 parches de test con el generador automático. Las
corridas entrenadas se seleccionan por el IoU medio del mejor emparejamiento del generador en
val (salvo `A_loss_select`). Las líneas "Resultado:" se llenan después, sin editar la predicción.

**Diagnósticos sobre el modelo sin entrenar**

1. **Filtros apagados** (`--pred-iou 0 --stability 0`).
   - Predicción: el IoU medio sube respecto a 0.161 y aumentan máscaras y cobertura; en el
     modelo v1 con filtros apagados el IoU se recupera en gran parte hacia el nivel base.
   - Razón: si el IoU se recupera, la culpa es de la cabeza de IoU/estabilidad (filtrado); si
     sigue bajo, las máscaras mismas empeoraron.
   - Resultado: **CONFIRMADO.** Sin entrenar: filtros apagados 0.262 vs encendidos 0.156 (329 vs 79 máscaras/parche; Δ +0.106 [+0.100, +0.112]). `A_loss_select` (estilo v1): apagados 0.249 vs encendidos 0.092, frente a 0.262 sin entrenar apagado (Δ -0.013): el IoU casi se recupera; la caída venía de los filtros, no de las máscaras.
2. **Barrido de `points_per_side` 16 / 32 / 64** (primeros `SWEEP_LIMIT` parches de test).
   - Predicción: 32 y 64 suben recall y IoU medio, sobre todo en campos pequeños y en España;
     el tiempo crece ~cuadráticamente (≈4× y ≈16× frente a 16).
   - Razón: a 16 una parcela mediana recibe clic ~20 % de las veces; más puntos reducen el
     coste del prompt en cobertura, con retornos decrecientes.
   - Resultado: **PARCIAL.** 40 parches, pps 16/32/64: IoU 0.083 / 0.146 / 0.185, 1.0 / 8.9 / 17.7 s por parche. Ganancia CONFIRMADA (pps32 en 132 parches de test: 0.253 vs 0.156). "Sobre todo España" REFUTADO: en 132 parches la ganancia es ES +0.067, FR +0.126. El tiempo creció ~9x y ~18x, no ~4x y ~16x (GPU compartida, ruidoso). No se midió el desglose por campos pequeños.

**Corridas entrenadas**

3. **A_centroid_tok0** (prompt centroide, pérdida en token 0).
   - Predicción: mejora modesta o nula frente a la línea base (~0.16); mejor que v1 porque la
     selección usa el generador.
   - Razón: sigue entrenando solo el token 0 con prompts que el generador no usa.
   - Resultado: **CONFIRMADO (trivialmente).** Se conservó la época 0 (IoU generador en val 0.121 -> 0.082 tras la época 1), así que test = sin entrenar: 0.156, Δ 0.000, 78.6 máscaras/parche. Sin ganancia, como se predijo, pero solo porque se eligieron los pesos sin entrenar.
4. **B_centroid_multi** (centroide, pérdida en tokens 1–3, mejor de tres).
   - Predicción: algo mejor que A.
   - Razón: el generador usa los tokens 1–3, así que se alinea entrenamiento y uso.
   - Resultado: **REFUTADO.** Se conservó la época 0 (val 0.121 -> 0.079 tras la época 1); test 0.156 = sin entrenar = A. No es mejor que A (val época 1: 0.079 vs 0.082).
5. **C_random_tok0** (puntos aleatorios, token 0).
   - Predicción: mejor que A, peor que D.
   - Razón: los puntos aleatorios imitan la rejilla del generador (incluye puntos fuera de
     centro y de fondo), pero el token 0 sigue desalineado.
   - Resultado: **REFUTADO.** Se conservó la época 0 (val 0.121 -> 0.093 tras la época 1); test 0.156 = sin entrenar. No se observó el orden A < C < D (val época 1: C 0.093 > D 0.082 > A 0.082).
6. **D_random_multi** (puntos aleatorios, tokens 1–3).
   - Predicción: **la mejor** de las cuatro y por encima de la línea base.
   - Razón: es la única que alinea tanto la distribución de prompts como los tokens con lo que
     hace el generador en inferencia.
   - Resultado: **REFUTADO.** Se conservó la época 0 (val 0.121 -> 0.082 tras la época 1); test 0.156 = sin entrenar: ni sobre la línea base ni la mejor de las cuatro (val época 1: E 0.110 > C 0.093 > A 0.082 ~ D 0.082 > B 0.079).
7. **A_loss_select** (como A pero parada por pérdida de val; replica v1).
   - Predicción: **la peor**; reproduce el colapso (IoU test <0.10, pocas máscaras).
   - Razón: la pérdida de val no ve el fallo del generador.
   - Resultado: **CONFIRMADO.** Época elegida 4: test 0.092 (Δ -0.064 [-0.073, -0.057]), 63.5 máscaras/parche vs 78.6 (< 0.10, menos máscaras). Con filtros apagados: 0.249.
8. **E_grid_multi** (añadida 2026-11-08 22:20 UTC-5, antes de cualquier resultado v2; los clics de
   entrenamiento son los puntos reales de la rejilla 16x16 del generador que caen en una parcela,
   tokens 1–3).
   - Predicción: más o menos igual que D, quizá algo mejor; sigue limitada por la cobertura de la
     rejilla (solo ~32 % de las parcelas recibe un punto con pps=16, Fase C), así que la mejora
     sobre la línea base será pequeña.
   - Razón: reproduce exactamente la distribución de prompts del generador, pero no puede ayudar
     a parcelas que no reciben ningún punto.
   - Resultado: **PARCIAL.** Se conservó la época 0 (val 0.121 -> 0.110 tras la época 1, la menor caída de A-E); test 0.156 = sin entrenar, Δ 0.000. Igual a D en test (ambos = sin entrenar) y sin ganancia sobre la línea base; solo coincide con "ganancia pequeña" si esta es cero.

9. **Seguimiento, añadido 2026-11-08 22:35 UTC-5 tras ver que A–E conservaron la época 0**
   (todas las configuraciones bajaron la métrica del generador en val tras una época con lr 3e-5):
   E_grid_multi y D_random_multi con lr 1e-5 y 3e-6.
   - Predicción: un lr menor reduce el daño pero no supera al modelo sin entrenar más allá del
     ruido (|Δ| < 0.01 en IoU medio); E_grid queda más cerca de la línea base que D.
   - Razón: si la supervisión densa desplaza la cabeza IoU / las máscaras fuera del punto de
     operación del generador, un paso menor solo ralentiza el desplazamiento.
   - Resultado: **CONFIRMADO.** Las cuatro corridas (D, E con lr 1e-5 y 3e-6) conservaron la época 0; Δ vs sin entrenar = 0.000 en test (|Δ| < 0.01). En val tras la época 1 aún cayó: D 0.082 / 0.091, E 0.109 / 0.113 (vs 0.121), E más cerca de la línea base que D. Un lr menor frenó la caída pero no la eliminó.

10. **Diagnóstico error-de-código vs hallazgo, añadido 2026-11-08 22:37 UTC-5 antes de resultados.**
   (a) E_grid_multi seleccionada por pérdida de val (conserva pesos entrenados) y evaluada sin filtros;
   (b) E_grid_multi con --iou-weight 0 (cabeza IoU sin tocar, solo Dice).
   - Predicción: sin filtros, las máscaras entrenadas son casi tan buenas como las sin entrenar
     (|Δ| < 0.01), así que la caída con filtros viene de la cabeza IoU + el umbral fijo
     pred_iou_thresh; (b) pierde menos que E.
   - Razón: el entrenamiento calibra la cabeza IoU al IoU real de la parcela (más bajo que las
     puntuaciones optimistas de SAM), y más máscaras quedan bajo el umbral fijo 0.5.
   - Resultado: **PARCIAL.** (a) `E_loss` (pesos entrenados): filtros encendidos 0.137 (Δ -0.019 [-0.024, -0.015], 76 máscaras); apagados 0.305 vs 0.262 sin entrenar (Δ +0.042 [+0.039, +0.046]). "|Δ| < 0.01 apagado" REFUTADO (las máscaras entrenadas son mejores), pero la caída con filtros sí viene de la cabeza de IoU + umbral fijo. (b) `E_noiou`: test 0.242, Δ +0.085 [+0.080, +0.091], 277.6 máscaras, apagado 0.305: CONFIRMADO (no pierde, gana).

11. **Corrida con correcciones de diseño F_fixed, añadida 2026-11-08 22:42 UTC-5 antes de
   resultados:** prompts de rejilla, tokens 1–3, puntos de rejilla sobre fondo como negativos
   (máscara objetivo vacía, IoU 0; --neg-frac 0.3), pérdida de la cabeza IoU apagada o encendida
   según E_noiou, lr 1e-5, métrica del generador cada 300 pasos.
   - Predicción: supera al modelo congelado por poco (+0.005 a +0.02 de IoU medio) o empata; no
     cierra la brecha con pps=64 (+0.10 en el subconjunto de 40 parches).
   - Razón: corrige el desajuste entrenamiento/inferencia (clics de fondo, tokens, número de pasos),
     pero la rejilla sigue sin tocar la mayoría de parcelas pequeñas: el prompt sigue siendo el
     cuello de botella.
   - Resultado: **PARCIAL.** `F_iouon` (pérdida IoU encendida): mejor en el paso 0, test 0.156, Δ 0.000: empate, CONFIRMADO. `F_iouoff` (pérdida IoU apagada): mejor paso 2100, val 0.121 -> 0.180, test 0.213, Δ +0.057 [+0.052, +0.062]: REFUTADO (mayor que +0.02). La brecha con pps64 no se cierra: pps32 solo da 0.253 en 132 parches y pps64 +0.102 en 40.

Hipótesis que cada resultado refuta: si D no supera a A, la alineación de prompt/token no es
el factor; si el barrido de rejilla supera a todas las corridas entrenadas, el cuello de
botella es la densidad del prompt y no el decodificador.

### 4.3 Fase B v2 — conclusión

- La caída de v1 vino de entrenar la cabeza de IoU bajo umbrales fijos de SA-1B (`pred_iou_thresh`, estabilidad), no de las máscaras: con filtros apagados, el modelo estilo v1 queda a -0.013 del no entrenado (0.249 vs 0.262), y las máscaras de `E_loss` superan al no entrenado (0.305 vs 0.262).
- Sin pérdida en la cabeza de IoU, el ajuste del decodificador sí ayuda: `E_noiou` 0.242 vs 0.156 (Δ +0.085 [+0.080, +0.091]); `F_iouoff` 0.213 (+0.057). Toda corrida con pérdida de IoU encendida conservó la época 0 (= sin entrenar).
- Sin entrenar nada, la densidad de prompts (pps32: 0.253 en 132 parches; pps64: +0.102 en 40) y apagar los filtros (0.262) aportan cada uno ≈ +0.10, con mayor costo (pps32/64 ~9x/18x más lentos; filtros apagados dan ~329 máscaras/parche).
- Verificación: evaluación independiente en parches gen-val 0.1212 = métrica en entrenamiento 0.12124; la auditoría de código no halló errores.

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
- **Claude Code (Anthropic)** se usó como herramienta de código: `src/train_decoder.py`, las
  opciones nuevas de `src/evaluate_masks.py`, los scripts de RunPod (`scripts/`), el ejecutor
  de experimentos, los scripts de análisis y el borrador de estos documentos.
- **Humanos:** eligieron el tema, ejecutaron los notebooks 00–07 y deben verificar cada
  decisión; ningún integrante debe defender algo que no pueda explicar.
- **Decidido por nosotros (el equipo):** el alcance de los datos, la elección del mes, la regla
  de una sola fecha, el diseño de la métrica y la intervención de la Fase B. Cada una de esas
  decisiones está listada arriba con la evidencia en que la basamos.
- **Verificación:** ejecuciones reales respaldan cada cifra (la descarga de 900 parches, los CSV
  de disponibilidad, la evaluación de la línea base).
