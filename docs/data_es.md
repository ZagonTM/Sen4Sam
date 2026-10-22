# Datos — Sen4AgriNet (Sentinel-2)

Usamos [Sen4AgriNet](https://github.com/Orion-AI-Lab/S4A) (Sykas et al., 2022), un
benchmark de **series temporales** de Sentinel-2 para segmentación agrícola. Todas las
cifras medidas que aparecen a continuación provienen de
`notebooks/00_es_data_availability.ipynb`.

## El conjunto de datos: información básica

El conjunto de datos Sen4AgriNet se basa en datos de Sentinel-2 (Drusch et al., 2012):

- Espacialmente incluye regiones de Occitanie, Francia (FR) y Cataluña, España (ES).
- Cubre los años 2019 y 2020, pero 2020 solo incluye datos de España.
- Trabajamos con el **subconjunto Patches Assembled Dataset (PAD)**: parches de imagen con
  etiquetas a nivel de píxel. (El otro, OAD, son estadísticas por parcela, inservibles para
  segmentación.)
- **Un parche** = un archivo netCDF sobre una rejilla de 10 m (366×366 px, dentro de una
  tesela de ~110×110 km con cientos de parches).
- Por cada parche: 13 bandas `B01..B12, B8A` como serie temporal (~14–70 fechas), más
  `parcels` (id de campo por píxel → verdad de referencia de instancias) y `labels` (tipo de
  cultivo → verdad de referencia semántica, no usada aquí). Ambos mapas de etiquetas son
  anuales.

## Selección de datos
Para entender el proceso de selección de datos, lee:
`notebooks/00_es_data_availability.ipynb`. **Aquí solo se da un breve resumen.**

Se hicieron las siguientes selecciones a partir del conjunto base:
- Rango temporal: solo imágenes de 2019: 2020 no incluye Francia y crearía un conjunto
  espacialmente desequilibrado.
- Teselas y parches: 3 teselas por país elegidas al azar, 150 parches cada una = 900
  seleccionados.
  - Al azar para evitar sesgos y garantizar condiciones del mundo real
  - Necesario porque el conjunto completo es demasiado grande para procesarlo en Google Colab
- Incluimos una adquisición (fecha) por parche. Todas del mes de mayo, ya que reúne dos
  requisitos importantes:
  - La cobertura de nubes (Hollstein et al., 2016) debe ser <5 % para cada parche
    individual, ya que SAM 2 (Ravi et al., 2024) trabaja con una sola imagen de entrada
    (sin series temporales).
  - La estacionalidad juega un papel importante en el análisis del uso agrícola del suelo.
    Los patrones estacionales quedan fuera del alcance de este estudio; nos centramos solo en
    la estación de crecimiento activo para reducir la variabilidad y facilitar el análisis de
    errores.
- **Cobertura de campo:** sin filtro — se conserva una mezcla realista de parches agrícolas y
  no agrícolas.
- **Resultado:** **888 parches** (148 por tesela, 444 ES + 444 FR), una imagen cada uno.

## Preprocesado

Convertimos un parche en una imagen RGB más su verdad de referencia: se elige la adquisición
de **mayo menos nubosa**, se construye `B04/B03/B02` → ÷10000 → estirado de percentiles 2–98
→ uint8, y se guarda junto con `parcels` y `labels`. Las métricas se calculan solo sobre
píxeles de campo (`parcels ≠ 0`).

## Advertencias

- Aplicar imágenes satelitales a un modelo de imágenes naturales es el **cambio de dominio**
  (SA-1B, Kirillov et al., 2023 → Sentinel-2) pedido en la Fase C.
- Cada parche conserva su propia fecha de mayo menos nubosa, así que los parches abarcan todo
  el mes (1–31 de mayo).

## Referencias

- Sykas, D., Sdraka, M., Zografakis, D., Papoutsis, I. (2022). *A Sentinel-2
  Multiyear, Multicountry Benchmark Dataset for Crop Classification and
  Segmentation With Deep Learning.* IEEE JSTARS. doi:10.1109/JSTARS.2022.3164771.
- Drusch, M. et al. (2012). *Sentinel-2: ESA's Optical High-Resolution Mission
  for GMES Operational Services.* Remote Sensing of Environment. doi:10.1016/j.rse.2011.11.026.
- Hollstein, A., Segl, K., Guanter, L., Brell, M., Enesco, M. (2016).
  *Ready-to-Use Methods for the Detection of Clouds, Cirrus, Snow, Shadow, Water
  and Clear Sky Pixels in Sentinel-2 MSI Images.* Remote Sensing 8(8):666.
  doi:10.3390/rs8080666.
- Kirillov, A. et al. (2023). *Segment Anything.* ICCV. arXiv:2304.02643.
- Ravi, N. et al. (2024). *SAM 2: Segment Anything in Images and Videos.*
  arXiv:2408.00714.
