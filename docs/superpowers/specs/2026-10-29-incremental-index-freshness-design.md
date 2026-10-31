# Segunda ronda: índices actuales y lecturas acotadas

El usuario pidió otra ronda de brainstorming e implementación, manteniendo la
delegación de decisiones de la primera ronda.

## Oportunidades y enfoques

| Alternativa | Beneficio | Coste o riesgo |
| --- | --- | --- |
| Caché por mtime/tamaño | Evitar lecturas de archivos iguales | Puede omitir cambios con metadatos iguales |
| Unificar lectores e introducir un resultado de indexación común | Una sola lectura por actualización | Amplía interfaces y responsabilidades |
| Corregir invalidación y acotar lecturas existentes | Recupera símbolos obsoletos y evita leer archivos descartables completos | Conserva dos lecturas cuando un archivo válido cambió |

Elijo la tercera alternativa: resuelve fallos observables sin cambiar APIs ni
formatos de persistencia. La invalidación por contenido sigue funcionando.

## Comportamiento esperado

- `ensure_indexed` compara hash **y versión de parser**, igual que `index_file`.
  Una corrección de parser debe llegar a archivos sin cambios y a filas antiguas
  con versión cero. También se actualizan archivos de configuración y fuentes
  que legítimamente no producen símbolos.
- Si no se consigue actualizar un archivo con versión antigua, no se informa una
  indexación exitosa solo porque su hash coincidía antes del intento.
- Una extensión no soportada se descarta sin leer el contenido ni consultar SQLite.
- Para fuentes, ambos puntos de entrada leen como máximo `MAX_FILE_SIZE + 1`
  bytes antes de descartar el archivo. El byte adicional diferencia el límite
  exacto de un archivo que lo supera. Se mantiene el límite existente de 1 MB.
- La configuración conserva su política actual de seguimiento por hash, incluso
  cuando supera el límite usado para fuentes.
- Mantener el retorno booleano, el conteo de símbolos, `notify_integrity`,
  la detección de idioma y la recuperación de errores de lectura actuales.

## Alcance y verificación

Modificar únicamente `smart_index.py`, `indexer.py`, pruebas y estos documentos.
No tocar cambios previos ni dependencias. Probar con archivos reales y SQLite
aislado: recuperación por versión, versión cero, cero símbolos, parser ausente,
fuentes grandes, límite exacto, configuración grande y extensiones no soportadas.
Ejecutar pruebas relacionadas, revisión independiente y la suite completa.
