# Diagnóstico cross-validation del corpus manual

Estado: evidencia de desarrollo; **no** reemplaza validation ni el holdout
sellado requerido para aceptar el mini-modelo.

## Protocolo

- encoder: `intfloat/multilingual-e5-small` congelado;
- 277 requests escritos manualmente;
- cinco folds deterministas, estratificados por conjunto exacto de policies;
- train, selección y test separados en cada fold;
- `context_before` se incluye explícitamente cuando la interpretación depende
  de un turno anterior;
- una predicción sólo cuenta como correcta si coincide todo el conjunto
  multi-label.

El comando reproduce el diagnóstico y escribe la evidencia completa fuera del
repositorio:

```bash
uv run python -m bench.task_policy_manual_cv \
  --head cardinality_mlp \
  --output /tmp/infinidev-task-policy-manual-cv.json
```

## Baseline observado

| Head | Exact match | Macro F1 | Precision | Recall | Falsas activaciones |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP cardinalidad 0/1/2 | 71.48% | 75.41% | 82.55% | 70.28% | 7 |
| MLP cardinalidad 0/1/2/3 | 71.48% | 75.51% | 82.24% | 70.68% | 6 |
| Ridge selectivo por policy | 63.90% | 68.30% | 84.00% | 59.04% | 0 |
| MLP independiente por policy | 66.06% | 71.93% | 81.04% | 68.67% | 7 |
| **E5-base + MLP 0/1/2/3** | **80.87%** | **84.70%** | **91.24%** | **79.52%** | **4** |

La corrección de cardinalidad era necesaria: el head anterior colapsaba todo
triple-label a dos y, por construcción, nunca podía acertarlo. Sin embargo, no
explica la brecha principal. El ridge prueba además que eliminar activaciones
falsas elevando thresholds destruye demasiado recall.

Un sweep controlado descartó falta de entrenamiento y capacidad del head:
60 epochs dieron 70.76% exact-match, mientras 180 y 360 dieron el mismo 71.48%;
duplicar el hidden size de 96 a 192 tampoco cambió exact-match y empeoró F1.
En cambio, cambiar solamente el encoder a `multilingual-e5-base` añadió 9.39
puntos de exact-match. La capacidad contextual del encoder sí es parte del
cuello de botella, aunque no reemplaza la cobertura ausente.

## Fronteras medidas

En el mejor baseline actual:

| Policy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| bugfix | 74.36% | 69.05% | 71.60% |
| feature | 74.07% | 47.62% | 57.97% |
| refactor | 75.76% | 62.50% | 68.49% |
| research | 86.11% | 72.09% | 78.48% |
| review | 92.11% | 87.50% | 89.74% |
| performance | 87.80% | 85.71% | 86.75% |

Los errores multi-label suelen conservar una policy y omitir las demás. Las
confusiones single-label más frecuentes son `feature -> refactor`,
`refactor -> bugfix` y `bugfix -> performance`. Las seis falsas activaciones
restantes se concentran en requests incompatibles, hipotéticos, de estado y en
tareas reales no cubiertas por estas policies.

`E5-base` usa 768 dimensiones y en esta máquina observó aproximadamente
1.41 GiB adicionales de RSS, 45 ejemplos/s y 30.0 ms p50 por request caliente.
`E5-small` usa 384 dimensiones y observó aproximadamente 665 MiB, 174
ejemplos/s y 9.3 ms p50 en la corrida comparable. Por eso `base` es un teacher
o candidato de calidad razonable, pero no una sustitución runtime gratuita.

## Decisión para la siguiente iteración

1. completar cobertura single-label sin usar variantes mecánicas;
2. aumentar cada combinación multi-label con operaciones causalmente
   independientes y lenguaje que no nombre las categorías;
3. profundizar negativos adversariales de `conflicting_request`,
   `hypothetical_future`, `status_only` y `unsupported_method`;
4. comparar el MLP de cardinalidad contra un head multi-label independiente con
   una salida explícita de abstención;
5. después de congelar arquitectura y thresholds, escribir validation nueva y
   finalmente abrir un holdout de familias nunca usadas.

No se afirmará 95% a partir de esta cross-validation. El gate final sigue
exigiendo más de 95% en exact match, macro F1, precision y recall, además de
cero activaciones sobre requests sin policy.

## Iteración `composition-08`

Se añadieron 20 escenarios dirigidos por los errores de `E5-base`: cuatro
`feature+research`, cuatro `bugfix+performance`, dos de cada par restante más
dos triples. El corpus pasó a 297 filas y las combinaciones prioritarias a
4–8 ejemplos cada una.

Los nuevos escenarios obtuvieron sólo 2/20 exact-match al quedar en su fold de
test. El modelo suele conservar una operación y omitir la otra. Esto no es una
regresión comparable uno-a-uno del corpus anterior: se cambió deliberadamente
la distribución evaluada y la asignación de folds. Sí demuestra que entrenar
con aproximadamente 2–5 positivos de una combinación no alcanza para
generalizar su composición.

Sobre las 297 filas, el sweep controlado con `E5-base` dio:

| Configuración | Exact | Macro F1 | Precision | Recall | Falsas activaciones |
| --- | ---: | ---: | ---: | ---: | ---: |
| hidden 96, 60 epochs | **79.46%** | **85.13%** | **92.71%** | **78.69%** | **2** |
| hidden 96, 180 epochs | 78.45% | 84.58% | 91.94% | 78.35% | 2 |
| hidden 96, 360 epochs | 78.45% | 84.58% | 91.94% | 78.35% | 2 |
| hidden 192, 180 epochs | 75.42% | 81.59% | 88.80% | 76.29% | 6 |

El experimento descarta más epochs o un head más ancho como corrección actual.
La siguiente expansión debe elevar la cantidad y diversidad causal de cada
combinación antes de volver a seleccionar arquitectura.

## Iteración `composition-09`

Se añadieron otros 40 escenarios multi-label escritos individualmente. El
corpus llegó a 337 filas y la cardinalidad 0/1/2/3 quedó en 61/186/77/13. Este
lote aumenta todas las combinaciones prioritarias sin repetir IDs, textos ni
familias de escenario.

Con `E5-base`, 60 epochs y hidden size 96, el MLP de cardinalidad obtuvo 69.44%
exact-match y 82.02% macro F1. El descenso no es comparable con el lote
anterior como una regresión del modelo: los 40 challenges nuevos forman parte
del test out-of-fold y sólo 7 acertaron el conjunto completo. En 32 de los 33
errores restantes el modelo predijo menos labels de los esperados.

El balance de la pérdida de cardinalidad explica sólo una fracción pequeña:

| Potencia de balance | Exact | Macro F1 | Precision | Recall | Falsas |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 69.44% | 82.02% | 88.15% | 76.52% | 2 |
| 0.25 | 70.03% | 82.37% | 87.99% | 77.31% | 2 |
| 0.50 | **70.33%** | **83.08%** | 87.43% | **78.89%** | 3 |
| 0.75 | 67.66% | 81.95% | 84.31% | 79.42% | 2 |
| 1.00 | 67.95% | 82.09% | 84.55% | 79.42% | 3 |

El head independiente se beneficia más del corpus composicional: alcanzó
71.81% exact-match, 85.15% macro F1 y 84.96% recall, aunque dejó seis falsas
activaciones. Elevar su precisión mínima de calibración de 85% a 95–100%
redujo las falsas activaciones a tres, pero bajó recall a 77.84% y no mejoró
exact-match. Un threshold más conservador no reemplaza negativos representativos.

## Curva de aprendizaje controlada

La curva usa exactamente los mismos folds de validation/test, `E5-base`, el
head independiente, hidden size 96 y 60 epochs. Sólo cambia la fracción
estratificada disponible para train:

| Fracción train | Ejemplos/fold aprox. | Exact | Macro F1 | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25% | 58 | 53.41% | 70.14% | 82.27% | 61.21% |
| 50% | 104 | 62.91% | 77.48% | 82.01% | 73.35% |
| 75% | 157 | 71.22% | 83.19% | 86.40% | 80.47% |
| 100% | 202 | **71.81%** | **85.15%** | 85.41% | **84.96%** |

Los datos importan de forma causal: retirar ejemplos degrada todas las
métricas. Sin embargo, 75% a 100% sólo añade 0.60 puntos de exact-match, señal
de rendimiento decreciente para más ejemplos de la misma distribución. La
siguiente autoría debe dirigirse a negativos que todavía atraviesan otros folds
y a contrastes composicionales, no expandir uniformemente el corpus.
