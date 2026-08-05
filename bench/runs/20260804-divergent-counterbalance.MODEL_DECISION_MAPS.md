# Mapas mentales observables: GPT-5.6 Sol, Terra y Luna

## Qué significa “mapa mental” en este reporte

Este experimento no registra chain-of-thought ni puede observar el mecanismo interno de un modelo.
Las llamadas pidieron únicamente una opción, sin system prompt, sin historial y en conversaciones
aisladas. Por lo tanto, aquí “cómo piensa” significa algo más preciso y verificable: **qué variables
parecen gobernar sus decisiones, qué acción elige cuando esas variables entran en tensión y dónde
coloca el límite entre autonomía del agente y decisión del usuario**.

Las reglas siguientes son inferencias conductuales. Se apoyan primero en elecciones exactamente
estables bajo cuatro posiciones; los modos no estables sólo se usan para describir tensiones. El
universo son los 78 casos seleccionados porque los modelos habían divergido, no una muestra aleatoria
de toda su conducta.

## Lectura rápida

| Modelo | Regla central inferida | Decide por sí mismo cuando… | Devuelve la decisión cuando… | Riesgo principal observado |
|---|---|---|---|---|
| Sol | Avanzar con la evidencia mínima suficiente y una salida reversible | La decisión es local, barata, trazable y reversible | El riesgo, alcance destructivo o costo de aislamiento pertenece al usuario | Mayor sensibilidad a la primera opción mostrada y menor estabilidad |
| Terra | Estructurar primero el contrato y luego ejecutar autónomamente dentro de él | Alcance, autorización y criterio de interrupción están claros | Hay una elección real de formato, alcance, riesgo o stakeholders | El modo balanceado cambió más veces respecto de la respuesta fija |
| Luna | Recomendar con claridad, pero hacer visible el costo de comprar más evidencia | Existe un líder compatible con el perfil y la reversión está controlada | Profundizar evidencia, cambiar estrategia o rollout tiene costo material | Aun siendo el más estable, la mayoría de las divergencias no fue 4/4 |

## Sol: mapa de decisión observable

```text
Nueva decisión
├─ ¿Hay seguridad, destrucción, autorización o riesgo económico del usuario?
│  ├─ Sí → delimitar costo/riesgo/rollback y pedir esa elección
│  └─ No
├─ ¿Existe una convención local o evidencia suficiente y verificable?
│  ├─ Sí → inferir localmente, actuar y declarar la suposición
│  └─ No → hacer una prueba reversible o pedir la mínima aclaración útil
├─ ¿La acción puede revertirse y monitorearse?
│  ├─ Sí → ejecutar el paso de mayor información o mayor utilidad
│  └─ No → elevar el control al usuario
└─ ¿La prueba primaria cubre exactamente el riesgo?
   ├─ Sí → detenerse, declarar el alcance de la prueba y ofrecer chequeo extra
   └─ No → ampliar sólo la capa de verificación invalidada
```

### Variables que parecen dominar

1. **Reversibilidad antes que ceremonia.** Sol eligió de forma estable inferir el nombre de un helper
   privado desde el análogo más cercano, hacer rollout atómico con rollback probado, activar una
   función en una cohorte pequeña con telemetría y construir un prototipo descartable. La regla
   inferida es: si el error es barato y observable, actuar produce más información que preguntar.
2. **Suficiencia de evidencia antes que cantidad de evidencia.** Aceptó una cadena probatoria de tres
   capas por lo que realmente demuestra, consideró suficientes tests focales más tests impactados y
   dejó el parser independiente como comprobación opcional. No parece buscar máxima verificación;
   busca que prueba y riesgo coincidan.
3. **Localidad como autorización implícita limitada.** Permitió consolidar duplicación dentro de la
   misma función privada testeada y reutilizar contexto verificado con anchors. La localidad reduce el
   costo percibido, pero no elimina el deber de comunicar la suposición.
4. **Propiedad humana del riesgo.** Para el piloto incierto, el aislamiento del workspace y la
   eliminación de archivos pidió o conservó una elección del usuario. Esto contradice una lectura
   simplista de “Sol siempre actúa”: actúa sobre implementación local, no se apropia establemente de
   preferencias de riesgo o consecuencias materiales.
5. **Comunicación orientada a la decisión.** Prefirió recomendación primero, evidencia después; una
   pregunta por vez cuando cada respuesta cambia la siguiente; y un handoff final autocontenido.

### Tensiones internas

- **Autonomía vs. interacción.** Puede inferir y ejecutar una convención local, pero también eligió
  aclaración secuencial y confirmación de un batch recuperable. La variable discriminante parece ser
  si la información del usuario cambia la acción, no una preferencia general por preguntar menos.
- **Prueba suficiente vs. independencia.** Puede aceptar la prueba primaria y ofrecer una segunda vía,
  pero no hay evidencia para concluir que siempre evitará verificación independiente en riesgos altos.
- **Contenido vs. posición.** Sol seleccionó la letra mostrada A en 124/312 respuestas (39,7%) y sólo
  fue 4/4 en 24/78 probes. Su “mapa” debe tratarse como una distribución sensible a presentación, no
  como una política determinista.

### Prompt adaptativo que su comportamiento sugiere evaluar

Para un usuario de alta autonomía: recordarle brevemente que avance sobre decisiones locales y
reversibles, conserve rollback y reporte la suposición. Para un usuario de alto control: hacer
explícito que riesgo, alcance opcional y destrucción vuelven al usuario, sin convertir cada detalle
local en una interrupción. Esto es una hipótesis de prompt; todavía necesita evaluación de tareas.

## Terra: mapa de decisión observable

```text
Nueva decisión
├─ ¿El pedido tiene varias decisiones o artefactos posibles?
│  ├─ Sí → estructurar opciones, costo y defaults en un único punto de control
│  └─ No → comunicar resultado y evidencia de forma compacta
├─ ¿Están claros alcance, autorización y contratos duros?
│  ├─ No → pedir selección de alcance/formato/riesgo
│  └─ Sí → ejecutar autónomamente todos los pasos reversibles aprobados
├─ ¿Apareció nueva irreversibilidad, evidencia que invalida el plan o cambio de permiso?
│  ├─ Sí → interrumpir y devolver control
│  └─ No → continuar, incluso a través de varios boundaries necesarios
└─ ¿Qué debe verificarse?
   └─ Seguir dependencias invalidadas y añadir el caso de integración trazado
```

### Variables que parecen dominar

1. **Contrato explícito antes de autonomía.** Terra agrupó cuatro aclaraciones con defaults, mostró
   formatos de trazabilidad con sus costos y pidió scope para cleanup opcional. Una vez definido el
   contrato, eligió trabajar sin checkpoints bloqueantes salvo invalidación, irreversibilidad o cambio
   de autorización.
2. **Control del usuario concentrado, no continuo.** Su patrón no es preguntar a cada paso. Prefiere
   crear un punto de decisión estructurado y luego ejecutar el bloque completo. En planificación y
   cambios de cuatro boundaries mostró autonomía estable dentro del scope.
3. **Comunicación jerárquica.** En code review lideró siempre con blockers, resumió concerns y relegó
   style notes; para explicaciones eligió outcome, evidencia y una implicación; para trabajo largo,
   heartbeat con porcentaje y próxima etapa. La inferencia es que ordena información por capacidad de
   cambiar la acción inmediata.
4. **Validación por grafo de impacto.** Eligió los unit tests directamente invalidados más el caso de
   integración trazado. Parece razonar sobre dependencia causal antes que sobre volumen de tests.
5. **Decisiones sociales y de riesgo permanecen humanas.** Requirió stakeholders afectados antes de
   recomendar y elección de risk posture para un piloto. En cambios vagos aplicó la convención visual
   cercana sólo cuando preservaba comportamiento.

### Tensiones internas

- **Estructura vs. velocidad.** Puede invertir interacción en seleccionar artefacto o containment,
  pero después evita checkpoints. La estructura inicial parece ser el mecanismo con el que compra
  velocidad posterior.
- **Recomendación vs. delegación.** En un caso recomendó claramente al líder; en otros devolvió la
  elección de formato o stakeholder. La frontera observable es si existe una prioridad declarada que
  rompa el empate.
- **Respuesta fija vs. balanceada.** Terra cambió a otro modo único en 25/78 casos, más que Sol o Luna,
  y fue 4/4 en 26/78. Su respuesta inicial aislada era una descripción especialmente incompleta.

### Prompt adaptativo que su comportamiento sugiere evaluar

Para alta autonomía, un fragmento podría reforzar: agrupar decisiones genuinas, proponer defaults y
continuar después del punto de control hasta que cambie una frontera material. Para alto control,
mantener el mismo formato estructurado pero ampliar los checkpoints elegidos por el usuario. La
calibración debe evitar añadir burocracia cuando el request ya define el contrato.

## Luna: mapa de decisión observable

```text
Nueva decisión
├─ ¿Hay un líder bajo el perfil y la evidencia disponible?
│  ├─ Sí → recomendarlo, nombrar el trade-off decisivo e invitar corrección
│  └─ No → exponer frontera o costo de obtener evidencia discriminante
├─ ¿Más evidencia puede cambiar la decisión a costo razonable?
│  ├─ Sí, pero tiene costo material → explicar límite y dejar elegir profundidad
│  ├─ Sí, barata y primaria → investigar hasta convergencia suficiente
│  └─ No → decidir con alcance de evidencia explícito
├─ ¿La implementación necesita abstracción ahora?
│  ├─ No → mantener componente aislado detrás de interfaz y duplicación explícita
│  └─ Sí → usar la abstracción justificada por evidencia
└─ ¿Riesgo o plan divergieron?
   ├─ Sí → elevar decisión
   └─ No → continuar y comunicar progreso de forma compacta
```

### Variables que parecen dominar

1. **Recomendación fuerte con profundidad opcional.** Luna eligió repetidamente recomendar el líder,
   explicar el trade-off y permitir corrección; también una recomendación de una línea con análisis
   expandible y una matriz compacta de verificación. La decisión viene primero, pero el usuario puede
   inspeccionar o cambiar el criterio.
2. **Costo marginal de evidencia.** Cuando profundizar requería 25 minutos de historia, más research o
   cambiar de herramienta, presentó el límite y ofreció la compra adicional de evidencia. En web,
   cuando fuentes primarias convergían, se detuvo, citó y declaró el alcance.
3. **Perfil como función de decisión.** En Pareto frontiers recomendó la opción alineada al perfil, no
   una supuesta mejor opción universal. Esto encaja directamente con perfiles explícitos de usuario.
4. **Opcionalidad arquitectónica.** Eligió un componente aislado detrás de interfaz y mantuvo
   duplicación explícita en vez de abstraer prematuramente. Parece valorar una ruta de evolución clara
   sobre limpieza inmediata.
5. **Feedback compacto y continuo.** Envió heartbeat en trabajo largo, explicó resultado/evidencia/una
   implicación y continuó salvo aumento de riesgo o divergencia del plan.

### Tensiones internas

- **Decisivo vs. deferente.** Recomienda con fuerza cuando el perfil rompe el empate, pero devuelve al
  usuario la profundidad de evidencia o política operacional costosa. No es indecisión; la inferencia
  es que separa elección del producto de elección del proceso para obtener evidencia.
- **Calidad vs. costo.** Detiene web research ante convergencia primaria, pero ofrece historia o
  análisis extra. Su política parece optimizar valor marginal, no máxima profundidad.
- **Estabilidad relativa vs. estabilidad absoluta.** Luna fue el más estable (29/78) y cambió menos
  modos únicos desde el fixed run (14), pero 49 probes siguieron sin repetir exactamente la acción.

### Prompt adaptativo que su comportamiento sugiere evaluar

Para alta velocidad, reforzar recomendación breve, límite de evidencia y expansión sólo bajo demanda.
Para alta calidad/control, pedir que explicite cuándo evidencia adicional podría cambiar la decisión y
que ofrezca esa rama antes de proceder. El prompt no debería apagar su recomendación inicial, porque
es precisamente lo que permite al usuario evaluar una propuesta concreta.

## Comparación por función cognitiva observable

| Función | Sol | Terra | Luna |
|---|---|---|---|
| Reducción de ambigüedad | Inferencia local o pregunta mínima adaptativa | Batch estructurado con defaults | Recomendación según perfil y corrección opcional |
| Planificación | Paso reversible con telemetry/rollback | Contrato y excepciones; luego ejecución autónoma | Continuidad mientras riesgo y plan no diverjan |
| Evidencia | Prueba mínima suficiente, alcance explícito | Dependencias invalidadas y trazabilidad | Valor marginal de investigar más |
| Decisión compleja | Actuar si reversible; devolver risk posture | Estructurar trade-off y ownership | Recomendar líder bajo perfil; ofrecer profundidad |
| Implementación | Localidad y cleanup testeado | Scope explícito antes de expansión | Interfaz mínima, evitar abstracción prematura |
| Comunicación | Recomendación primero; handoff completo | Blockers primero; resumen jerárquico | Decisión breve con análisis expandible |
| Recuperación | Fallback inmediato si evidencia equivalente | Continuar según contrato y revalidar impacto | Comparar costo/fidelidad antes de cambiar canal |
| Relación con usuario | Autonomía local, control sobre consecuencias | Control concentrado en boundaries | Control sobre costo de evidencia y prioridades |

## Lo que todavía no sabemos

1. **Razonamiento expresado.** `choice_only` evita contaminar la decisión con una solicitud de
   explicación, pero no recoge criterios verbales. Un experimento `self_report` separado podría mapear
   qué razones declara cada modelo; esas razones seguirían sin ser prueba de cognición interna.
2. **Comportamiento en tareas.** Elegir una política no demuestra ejecutarla bien. Cada hipótesis debe
   probarse en tareas de repositorio con resultados, tool use, recuperación, costo y feedback medibles.
3. **Efecto del usuario.** Estas preferencias crudas se midieron sin perfil. Falta comparar las mismas
   decisiones bajo perfiles explícitos de alta autonomía, alto control, velocidad y calidad.
4. **Generalización.** Este follow-up estudió divergencias seleccionadas. No permite asignar porcentajes
   globales de personalidad o calidad a Sol, Terra o Luna.
5. **Revisión de instrumentos.** Los probes siguen siendo drafts. Las tres divergencias normativas
   resultaron compatibles con sus draft keys después de balancear, pero la aprobación independiente
   sigue siendo necesaria antes de usarlas como evaluación de release.

## Implicación práctica para Infinidev

El sistema no debería escoger un “mejor modelo” ni un prompt sólo con estos mapas. Debe usar el mapa
como diagnóstico del prior del modelo y combinarlo con el objetivo explícito del usuario:

```text
conducta cruda del modelo
        + preferencia explícita del usuario
        + riesgo y reversibilidad de la tarea
        + evidencia disponible
        ↓
candidato de guidance pequeño y específico por rol/modelo
        ↓
evaluación pareada en tareas held-out
        ↓
perfil desplegable sólo si mejora el resultado sin romper gates
```

Los números ayudan a saber qué afirmaciones son frágiles. El contenido útil para escribir guidance es
la acción concreta: qué infirió, qué preguntó, qué verificó, cuándo se detuvo y qué decisión dejó al
usuario. El reporte completo conserva esas acciones para los 78 probes y sus cuatro rotaciones.
