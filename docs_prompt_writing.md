# Cómo se escriben los prompts acá

## De dónde sale este documento

De dos piezas del propio repo, no de teoría:

- **`src/infinidev/prompts/analyst/planner_prompt.py`** — el prompt mejor
  construido del proyecto. Su docstring declara el método: *"organised as an
  epistemology rather than as a form description"*. Este documento es ese
  método, extraído y generalizado.
- **`tests/test_prompt_style_rules.py`** — las reglas que ya son ejecutables.
  Lo que un test puede verificar, lo verifica; lo que no, vive acá.

Lo que sigue es el criterio destilado de esas dos fuentes. No es una teoría
importada sobre cognición: es la observación de qué distingue al prompt que
funciona de los que envejecieron mal en este mismo repositorio.

---

## El marco: qué hace un experto que un prompt malo impide

Un programador con experiencia, soltado en un repositorio ajeno, no ejecuta una
lista de reglas. Hace cinco cosas, en este orden:

1. **Se orienta antes de actuar.** No edita lo que no leyó. Sabe que su memoria
   de otros proyectos es una hipótesis, no un dato.
2. **Hace explícito lo que no sabe.** Distingue "leí esto" de "supongo esto", y
   trata la segunda categoría como deuda a pagar antes de escribir.
3. **Hace el cambio más chico que puede verificar.** No porque sea prudente,
   sino porque un cambio grande que falla no dice *dónde* falló.
4. **Sabe cuándo parar.** Reconoce el patrón "estoy peleando con el problema
   equivocado" y cambia de nivel en vez de intentar una cuarta vez.
5. **Deja registro de lo aprendido**, porque sabe que va a volver acá sin
   acordarse de nada.

Un prompt escrito como catálogo de features no habilita nada de eso. Enumera
capacidades; el experto necesita un **orden de operaciones y un criterio de
verdad**. La diferencia práctica: el catálogo responde "qué podés hacer", y el
trabajo necesita "qué hacés primero y cómo sabés que estuvo bien".

---

## Los principios

### 1. El prompt no vio la tarea. Decilo y cedé la autoridad.

Toda página de instrucciones se escribió antes de que existiera el problema
concreto. Si no lo declara, el modelo la trata como más autoritativa que el
repositorio que tiene enfrente — y entonces una regla general gana sobre una
observación específica, que es exactamente al revés.

> This page was written before your task existed and it cannot see your packet
> or your repository. Where this page and the repository disagree, the
> repository is right.

Esto no es humildad decorativa: es la regla que hace que el modelo prefiera lo
que acaba de leer sobre lo que le dijeron en abstracto.

### 2. El orden del documento es el orden del trabajo

No el índice de features. El planner va: qué te dieron → las llamadas que
convierten un handoff en estructura observada → cómo la evidencia se vuelve
pasos → el check que lleva cada paso → los campos de tarea → los hechos de la
máquina → la llamada final.

Un modelo lee de arriba abajo una vez. Si el documento está ordenado por tema,
tiene que reconstruir la secuencia solo. Si está ordenado por secuencia, la
lectura *es* el procedimiento.

**Prueba rápida:** si podés reordenar dos secciones sin que nada se rompa, el
documento está ordenado por tema.

### 3. Una regla raíz de la que cuelga el resto

El planner tiene una:

> **every path in every step traces back to the packet, to a call you made this
> turn, or to a step above it.** A path you recall from training is a guess
> wearing the costume of a fact.

Todo lo demás en esa página es consecuencia de esa regla. Un prompt sin regla
raíz es una lista de veinte reglas de igual peso, y el modelo que tiene que
sacrificar una no sabe cuál.

### 4. Cada regla lleva su consecuencia, no su justificación

Comparar:

| forma | efecto |
|---|---|
| "Sé cuidadoso con los paths" | ninguno: no dice qué pasa ni cómo verificar |
| "El developer no puede editar tus pasos, así que un path equivocado se sortea durante toda la corrida en vez de arreglarse" | el modelo entiende el costo y puede decidir bajo presión |

La consecuencia es lo que permite **generalizar a un caso no previsto**. La
justificación moral ("es una buena práctica") no generaliza a nada.

### 5. Registro: imperativo puro para el modelo, sujeto no-humano para lo que suaviza

Regla del planner, y la más sutil de todas:

> every softening sentence takes THE PAGE, THE PACKET or THE MACHINE as its
> subject, and every verb whose subject is "you" stays a bare imperative

O sea: la página puede admitir sus límites ("esta página no puede ver tu
repositorio"), pero en cuanto el sujeto es el modelo, el verbo es una orden.
Nunca "deberías leer el archivo" — siempre "leé el archivo".

Esto es lo que deja que el documento se lea como orientación sin violar la
prohibición de hedging. Un prompt sin esta distinción termina eligiendo entre
sonar a manual militar o darle permiso al modelo para saltearse cosas.

### 6. Separar el método de los hechos de la máquina

El planner lo marca explícitamente:

> The three facts below are about the machine, not about method. The machine
> does not read this page.

Los hechos de máquina (el turno termina en la primera llamada, un plan de cero
pasos se descarta, el contador corta a las N llamadas) no son consejos: son
física. Mezclarlos con el método hace que el modelo trate al método como física
y a la física como sugerencia.

### 7. Nombrar el antipatrón, con su costo

"BAD: Set up authentication / GOOD: auth.py validate_token: reject tokens past
exp" enseña más que tres párrafos sobre granularidad, porque el modelo reconoce
formas, no definiciones. El planner va más lejos y le pone nombre a la
diferencia: *"'Set up authentication' is a wish"*.

Un antipatrón sin ejemplo es una advertencia que el modelo no puede aplicar a
su propio output.

### 8. La tabla pregunta → llamada → qué produce

Cuando hay un presupuesto de acciones, la decisión no es "cuántas" sino
"cuáles". El planner lo resuelve con una tabla de cuatro filas: la pregunta, la
llamada que la responde, y **en qué parte del entregable se convierte la
respuesta**. Esa tercera columna es la que evita la exploración sin destino.

### 9. Las tres clases de palabra prohibida

Ya son ejecutables en `tests/test_prompt_style_rules.py`:

| clase | ejemplos | por qué |
|---|---|---|
| **hedge** | could, should, might, prefer, generally, typically, try to | le da permiso al modelo de no hacerlo. Una instrucción o aplica o no es una instrucción |
| **unknown** | appropriate, relevant, as needed, if necessary, reasonable, proper, significant | deja la instrucción en pie y obliga al modelo a inventar el umbral. "Corré el test relevante" — ¿relevante para qué? |
| **flecha** | `→`, `=>`, `->` | la palabra ("entonces", "produce", "en cambio") está en el corpus de entrenamiento; el glifo no |

La diferencia entre hedge y unknown importa: **el hedge da permiso de saltear,
el unknown obliga a adivinar.** El arreglo de un unknown nunca es un sinónimo —
es el criterio que la palabra estaba tapando ("el test que cubre el archivo que
editaste").

### 10. El ejemplo concreto va al final, completo

El planner cierra con una llamada `emit_plan` entera, con valores reales, no con
placeholders. Un ejemplo con `<tu valor acá>` obliga a inferir el formato; uno
completo se copia.

---

## Cómo revisar un prompt

Nueve preguntas, en orden de cuánto revelan:

1. ¿Está ordenado como el trabajo, o como un índice? *(reordená dos secciones:
   si no se rompe nada, es un índice)*
2. ¿Cuál es la regla raíz? Si no la podés nombrar en una frase, no hay.
3. ¿Cada regla dice qué pasa si no se cumple?
4. ¿Los verbos cuyo sujeto es el modelo son imperativos puros?
5. ¿Los hechos de la máquina están separados del método?
6. ¿Declara que no puede ver el problema concreto y quién gana en caso de
   desacuerdo?
7. ¿Los antipatrones tienen ejemplo, no solo prohibición?
8. ¿Sobrevive `uv run pytest tests/test_prompt_style_rules.py`?
9. ¿Un lector que nunca vio el sistema puede ejecutarlo sin preguntar nada?

---

## Estado de los prompts del repo

Medido contra los principios de arriba, al 2026-07-31:

| prompt | vivo en | estado |
|---|---|---|
| `analyst/planner_prompt.py` | pipeline actual | **la referencia** |
| `engine/loop/prompt/text.py` (`LOOP_PROTOCOL`) | pipeline actual, cada iteración del developer | reescrito con estos principios |
| `engine/loop/prompt/text.py` (`LOOP_PROTOCOL_SMALL`) | modelos <40B | catálogo; deliberadamente plano por el tamaño del modelo |
| `chat_agent/system.py` | pipeline actual | deuda declarada en `UNKNOWN_BASELINE` |
| `reviewer/*.py` | tras el developer | deuda declarada |
| `phases/*.py`, `variants/*.py`, `flows/*.py` | ruta legacy `PhaseEngine` | deuda declarada; sin uso en el pipeline por defecto |

`UNKNOWN_BASELINE` en `tests/test_prompt_style_rules.py` es **deuda declarada,
no aprobación**: la lista existe para que la regla proteja lo nuevo sin esperar
a barrer lo viejo. **Sacar un archivo de esa lista es el arreglo.**

---

## Lo que este documento no dice

No hay evidencia acá de que estos principios mejoren una métrica medida. Son el
criterio con el que está escrito el prompt que mejor funciona en este repo, y
las reglas que un test ya hace cumplir. Un cambio de prompt no tiene forma
barata de validarse en este proyecto: no hay banco de tareas con resultado
esperado. Tenerlo sería la manera de convertir este documento en algo
verificable en vez de argumentado.
