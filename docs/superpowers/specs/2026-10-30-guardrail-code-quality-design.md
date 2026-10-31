# Calidad interna del cierre y corrección del engine

## Objetivo y autoridad

El usuario pidió aplicar los skills de libros para mejorar sustancialmente la
calidad del código. La sesión mantiene la delegación de decisiones y ejecución.
Esta ronda cambia código de producción, no los prompts ni los perfiles personales.

El objetivo es poder entender y cambiar la validación sin reconstruir a la vez
los detalles del modelo, herramientas, presupuestos y caché. Se preserva el
comportamiento observable de la corrección y las mejoras de rondas anteriores.

## Diagnóstico

`guardrail_runner.evaluate_guardrail` tiene 253 líneas y 45 nodos de control de
flujo (If/For/While/ExceptHandler/IfExp/BoolOp contados mediante AST). Mezcla:

- aceptación y tipo del retorno del validador;
- cantidad de correcciones y revalidación del último candidato;
- contexto por intento y protocolo FC/manual;
- rondas de modelo y uso de tokens;
- presupuestos por intento, por ejecución y fuse acumulativo;
- batch terminal, resultados de tools omitidas y bloqueo explícito;
- ejecución de tools, tracking de archivos y refresh de caché.

El problema es el número de decisiones simultáneas y su anidamiento, no solo
la longitud. La base tiene regresiones de contexto, cancelación, protocolos,
presupuestos y finalización que permiten refactorizar con evidencia.

## Fuentes y decisiones

- Ousterhout: una operación semántica que oculta los detalles, sin helpers de
  reenvío ni una jerarquía nueva.
- Fowler y Feathers: caracterizar combinaciones de límites antes de mover código;
  separar transformación estructural de cambios funcionales.
- Clean Architecture: política de validación separada del mecanismo de corrección,
  con la frontera más ligera que permite probarlos.
- Clean Code y Code Complete: estado con un dueño, nombres explícitos y flujo local.
- Release It!: preservar límites, cancelación y accounting en fallos parciales.

## Alternativas

1. Extraer funciones con muchos parámetros compartidos: acorta el cuerpo pero
   mantiene al llamador responsable de flags, contadores y secuencias. Se descarta.
2. Reutilizar el loop completo del developer: concentra más código, pero mezcla
   estados de Task/Step con propuestas correctivas y amplía mucho el cambio.
3. Una sesión privada de corrección: el evaluador conserva la política y una
   instancia por evaluación posee los detalles operativos. Es la decisión del
   implementador, compatible con la delegación del usuario.

## Diseño elegido

Todo permanece en `engine/loop/guardrail_runner.py`. No se crean paquetes de una
sola implementación ni nuevas dependencias circulares. Los aliases existentes
de llamada al modelo y dispatch siguen siendo seams de integración; no se añade
un argumento público para satisfacer tests.

### Política pública

`evaluate_guardrail` conserva su firma y `GuardrailOutcome`. Su cuerpo comprueba
cancelación, valida estrictamente bool o tupla (bool, feedback), decide aceptación
o rechazo, solicita una corrección y revalida el candidato final. Conserva el catch
de fallo que entrega texto parcial con estado failed y la prioridad de cancelación.

`apply_guardrail` conserva íntegros firma, retorno string, copia superficial del
contexto y estado compartido. Un caller legacy recibe el último texto aun cuando
la evaluación tipada no sea done.

### Mecanismo privado

`_CorrectionSession(ctx, *, hooks, build_context_messages, run_tool, is_cancelled)`
posee el contexto del run y el número acumulado de tools de corrección.
Su única operación de uso por la política es:

`run(result: str, feedback: Any, *, attempt: int) -> _CorrectionResult`.

`_CorrectionResult` es inmutable, con texto, estado
`candidate | blocked | exhausted | cancelled | not_started`, y motivo. `candidate` significa
propuesta aún no validada, nunca éxito de la Task. Los helpers de la sesión
construyen contexto fresco, normalizan llamadas, procesan el batch y ejecutan
tools con sus efectos observables. Se eligen por responsabilidad, no por tamaño.

La sesión reinicia contadores y allowance del intento al comenzar `run`; conserva
el contador acumulado hasta terminar la evaluación. Es privada y de uso secuencial
por un único evaluador; ninguna fase de prepare/finalize queda expuesta al caller.
Los helpers de presupuestos dejan de recibir ctx y su propio state por separado.

`not_started` distingue un presupuesto ya agotado antes de comenzar: devuelve
exhausted sin volver a invocar el validador. Un intento que sí comenzó y agotó
un fuse devuelve exhausted con su candidato, que se revalida como antes.

## Invariantes que no cambian

- Último candidato revalidado incluso al agotar reintentos/tokens/tools/rondas.
- Aceptación del candidato gana al agotamiento que produjo ese candidato.
- Bool False no reintenta; tupla False puede corregir. Retornos inválidos fallan.
- Fuse de 16 tools para configuración sin límite; 8 rondas de modelo por intento.
- Per-action se renueva por corrección; límite global y contador de LoopState no.
- Un step_complete blocked/explore sigue blocked aunque el formato sea válido.
- Todos los call IDs no cancelados reciben resultado; trailing/over-budget tools
  se responden sin ejecutar. Un terminal vacío termina el intento.
- FC conserva reasoning/campos opacos; manual no envía schemas ni roles tool.
- Tool accounting ocurre antes del dispatch. Captura, hooks, archivos y caché
  conservan orden, incluso refresh tras shell que falla después de modificar.
- Nuevas instrucciones/contexto se consultan por intento mediante el callback ya
  existente. Cancelación cooperativa y scopes foreground siguen en el engine.
- Excepciones conservan el último candidato y el mismo mensaje/log de diagnóstico.

## Validación y límites

Añadir caracterización de presupuestos entre intentos, aislamiento entre
evaluaciones, éxito tras batch que agotó tools y fallo después de un candidato
parcial. Estas pruebas deben pasar antes de refactorizar: describen comportamiento
existente, no una funcionalidad inventada. Probar que detectan mutaciones del
contador/límite en procesos temporales sin alterar el checkout.

Ejecutar los tests actuales sin debilitar asserts ni migrar mocks a helpers nuevos.
Comparar escenarios deterministas entre snapshot anterior y código nuevo como
comprobación adicional, manteniendo outcomes, secuencia de validación, calls,
contadores y transcripciones. Una medida AST antes/después documentará la reducción
de complejidad por función, sin presentarla como prueba de corrección o rendimiento.

No refactorizar router, UI, loop principal, esquemas, prompts o persistencia en esta
ronda. Preservar cambios locales previos. Sin commits ni pushes. Ejecutar revisión
independiente y la suite completa con `uv run --frozen --no-sync pytest`.
