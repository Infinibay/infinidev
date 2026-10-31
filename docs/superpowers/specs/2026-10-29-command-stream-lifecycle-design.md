# Brainstorming: ejecución de comandos más fiable

El usuario delegó la elección de mejoras y su implementación. Estas decisiones son
propuestas del implementador basadas en el código actual.

## Oportunidades consideradas

| Área | Mejora | Decisión |
| --- | --- | --- |
| Lógica del agente | Reforzar la evidencia de finalización | Ya hay cambios locales en curso; posponer |
| Búsquedas | Medir y reducir latencia de indexación | Requiere un escenario de medición separado |
| Ejecución de comandos | Evitar espera activa y procesos huérfanos | Implementar |
| Integridad de evidencia | No anunciar capturas incompletas como completas | Implementar en la misma ruta |
| Arquitectura | Unificar los ejecutores interactivo y clásico | Posponer para evitar ampliar el cambio |

## Enfoques

1. Reemplazar ambos ejecutores por un lector común: permitiría cancelación y
   memoria acotada en ambos modos, pero cambia también la semántica del modo clásico.
2. Corregir el ciclo de vida del lector interactivo existente: impacto acotado,
   mantiene los contratos y permite probar los fallos con procesos reales.
3. Ajustar solamente los intervalos de sondeo: no resuelve EOF ni fugas.

Se elige el segundo enfoque por su relación entre impacto, riesgo y verificabilidad.

## Diseño y límites

- Retirar cada pipe del conjunto vigilado al recibir EOF. Si ambos cierran antes
  de que termine el proceso, seguir esperando sin espera activa y conservar
  cancelación y timeout.
- Ante una excepción, terminar el grupo de procesos antes de cerrar los pipes y
  propagar la excepción al manejo existente.
- Registrar qué streams descartaron bytes por el límite de memoria. Sus colas
  siguen apareciendo en el resultado, pero no producen un handle de captura
  completa; los otros streams completos siguen siendo recuperables.
- Conservar permisos, prompts de stdin, códigos de salida, límites y JSON público.
- Python 3.13+, biblioteca estándar y dependencias existentes; sin ajustes globales
  ni cambios a archivos que ya estaban modificados al comenzar.

## Verificación

Usar procesos reales para comprobar EOF de uno o ambos streams, cancelación y
timeout tras EOF, limpieza ante excepciones y capturas completas/parciales por
stream. Ejecutar las pruebas relacionadas y luego `uv run pytest`.
