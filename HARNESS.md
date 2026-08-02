# Infinidev Harness Notes

Procedimientos reutilizables para auditar y validar `infinigpu`. Estas notas separan hechos observados de capacidades todavía no demostradas; no sustituyen los resultados conservados en el repositorio del proyecto.

## Repositorios anidados: comprobar el Git correcto

`infinigpu` es un repositorio Git anidado bajo `/home/andres/lxd/repos/infinigpu`. Ejecutar `git status` o `git diff` desde `/home/andres/lxd/repos` sólo muestra el estado del repositorio superior y puede ocultar por completo cambios del proyecto.

Usar siempre un directorio de trabajo explícito:

```sh
git -C /home/andres/lxd/repos/infinigpu status --short
git -C /home/andres/lxd/repos/infinigpu diff --name-only
git -C /home/andres/lxd/repos/infinigpu diff --check
```

Registrar por separado:

1. el estado del repositorio anidado `infinigpu`;
2. cualquier cambio diagnóstico o de orquestación en el repositorio superior;
3. los artefactos sin seguimiento, sin confundirlos con cambios versionados.

No declarar un parche vacío ni limpio basándose únicamente en el repositorio superior.

## Fuente de verdad para la auditoría

La fuente de verdad actual es:

- `docs/CODE-QUALITY-AUDIT.md` para asuntos abiertos, evidencia y clasificación `PASS`/`FAIL`/`INFRA`;
- `docs/DESKTOP-ACCELERATION-ROADMAP.md` para la matriz por capas de aceleración de escritorio;
- el código y las pruebas de la rama actual para confirmar cualquier afirmación documental.

No usar `docs/3D-COMPLETENESS-ROADMAP.md` de forma aislada para inferir el estado actual: puede contener hitos históricos anteriores a cierres posteriores. Correlacionar siempre IDs, código vigente, historial y resultados ejecutados.

## Gates y evidencia reproducible

La implementación canónica del gate es `scripts/capability_gate/`. No desarrollar en variantes duplicadas con guion o guion bajo como `scripts/capability-gate`, `scripts/a5000-gate` o `scripts/a5000_gate`.

El gate debe ser fail-closed:

- `PASS`: evidencia completa y positiva;
- `FAIL`: evidencia completa que demuestra incumplimiento;
- `INFRA`/`INFRA_ERROR`: falta hardware, VM, toolkit, permisos, artefactos o condiciones necesarias para decidir.

Un código de salida cero del comando no basta. Conservar identidad de VM/GPU/ICD, renderer, resultados de conformidad, actividad real y oráculo gráfico. Los bundles deben tener conjunto cerrado de artefactos y hashes SHA-256 verificables.

Para OpenGL/Zink y escritorio, no marcar `PASS` salvo que la celda correspondiente tenga, como mínimo:

1. identidad del renderer (Zink, sin `llvmpipe`, `softpipe`, `lavapipe`, `swrast` ni otro fallback software);
2. ruta DRM/native-buffer aplicable;
3. actividad GPU observada;
4. un oráculo gráfico (por ejemplo, readback o frames correctos), no sólo retornos exitosos de API.

Qt o cualquier toolkit/hardware ausente permanece `INFRA` o “no demostrado”; la ausencia de laboratorio no es un fallo funcional ni un pase.

## Frontera host frente a guest

Mantener separadas estas clases de evidencia:

- **Host sin privilegios:** sirve para compilación, tests puros, pruebas headless y comprobar que los diagnósticos fallan de forma segura. Un `EACCES` al abrir DRM es `INFRA`, no una prueba de fallo del ICD.
- **Guest con dispositivo infinigpu:** es necesario para demostrar DRM, EGL/Zink y actividad GPU real.
- **WSI/KMS/desktop:** requiere además la ruta de window system/native buffers y un oráculo visible o equivalente. Un pbuffer EGL o WSI headless exitoso no demuestra presentación Wayland/X11/KMS.

Nunca usar éxito WSI/KMS como sustituto de una prueba EGL ni éxito EGL pbuffer como sustituto de WSI/KMS.

## Regresión EGL/A112

El diagnóstico relevante es `guest/diagnostics/egl_device_probe.c`. Su contrato de regresión debe:

- ejecutar controles positivos GLES2 y GLES3;
- ejecutar por separado la solicitud GLES3 robusta que reproduce la frontera de `EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT`;
- distinguir `EGL_BAD_MATCH`, `FAIL` e `INFRA`;
- repetir los casos para detectar resultados no deterministas;
- rechazar renderer software y renderer distinto de Zink;
- realizar render y readback con un color conocido;
- aclarar que `eglSwapBuffers` sobre pbuffer no demuestra window-system/KMS.

La frontera común investigada para A112 pasa por el anuncio `robustBufferAccess` en `guest/icd/infinigpu_physical_device.c`, consumido por Zink/Mesa al decidir si puede satisfacer el contexto EGL robusto. No añadir condiciones específicas para Firefox, toolkit o proveedor. Mantener ABI 0.36 salvo una revisión coordinada y explícitamente justificada.

A112 sólo puede cerrarse cuando la solicitud EGL mínima deja de producir el `EGL_BAD_MATCH` original **y** Firefox/WebRender completa su propia celda de evidencia de escritorio. Una sonda EGL positiva por sí sola no cierra Firefox/WebRender.

## Secuencia práctica de validación

1. Capturar `git status --short`, `git diff --name-only` y `git diff --check` dentro del repositorio anidado.
2. Compilar el diagnóstico EGL con warnings tratados como errores.
3. Ejecutar tests unitarios del gate canónico.
4. Ejecutar gates rápidos y regresiones existentes en host; clasificar permisos/hardware ausentes como `INFRA`.
5. Ejecutar la sonda y el gate dentro del guest con el nodo infinigpu y Zink reales.
6. Rechazar explícitamente cualquier fallback software.
7. Ejecutar matrices de escritorio por toolkit y window system sin inferir una capa desde otra.
8. Actualizar la auditoría sólo con resultados observados, comandos, fechas/entorno y artefactos.
9. Volver a revisar el diff desde el repositorio anidado antes de declarar cierre.

No documentar `PASS` basándose en “debería funcionar”, compilación aislada, presencia de paquetes, extensiones anunciadas o resultados de una capa distinta.
