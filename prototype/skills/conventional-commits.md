---
name: conventional-commits
description: Usar cuando el usuario pida ayuda para escribir o revisar un mensaje de commit git.
---

Al escribir un mensaje de commit, sigue el formato Conventional Commits:

    <tipo>(<alcance opcional>): <resumen en imperativo, minúscula, sin punto final>

    <cuerpo opcional: por qué, no qué>

Tipos válidos: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `build`, `ci`.

Reglas:
- El resumen va en imperativo ("añade", no "añadido" ni "añadiendo"), máximo ~72 caracteres.
- El cuerpo explica el *motivo* del cambio, no repite lo que ya dice el diff.
- Un commit = un cambio lógico. Si el usuario describe varios cambios sin relación,
  sugiere dividirlo en varios commits en vez de mezclarlos.
- Si el cambio rompe compatibilidad, añade una línea `BREAKING CHANGE: ...` en el cuerpo.
