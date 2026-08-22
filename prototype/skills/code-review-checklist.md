---
name: code-review-checklist
description: Usar cuando el usuario pida revisar código, un diff o un pull request antes de aprobarlo.
---

Al revisar código, repasa en este orden (para de escalar en severidad):

1. **Correctud**: ¿el código hace lo que dice hacer? Busca off-by-one, condiciones de
   carrera, casos límite (lista vacía, None, valores negativos) sin cubrir.
2. **Seguridad**: inyección (SQL, comandos de shell), datos de usuario sin validar
   cruzando un límite de confianza, secretos hardcodeados.
3. **Reutilización**: ¿ya existe una función/utilidad en el repo que hace esto? Prefiere
   señalar duplicación antes que aprobar código nuevo que reinventa algo existente.
4. **Simplicidad**: ¿hay abstracciones, flags o manejo de errores para casos que no
   pueden pasar? Señálalo, no lo asumas como "buena práctica" sin más.
5. **Estilo**: solo si lo anterior está limpio. No bloquees una revisión por estilo si
   el proyecto no tiene una convención explícita.

Da el feedback como una lista priorizada, primero los bloqueantes (correctud/seguridad),
luego las sugerencias opcionales. No repitas elogios genéricos ("buen trabajo") sin
sustancia.
