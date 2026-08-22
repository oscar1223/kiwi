#!/usr/bin/env python3
"""Comandos interactivos /model /config /mcp /skill /memory de la TUI de
Kiwi: cada uno es una corutina que navega por menús modales reales
(OptionMenu/TextPrompt en ui_widgets.py) con flechas y Enter, en vez de
esperar que el usuario escriba comandos literales en el chat.

Cada función recibe la instancia de KiwiApp (`app`) y usa
`await app.push_screen_wait(...)` para mostrar menús/cuadros de texto, y
`app.write_system`/`app.write_error` para dejar constancia de resultados en
el historial de chat.
"""

import os

import agent
from ui_widgets import OptionMenu, TextPrompt, confirm

NEW_PROFILE = "__new__"
DELETE_PROFILE = "__delete__"
NEW_ENV_VAR = "__new__"
NEW_MCP_SERVER = "__new__"
NEW_SKILL = "__new__"


# --- /model ---


async def model_flow(app) -> None:
    while True:
        cfg = agent.load_model_config()
        names = sorted(cfg["profiles"])
        options = []
        for name in names:
            profile = cfg["profiles"][name]
            marker = "→ " if name == cfg["current"] else "  "
            env_name = agent.profile_api_key_env(profile)
            warn = f"  ⚠ falta {env_name}" if env_name and not os.getenv(env_name) else ""
            options.append((f"{marker}{name} — {profile['provider']}/{profile['model']}{warn}", name))
        options.append(("+ Nuevo perfil de modelo", NEW_PROFILE))
        if len(names) > 1:
            options.append(("🗑 Eliminar un perfil", DELETE_PROFILE))

        choice = await app.push_screen_wait(OptionMenu(f"Modelo /model — actual: {agent.current_model_label()}", options))
        if choice is None:
            return

        if choice == NEW_PROFILE:
            await _new_model_profile(app)
            continue

        if choice == DELETE_PROFILE:
            deletable = [n for n in names if n != cfg["current"]]
            target = await app.push_screen_wait(OptionMenu("Eliminar perfil de modelo", [(n, n) for n in deletable]))
            if target is None:
                continue
            try:
                agent.remove_model_profile(target)
                app.write_system(f"Perfil de modelo '{target}' eliminado.")
            except ValueError as exc:
                app.write_error(str(exc))
            continue

        agent.set_current_model(choice)
        app._rebuild_executor()


async def _new_model_profile(app) -> None:
    providers = list(agent.PROVIDER_INFO)
    provider = await app.push_screen_wait(
        OptionMenu("Proveedor", [(agent.PROVIDER_INFO[p]["label"], p) for p in providers])
    )
    if provider is None:
        return

    model = await app.push_screen_wait(TextPrompt("Nombre del modelo del proveedor:", placeholder="p. ej. gpt-4o, deepseek-chat"))
    if not model:
        return

    base_url = api_key_env = None
    if agent.PROVIDER_INFO[provider]["custom_base_url"]:
        base_url = await app.push_screen_wait(TextPrompt("URL base de la API:", placeholder="https://api.deepseek.com/v1"))
        if base_url is None:
            return
        api_key_env = await app.push_screen_wait(TextPrompt("Variable de entorno para la API key:", placeholder="DEEPSEEK_API_KEY"))
        if api_key_env is None:
            return

    alias = await app.push_screen_wait(TextPrompt("Alias para este perfil:", default=model))
    if alias is None:
        return
    alias = alias.strip() or model

    try:
        agent.add_model_profile(alias, provider, model, base_url=base_url, api_key_env=api_key_env)
    except ValueError as exc:
        app.write_error(str(exc))
        return
    app.write_system(f"Perfil '{alias}' guardado.")

    if await confirm(app, f"¿Cambiar al modelo '{alias}' ahora?"):
        agent.set_current_model(alias)
        app._rebuild_executor()


# --- /config ---


async def config_flow(app) -> None:
    while True:
        keys = agent.list_env_keys()
        options = [(f"{key} = {agent.get_masked_env(key)}", key) for key in keys]
        options.append(("+ Nueva variable", NEW_ENV_VAR))

        choice = await app.push_screen_wait(OptionMenu("Variables de entorno /config (.env)", options))
        if choice is None:
            return

        if choice == NEW_ENV_VAR:
            key = await app.push_screen_wait(TextPrompt("Nombre de la variable:", placeholder="QWEN_API_KEY"))
            if not key:
                continue
            key = key.strip()
        else:
            key = choice

        action = await app.push_screen_wait(
            OptionMenu(f"{key} = {agent.get_masked_env(key)}", [("Editar valor", "edit"), ("Borrar variable", "delete"), ("Cancelar", "cancel")])
        )
        if action in (None, "cancel"):
            continue

        if action == "delete":
            agent.unset_env_value(key)
            app.write_system(f"{key} eliminada de .env.")
        else:
            value = await app.push_screen_wait(TextPrompt(f"Nuevo valor para {key}:"))
            if value is None:
                continue
            agent.set_env_value(key, value.strip())
            app.write_system(f"{key} actualizada.")

        profile = agent.get_current_profile()
        if agent.profile_api_key_env(profile) == key:
            app._rebuild_executor()


# --- /mcp ---


async def mcp_flow(app) -> None:
    while True:
        config = agent.read_mcp_config()
        names = sorted(config)
        options = [
            (f"{name} — {config[name].get('command', '')} {' '.join(config[name].get('args', []))}", name)
            for name in names
        ]
        options.append(("+ Añadir servidor MCP", NEW_MCP_SERVER))

        choice = await app.push_screen_wait(OptionMenu("Servidores MCP /mcp", options))
        if choice is None:
            return

        if choice == NEW_MCP_SERVER:
            await _new_mcp_server(app)
            continue

        action = await app.push_screen_wait(OptionMenu(f"Servidor '{choice}'", [("Eliminar", "delete"), ("Cancelar", "cancel")]))
        if action == "delete" and await confirm(app, f"¿Eliminar el servidor MCP '{choice}'?"):
            agent.remove_mcp_server(choice)
            app.write_system(f"Servidor MCP '{choice}' eliminado.")
            app._rebuild_executor()


async def _new_mcp_server(app) -> None:
    name = await app.push_screen_wait(TextPrompt("Nombre del servidor:"))
    if not name:
        return
    command = await app.push_screen_wait(TextPrompt("Comando a ejecutar:", placeholder="npx"))
    if not command:
        return
    args_raw = await app.push_screen_wait(TextPrompt("Argumentos separados por espacio:", placeholder="-y @modelcontextprotocol/server-filesystem /ruta"))
    if args_raw is None:
        return
    env_raw = await app.push_screen_wait(TextPrompt("Variables de entorno (opcional):", placeholder="CLAVE=valor,CLAVE2=valor2"))
    if env_raw is None:
        return

    env = None
    if env_raw.strip():
        env = {}
        for pair in env_raw.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                env[k.strip()] = v.strip()

    agent.add_mcp_server(name.strip(), command.strip(), args_raw.split(), env=env)
    app.write_system(f"Servidor MCP '{name}' guardado. Reconectando...")
    app._rebuild_executor()


# --- /skill ---


async def skill_flow(app) -> None:
    while True:
        skills = agent.load_skills()
        names = sorted(skills)
        options = [(f"{name}: {skills[name]['description']}", name) for name in names]
        options.append(("+ Nueva skill", NEW_SKILL))

        choice = await app.push_screen_wait(OptionMenu("Skills /skill", options))
        if choice is None:
            return

        if choice == NEW_SKILL:
            await _new_skill(app)
            continue

        action = await app.push_screen_wait(
            OptionMenu(f"Skill '{choice}'", [("Ver contenido", "view"), ("Eliminar", "delete"), ("Cancelar", "cancel")])
        )
        if action == "view":
            body = skills[choice]["body"]
            if len(body) > agent.MAX_OUTPUT:
                body = body[: agent.MAX_OUTPUT] + f"\n... [truncado, {len(body)} chars total]"
            app.write_system(f"--- {choice} ---\n{body}")
        elif action == "delete" and await confirm(app, f"¿Eliminar la skill '{choice}'?"):
            agent.delete_skill(choice)
            app.write_system(f"Skill '{choice}' eliminada.")
            app._rebuild_executor()


async def _new_skill(app) -> None:
    name = await app.push_screen_wait(TextPrompt("Nombre de la skill (kebab-case):", placeholder="mi-skill"))
    if not name:
        return
    description = await app.push_screen_wait(TextPrompt("Descripción breve (cuándo usarla):"))
    if description is None:
        return
    raw = await app.push_screen_wait(
        TextPrompt("Contenido: ruta a un .md existente para importarlo, o texto directo:")
    )
    if raw is None:
        return

    expanded = os.path.expanduser(raw.strip())
    if os.path.exists(expanded):
        with open(expanded, "r") as f:
            body = f.read()
    else:
        body = raw

    path = agent.save_skill(name.strip(), description.strip(), body)
    app.write_system(f"Skill guardada en {path}.")
    app._rebuild_executor()


# --- /memory ---


async def memory_flow(app) -> None:
    while True:
        turns = len(app.chat_history) // 2
        tokens = agent._estimate_tokens(app.chat_history)
        choice = await app.push_screen_wait(
            OptionMenu(
                f"Memoria /memory — {turns} turnos (~{tokens} tokens)",
                [("Ver últimos turnos", "view"), ("Borrar memoria", "clear"), ("Cerrar", "close")],
            )
        )
        if choice in (None, "close"):
            return

        if choice == "view":
            if turns == 0:
                app.write_system("No hay turnos guardados.")
                continue
            raw_n = await app.push_screen_wait(TextPrompt("¿Cuántos turnos ver?", default="5"))
            if raw_n is None:
                continue
            n = int(raw_n) if raw_n.strip().isdigit() else 5
            for msg in app.chat_history[-(n * 2):]:
                if msg.type == "human":
                    app.write_user(msg.content)
                else:
                    app.write_kiwi(msg.content)
            continue

        if choice == "clear":
            if await confirm(app, "¿Seguro que quieres borrar toda la memoria de conversación?"):
                app.chat_history = []
                agent.save_history([])
                app.write_system("Memoria borrada.")
