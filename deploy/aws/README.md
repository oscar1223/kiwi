# kiwi serve en AWS

Un stack de CDK que deja `kiwi serve` corriendo en un EC2, disponible aunque tu
portátil esté apagado.

```
Telegram  <── long polling (HTTPS, sale la instancia) ──  EC2 t4g.small
                                                           ├─ systemd: kiwi serve
Secrets Manager ── token, IDs, API key, GH_TOKEN ───────>  ├─ ~/.config/kiwi/.env (0600)
                                                           └─ ~/work/<repo>  ── PRs ──> GitHub
Tú ── SSM Session Manager (sin SSH) ───────────────────>
```

- **Sin puertos de entrada.** El security group no tiene ninguna regla de entrada y
  solo deja salir HTTPS. No hay SSH ni par de claves: se entra por Session Manager.
- **Sin NAT Gateway.** La instancia va en una subred pública con IP pública. Como no
  admite tráfico de entrada, la IP solo sirve para salir, y un NAT costaría más que la
  propia instancia.
- **Secretos fuera del disco del repo.** Viven en Secrets Manager. El rol de la
  instancia solo puede leer ese secreto, y en cada arranque el servicio los vuelca al
  `.env` de Kiwi con permisos 0600.
- **Usuario `kiwi` sin sudo,** con un servicio `systemd` que se reinicia solo.

## Coste aproximado

Unos 15-20 $/mes, sin contar lo que gastes en el modelo:

- `t4g.small`
- 20 GB de gp3
- la IP pública (AWS la cobra aparte)
- el secreto de Secrets Manager

Las cifras exactas dependen de la región: compruébalas en la calculadora de AWS antes
de desplegar.

## Antes de empezar

1. **Una release con `kiwi serve`.** La instancia instala la última release, y la v0.1.3
   todavía no trae el comando.
2. **Un bot de Telegram solo para el EC2.** Créalo con @BotFather. Telegram solo deja a
   un proceso leer cada token: si tu `kiwi-dev serve` local usa el mismo bot, los dos
   chocan con un `409 Conflict`.
3. **Un token de GitHub de grano fino** (*fine-grained*), limitado al repo o repos en
   los que vaya a trabajar. Permisos: *Contents* y *Pull requests* en lectura y
   escritura. Nada más.
4. **CDK con la cuenta inicializada** (`npx cdk bootstrap`), una sola vez por cuenta
   y región.

## Desplegar

```sh
cd deploy/aws
npm ci
npm test               # tests de la plantilla, sin AWS
npx cdk diff           # qué se va a crear
npx cdk deploy
```

El primer `synth` busca la AMI de Amazon Linux 2023 y la guarda en `cdk.context.json`.
**No borres ese fichero.** Si lo pierdes, el siguiente deploy coge una AMI más nueva y
**reemplaza la instancia**, y con ella se van las sesiones guardadas. Está en
`.gitignore` porque la clave incluye el ID de tu cuenta y este repo es público.

Para cambiar el repo de trabajo, la versión o el tipo de instancia:

```sh
npx cdk deploy -c kiwi:repo=oscar1223/otro -c kiwi:version=v0.2.0
```

El user data solo se ejecuta en el primer arranque, así que estos cambios solo se
aplican en una instancia nueva.

## Rellenar el secreto

El deploy crea el secreto vacío; su ARN aparece en la salida `SecretArn`. Rellénalo
desde un fichero, para que los valores no queden en el historial de la shell:

```sh
cat > /tmp/kiwi-secret.json <<'EOF'
{
  "KIWI_TELEGRAM_TOKEN": "123456:ABC...",
  "KIWI_TELEGRAM_ALLOWED_USERS": "11111111",
  "ANTHROPIC_API_KEY": "sk-ant-...",
  "GH_TOKEN": "github_pat_..."
}
EOF
aws secretsmanager put-secret-value --secret-id <SecretArn> --secret-string file:///tmp/kiwi-secret.json
rm /tmp/kiwi-secret.json
```

- **Cualquier otra clave** que pongas en el secreto acaba también en el `.env`. Sirve,
  por ejemplo, para la clave de otro proveedor de modelos, o para `KIWI_TZ` si tus
  tareas de `/cron` no van en hora de Madrid (la instancia está en UTC).
- **Mientras falten el token o la lista blanca,** el servicio falla al arrancar y
  systemd lo reintenta cada 10 s. En cuanto los pones, arranca solo.
- **Si cambias un valor más tarde,** basta con `sudo systemctl restart kiwi`.

## Operar

```sh
# entrar en la instancia
aws ssm start-session --target <InstanceId>

# dentro
sudo systemctl status kiwi
sudo journalctl -u kiwi -f        # cada tool call y lo que se bloquea

# actualizar kiwi a la última release
sudo -H -u kiwi ~/.local/bin/kiwi update && sudo systemctl restart kiwi
```

Lo que haga el agente queda en `/home/kiwi/work/<repo>`. Lo que quieras traerte, que
lo suba como rama y abra un PR: no aparece en tu disco.

## Quitarlo

```sh
npx cdk destroy
```

Se borran la instancia, su disco y la red. El secreto pasa a estar programado para
borrarse: Secrets Manager lo guarda 30 días por si te arrepientes.
