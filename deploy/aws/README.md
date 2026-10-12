# kiwi serve on AWS

A CDK stack that leaves `kiwi serve` running on an EC2 instance, available even when
your laptop is off.

```
Telegram  <── long polling (HTTPS, outbound from the instance) ── EC2 t4g.small
                                                                  ├─ systemd: kiwi serve
Secrets Manager ── token, IDs, API key, GH_TOKEN ──────────────>  ├─ ~/.config/kiwi/.env (0600)
                                                                  └─ ~/work/<repo>  ── PRs ──> GitHub
You ── SSM Session Manager (no SSH) ───────────────────────────>
```

- **No inbound ports.** The security group has no inbound rules and only lets
  HTTPS out. There is no SSH and no key pair: you get in through Session Manager.
- **No NAT Gateway.** The instance sits in a public subnet with a public IP. Since it
  accepts no inbound traffic, the IP is only used to go out, and a NAT would cost more
  than the instance itself.
- **Secrets stay off the repo's disk.** They live in Secrets Manager. The instance role
  can only read that secret, and on every start the service dumps it into Kiwi's
  `.env` with mode 0600.
- **A `kiwi` user without sudo,** with a `systemd` service that restarts itself.

## Approximate cost

About $15-20/month, not counting what you spend on the model:

- `t4g.small`
- 20 GB of gp3
- the public IP (AWS bills it separately)
- the Secrets Manager secret

Exact figures depend on the region: check them in the AWS pricing calculator before
deploying.

## Before you start

1. **A release with `kiwi serve`.** The instance installs the latest release, and v0.1.3
   does not ship the command yet.
2. **A Telegram bot just for the EC2 instance.** Create it with @BotFather. Telegram only
   lets one process read each token: if your local `kiwi-dev serve` uses the same bot,
   both collide with a `409 Conflict`.
3. **A fine-grained GitHub token**, limited to the repo or repos it will work in.
   Permissions: *Contents* and *Pull requests*, read and write. Nothing else.
4. **CDK with the account bootstrapped** (`npx cdk bootstrap`), once per account and
   region.

## Deploy

```sh
cd deploy/aws
npm ci
npm test               # template tests, no AWS needed
npx cdk diff           # what will be created
npx cdk deploy
```

The first `synth` looks up the Amazon Linux 2023 AMI and stores it in `cdk.context.json`.
**Do not delete that file.** If you lose it, the next deploy picks a newer AMI and
**replaces the instance**, taking the saved sessions with it. It is in `.gitignore`
because the key includes your account ID and this repo is public.

To change the working repo, the version or the instance type:

```sh
npx cdk deploy -c kiwi:repo=oscar1223/other -c kiwi:version=v0.2.0
```

User data only runs on the first boot, so these changes only apply to a new instance.

### If a zone runs out of capacity

By default the instance goes in the region's first zone. If the deploy fails with
`We currently do not have sufficient t4g.small capacity in the Availability Zone`, pick
another one from those the message suggests. Store it in `cdk.context.json`, next to the
AMI, so you do not have to pass it on every deploy:

```json
"kiwi:az": "eu-north-1b"
```

A failed first deploy leaves the stack in `ROLLBACK_COMPLETE`, with no resources. The
next `cdk deploy` deletes it and creates it again. **Changing the zone of a stack that
already works replaces the instance**, just like changing the AMI.

## Fill in the secret

The deploy creates the secret empty; its ARN shows up in the `SecretArn` output. Fill it
from a file, so the values do not end up in your shell history:

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

- **Any other key** you put in the secret also ends up in the `.env`. That works, for
  example, for another model provider's key, or for `KIWI_TZ` if your `/cron` tasks
  should not run on Madrid time (the instance is on UTC).
- **While the token or the allowlist is missing,** the service fails to start and
  systemd retries every 10 s. As soon as you set them, it starts on its own.
- **If you change a value later,** `sudo systemctl restart kiwi` is enough.

## Operate

```sh
# get into the instance
aws ssm start-session --target <InstanceId>

# inside
sudo systemctl status kiwi
sudo journalctl -u kiwi -f        # every tool call and whatever gets blocked

# update kiwi to the latest release (full path: a ~ would be expanded by
# your shell, ec2-user's, before sudo switches user)
sudo -H -u kiwi /home/kiwi/.local/bin/kiwi update && sudo systemctl restart kiwi
```

Whatever the agent does stays in `/home/kiwi/work/<repo>`. Anything you want to bring
back, have it push as a branch and open a PR: it does not show up on your disk.

## Tear it down

```sh
npx cdk destroy
```

The instance, its disk and the network are deleted. The secret is scheduled for
deletion: Secrets Manager keeps it for 30 days in case you change your mind.
