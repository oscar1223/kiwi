import { readFileSync } from "node:fs";
import * as path from "node:path";

import { CfnOutput, RemovalPolicy, Stack, StackProps, Tags, Validations } from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import { Construct } from "constructs";

export interface KiwiServeStackProps extends StackProps {
  /** Repo de GitHub en el que trabaja el agente (owner/name). Vacío: ninguno. */
  readonly repo: string;
  /** Versión de kiwi a instalar ("v0.2.0"), o "latest". */
  readonly kiwiVersion: string;
  /** Tipo de instancia. Tiene que ser arm64 (Graviton). */
  readonly instanceType: string;
}

/** Claves del secreto. Las vacías no se escriben en el .env. */
export const SECRET_KEYS = [
  "KIWI_TELEGRAM_TOKEN",
  "KIWI_TELEGRAM_ALLOWED_USERS",
  "ANTHROPIC_API_KEY",
  "GH_TOKEN",
] as const;

const instanceFile = (name: string) =>
  readFileSync(path.join(__dirname, "instance", name), "utf8");

/**
 * kiwi serve en un EC2 sin puertos de entrada.
 *
 * El bot hace long polling contra Telegram, así que la instancia solo necesita
 * salir por HTTPS: ni SSH, ni IP elástica, ni balanceador. Se entra por SSM
 * Session Manager.
 */
export class KiwiServeStack extends Stack {
  constructor(scope: Construct, id: string, props: KiwiServeStackProps) {
    super(scope, id, props);
    Tags.of(this).add("project", "kiwi");

    // Una subred pública en una sola AZ. Un NAT Gateway costaría más que la
    // instancia y no aporta nada: sin reglas de entrada, la IP pública solo
    // sirve para salir.
    const vpc = new ec2.Vpc(this, "Vpc", {
      maxAzs: 1,
      natGateways: 0,
      subnetConfiguration: [{ name: "public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 }],
    });

    const sg = new ec2.SecurityGroup(this, "Sg", {
      vpc,
      description: "kiwi serve - sin entrada, solo HTTPS de salida",
      allowAllOutbound: false,
    });
    // Telegram, el proveedor del modelo, GitHub, los paquetes y SSM van por
    // 443. El DNS y el NTP de Amazon no pasan por el security group.
    sg.addEgressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), "HTTPS de salida");

    const secret = new secretsmanager.Secret(this, "Env", {
      description: "Variables de entorno de kiwi serve. Rellenar tras el primer deploy.",
      // CDK no deja crear un secreto sin valor: el relleno va en "_placeholder"
      // y kiwi-prepare lo ignora, como a las claves vacías.
      generateSecretString: {
        secretStringTemplate: JSON.stringify(Object.fromEntries(SECRET_KEYS.map((k) => [k, ""]))),
        generateStringKey: "_placeholder",
        excludePunctuation: true,
      },
      // Secrets Manager guarda el secreto borrado 30 días, así que destruir el
      // stack no lo pierde de golpe.
      removalPolicy: RemovalPolicy.DESTROY,
    });

    const repoName = props.repo.split("/").pop() ?? "";
    const workdir = repoName ? `/home/kiwi/work/${repoName}` : "/home/kiwi/work";
    const installUrl =
      props.kiwiVersion === "latest"
        ? "https://raw.githubusercontent.com/oscar1223/kiwi/main/install.sh"
        : `https://raw.githubusercontent.com/oscar1223/kiwi/${props.kiwiVersion}/install.sh`;

    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      "set -euxo pipefail",
      // Herramientas para el agente: git y gh para ramas y PRs, go y make
      // para el repo de kiwi. Go baja la versión que pida go.mod él solo.
      "dnf install -y git make golang python3 'dnf-command(config-manager)'",
      "dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo",
      "dnf install -y gh",

      // Usuario sin sudo, sin contraseña y sin claves SSH.
      "useradd --create-home --shell /bin/bash kiwi",
      // El token de GitHub nunca se escribe en el repo: git lo pide aquí.
      `cat > /home/kiwi/.gitconfig <<'EOF'
[user]
	name = Kiwi (EC2)
	email = kiwi-ec2@users.noreply.github.com
[credential "https://github.com"]
	helper = "!f() { test \\"$1\\" = get && echo username=x-access-token && echo password=$GH_TOKEN; }; f"
EOF`,
      "chown kiwi:kiwi /home/kiwi/.gitconfig",

      `sudo -H -u kiwi env KIWI_VERSION=${props.kiwiVersion === "latest" ? "" : props.kiwiVersion} sh -c 'curl -fsSL ${installUrl} | sh'`,

      "install -d -m 755 /etc/kiwi",
      `cat > /etc/kiwi/serve.conf <<'EOF'
SECRET_ID=${secret.secretArn}
AWS_REGION=${this.region}
KIWI_REPO=${props.repo}
KIWI_WORKDIR=${workdir}
EOF`,
      `cat > /usr/local/bin/kiwi-prepare <<'EOF'\n${instanceFile("kiwi-prepare.sh")}EOF`,
      "chmod 755 /usr/local/bin/kiwi-prepare",
      `cat > /etc/systemd/system/kiwi.service <<'EOF'\n${instanceFile("kiwi.service")}EOF`,
      "systemctl daemon-reload",
      // Sin valores en el secreto el primer arranque falla y systemd lo
      // reintenta cada 10 s: en cuanto se rellena, el bot arranca solo.
      "systemctl enable --now kiwi",
    );

    const instance = new ec2.Instance(this, "Instance", {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      securityGroup: sg,
      instanceType: new ec2.InstanceType(props.instanceType),
      // Fijada en cdk.context.json. Sin esto, cada deploy busca la última AMI
      // y, cuando Amazon publica una, reemplaza la instancia y con ella las
      // sesiones guardadas en el disco. Para actualizarla a propósito:
      // cdk context --reset <clave> y deploy.
      machineImage: ec2.MachineImage.latestAmazonLinux2023({
        cpuType: ec2.AmazonLinuxCpuType.ARM_64,
        cachedInContext: true,
      }),
      requireImdsv2: true,
      ssmSessionPermissions: true,
      userData,
      blockDevices: [
        {
          deviceName: "/dev/xvda",
          volume: ec2.BlockDeviceVolume.ebs(20, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
            deleteOnTermination: true,
          }),
        },
      ],
    });
    secret.grantRead(instance.role);

    new CfnOutput(this, "InstanceId", {
      value: instance.instanceId,
      description: "aws ssm start-session --target <id>",
    });
    new CfnOutput(this, "SecretArn", {
      value: secret.secretArn,
      description: "Rellenar con put-secret-value (ver deploy/aws/README.md)",
    });

    // Cada excepción a cdk-nag (AwsSolutions), con su porqué.
    const acknowledge = (id: string, reason: string) => Validations.of(this).acknowledge({ id, reason });
    acknowledge(
      "AwsSolutions-VPC7",
      "Sin tráfico de entrada que auditar; los flow logs costarían más que lo que vigilan.",
    );
    acknowledge(
      "AwsSolutions-IAM4[Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonSSMManagedInstanceCore]",
      "AmazonSSMManagedInstanceCore es la política que AWS mantiene para Session Manager.",
    );
    acknowledge(
      "AwsSolutions-SMG4",
      "Token de bot, API keys y PAT de terceros: no hay rotación automática posible.",
    );
    acknowledge(
      "AwsSolutions-EC28",
      "Una instancia de uso personal; el monitoring básico de 5 min basta.",
    );
    acknowledge(
      "AwsSolutions-EC29",
      "Se recrea con cdk deploy; la protección de terminación estorbaría para destruirla.",
    );
  }
}
