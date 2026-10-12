import { readFileSync } from "node:fs";
import * as path from "node:path";

import { CfnOutput, RemovalPolicy, Stack, StackProps, Tags, Validations } from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import { Construct } from "constructs";

export interface KiwiServeStackProps extends StackProps {
  /** GitHub repo the agent works in (owner/name). Empty: none. */
  readonly repo: string;
  /** kiwi version to install ("v0.2.0"), or "latest". */
  readonly kiwiVersion: string;
  /** Instance type. Must be arm64 (Graviton). */
  readonly instanceType: string;
  /** Availability zone ("eu-north-1b"). Empty: the first one in the region. */
  readonly availabilityZone?: string;
}

/** Secret keys. Empty ones are not written to the .env. */
export const SECRET_KEYS = [
  "KIWI_TELEGRAM_TOKEN",
  "KIWI_TELEGRAM_ALLOWED_USERS",
  "ANTHROPIC_API_KEY",
  "GH_TOKEN",
] as const;

/**
 * Command that writes the file lib/instance/<name> on the instance. It goes
 * as base64 because CloudFormation replaces non-ASCII in user data with "?":
 * accents arrived broken and, since the stored template never matched the
 * synthesized one, every diff saw a change to the instance.
 */
const writeInstanceFile = (name: string, dest: string) => {
  const b64 = readFileSync(path.join(__dirname, "instance", name)).toString("base64");
  return `base64 -d > ${dest} <<'EOF'\n${b64.replace(/.{76}/g, "$&\n")}\nEOF`;
};

/**
 * kiwi serve on an EC2 instance with no inbound ports.
 *
 * The bot long-polls Telegram, so the instance only needs outbound HTTPS:
 * no SSH, no Elastic IP, no load balancer. Access is through SSM
 * Session Manager.
 */
export class KiwiServeStack extends Stack {
  constructor(scope: Construct, id: string, props: KiwiServeStackProps) {
    super(scope, id, props);
    Tags.of(this).add("project", "kiwi");

    // One public subnet in a single AZ. A NAT Gateway would cost more than the
    // instance and add nothing: with no inbound rules, the public IP is only
    // used to go out.
    // The zone can be pinned because sometimes one runs out of capacity for
    // the instance type. Changing it on a deployed stack replaces the subnet
    // and, with it, the instance.
    const vpc = new ec2.Vpc(this, "Vpc", {
      ...(props.availabilityZone ? { availabilityZones: [props.availabilityZone] } : { maxAzs: 1 }),
      natGateways: 0,
      subnetConfiguration: [{ name: "public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 }],
    });

    const sg = new ec2.SecurityGroup(this, "Sg", {
      vpc,
      description: "kiwi serve - no inbound, outbound HTTPS only",
      allowAllOutbound: false,
    });
    // Telegram, the model provider, GitHub, packages and SSM all go over
    // 443. Amazon's DNS and NTP do not go through the security group.
    sg.addEgressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), "HTTPS de salida");

    const secret = new secretsmanager.Secret(this, "Env", {
      description: "Environment variables for kiwi serve. Fill in after the first deploy.",
      // CDK does not allow a secret without a value: the filler goes in
      // "_placeholder" and kiwi-prepare ignores it, like empty keys.
      generateSecretString: {
        secretStringTemplate: JSON.stringify(Object.fromEntries(SECRET_KEYS.map((k) => [k, ""]))),
        generateStringKey: "_placeholder",
        excludePunctuation: true,
      },
      // Secrets Manager keeps a deleted secret for 30 days, so destroying the
      // stack does not lose it at once.
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
      // Tools for the agent: git and gh for branches and PRs, go and make
      // for the kiwi repo. Go downloads whatever version go.mod asks for.
      "dnf install -y git make golang python3 'dnf-command(config-manager)'",
      "dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo",
      "dnf install -y gh",

      // A user with no sudo, no password and no SSH keys.
      "useradd --create-home --shell /bin/bash kiwi",
      // The GitHub token is never written into the repo: git asks for it here.
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
      writeInstanceFile("kiwi-prepare.sh", "/usr/local/bin/kiwi-prepare"),
      "chmod 755 /usr/local/bin/kiwi-prepare",
      writeInstanceFile("kiwi.service", "/etc/systemd/system/kiwi.service"),
      "systemctl daemon-reload",
      // With no values in the secret the first start fails and systemd
      // retries every 10 s: as soon as it is filled in, the bot starts.
      "systemctl enable --now kiwi",
    );

    const instance = new ec2.Instance(this, "Instance", {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      securityGroup: sg,
      instanceType: new ec2.InstanceType(props.instanceType),
      // Pinned in cdk.context.json. Without this, every deploy looks up the
      // latest AMI and, when Amazon publishes one, replaces the instance and
      // with it the sessions saved on disk. To update it on purpose:
      // cdk context --reset <key> and deploy.
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
      description: "Fill in with put-secret-value (see deploy/aws/README.md)",
    });

    // Every cdk-nag (AwsSolutions) exception, with its reason.
    const acknowledge = (id: string, reason: string) => Validations.of(this).acknowledge({ id, reason });
    acknowledge(
      "AwsSolutions-VPC7",
      "No inbound traffic to audit; flow logs would cost more than what they watch.",
    );
    acknowledge(
      "AwsSolutions-IAM4[Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonSSMManagedInstanceCore]",
      "AmazonSSMManagedInstanceCore is the policy AWS maintains for Session Manager.",
    );
    acknowledge(
      "AwsSolutions-SMG4",
      "Bot token, API keys and third-party PATs: automatic rotation is not possible.",
    );
    acknowledge(
      "AwsSolutions-EC28",
      "A single personal-use instance; basic 5-minute monitoring is enough.",
    );
    acknowledge(
      "AwsSolutions-EC29",
      "It is recreated with cdk deploy; termination protection would get in the way of destroying it.",
    );
  }
}
