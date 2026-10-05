import { strict as assert } from "node:assert";
import { execFileSync } from "node:child_process";
import { test } from "node:test";

import { App, Validations } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { AwsSolutionsChecks } from "cdk-nag";

import { KiwiServeStack } from "../lib/kiwi-serve-stack";

// La AMI se busca con un lookup (cachedInContext). Dándole la respuesta aquí,
// el test no necesita credenciales y no sale el aviso del valor de relleno.
const context = {
  "ssm:account=123456789012:parameterName=/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-arm64:region=eu-west-1":
    "ami-0123456789abcdef0",
};

function synth(repo = "oscar1223/kiwi") {
  const app = new App({ context });
  const stack = new KiwiServeStack(app, "Test", {
    env: { account: "123456789012", region: "eu-west-1" },
    repo,
    kiwiVersion: "latest",
    instanceType: "t4g.small",
  });
  return { app, stack, template: Template.fromStack(stack) };
}

const { template } = synth();

test("el security group no tiene ninguna regla de entrada", () => {
  for (const [, sg] of Object.entries(template.findResources("AWS::EC2::SecurityGroup"))) {
    assert.equal(sg.Properties.SecurityGroupIngress, undefined);
  }
  template.resourceCountIs("AWS::EC2::SecurityGroupIngress", 0);
});

test("solo sale HTTPS", () => {
  template.hasResourceProperties("AWS::EC2::SecurityGroup", {
    SecurityGroupEgress: [Match.objectLike({ IpProtocol: "tcp", FromPort: 443, ToPort: 443, CidrIp: "0.0.0.0/0" })],
  });
});

test("sin SSH: ni par de claves ni NAT", () => {
  template.hasResourceProperties("AWS::EC2::Instance", { KeyName: Match.absent() });
  template.resourceCountIs("AWS::EC2::NatGateway", 0);
});

test("IMDSv2 obligatorio", () => {
  template.hasResourceProperties("AWS::EC2::LaunchTemplate", {
    LaunchTemplateData: { MetadataOptions: { HttpTokens: "required" } },
  });
});

test("la AMI sale del contexto, no de un parámetro que cambie en cada deploy", () => {
  template.hasResourceProperties("AWS::EC2::Instance", { ImageId: "ami-0123456789abcdef0" });
});

test("disco gp3 cifrado", () => {
  template.hasResourceProperties("AWS::EC2::Instance", {
    InstanceType: "t4g.small",
    BlockDeviceMappings: [Match.objectLike({ Ebs: Match.objectLike({ Encrypted: true, VolumeType: "gp3" }) })],
  });
});

test("el rol solo lee su propio secreto", () => {
  const secretId = Object.keys(template.findResources("AWS::SecretsManager::Secret"))[0];
  const policies = template.findResources("AWS::IAM::Policy");
  const statements = Object.values(policies).flatMap((p) => p.Properties.PolicyDocument.Statement);
  const secretStatements = statements.filter((s) =>
    [s.Action].flat().some((a: string) => a.startsWith("secretsmanager:")),
  );
  assert.ok(secretStatements.length > 0, "el rol debería poder leer el secreto");
  for (const s of secretStatements) {
    assert.deepEqual(s.Resource, { Ref: secretId }, "solo sobre este secreto, nunca sobre *");
    for (const a of [s.Action].flat()) {
      assert.match(a, /^secretsmanager:(GetSecretValue|DescribeSecret)$/);
    }
  }
});

test("Session Manager disponible", () => {
  template.hasResourceProperties("AWS::IAM::Role", {
    ManagedPolicyArns: Match.arrayWith([
      Match.objectLike({ "Fn::Join": Match.arrayWith([Match.arrayWith([Match.stringLikeRegexp("AmazonSSMManagedInstanceCore")])]) }),
    ]),
  });
});

test("el user data instala kiwi como servicio y clona el repo configurado", () => {
  const instance = Object.values(template.findResources("AWS::EC2::Instance"))[0];
  const userData = JSON.stringify(instance.Properties.UserData);
  for (const needle of [
    "useradd --create-home --shell /bin/bash kiwi",
    "systemctl enable --now kiwi",
    "KIWI_REPO=oscar1223/kiwi",
    "KIWI_WORKDIR=/home/kiwi/work/kiwi",
    "ExecStartPre=+/usr/local/bin/kiwi-prepare",
    // Sin -H, sudo puede dejar HOME en /root e instalar kiwi donde el
    // servicio no lo busca.
    "sudo -H -u kiwi",
  ]) {
    assert.ok(userData.includes(needle), `falta en el user data: ${needle}`);
  }
});

// El user data es un Fn::Join de texto y tokens (el ARN del secreto).
function renderUserData(t: Template): string {
  const instance = Object.values(t.findResources("AWS::EC2::Instance"))[0];
  const parts: unknown[] = instance.Properties.UserData["Fn::Base64"]["Fn::Join"][1];
  return parts.map((p) => (typeof p === "string" ? p : "arn:aws:secretsmanager:eu-west-1:123456789012:secret:Env")).join("");
}

test("el user data y los scripts que escribe son bash válido", () => {
  const userData = renderUserData(template);
  // Un error de sintaxis solo se vería al arrancar la instancia.
  execFileSync("bash", ["-n"], { input: userData });

  const prepare = userData.match(/cat > \/usr\/local\/bin\/kiwi-prepare <<'EOF'\n([\s\S]*?)\nEOF\n/);
  assert.ok(prepare, "el user data debería escribir kiwi-prepare");
  execFileSync("bash", ["-n"], { input: prepare[1] });

  assert.match(userData, /cat > \/etc\/systemd\/system\/kiwi.service <<'EOF'\n\[Unit\][\s\S]*?\nEOF\n/);
});

test("sin repo, trabaja en un directorio vacío", () => {
  const { template } = synth("");
  const userData = JSON.stringify(Object.values(template.findResources("AWS::EC2::Instance"))[0].Properties.UserData);
  assert.ok(userData.includes("KIWI_WORKDIR=/home/kiwi/work\\n"));
});

test("cdk-nag (AwsSolutions) pasa, con cada excepción justificada", () => {
  const app = new App({ context });
  new KiwiServeStack(app, "Nag", {
    env: { account: "123456789012", region: "eu-west-1" },
    repo: "oscar1223/kiwi",
    kiwiVersion: "latest",
    instanceType: "t4g.small",
  });
  Validations.of(app).addPlugins(new AwsSolutionsChecks(app));
  // Un hallazgo sin acknowledge hace fallar el synth.
  assert.doesNotThrow(() => app.synth());
});
