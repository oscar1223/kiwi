import { strict as assert } from "node:assert";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";

import { App, Validations } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { AwsSolutionsChecks } from "cdk-nag";

import { KiwiServeStack } from "../lib/kiwi-serve-stack";

// The AMI comes from a lookup (cachedInContext). Giving the answer here means
// the test needs no credentials and the placeholder warning does not show.
const context = {
  "ssm:account=123456789012:parameterName=/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-arm64:region=eu-west-1":
    "ami-0123456789abcdef0",
};

function synth(repo = "oscar1223/kiwi", availabilityZone?: string) {
  const app = new App({ context });
  const stack = new KiwiServeStack(app, "Test", {
    env: { account: "123456789012", region: "eu-west-1" },
    repo,
    kiwiVersion: "latest",
    instanceType: "t4g.small",
    availabilityZone,
  });
  return { app, stack, template: Template.fromStack(stack) };
}

const { template } = synth();

test("the security group has no inbound rules", () => {
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

test("no SSH: no key pair and no NAT", () => {
  template.hasResourceProperties("AWS::EC2::Instance", { KeyName: Match.absent() });
  template.resourceCountIs("AWS::EC2::NatGateway", 0);
});

test("IMDSv2 obligatorio", () => {
  template.hasResourceProperties("AWS::EC2::LaunchTemplate", {
    LaunchTemplateData: { MetadataOptions: { HttpTokens: "required" } },
  });
});

test("the AMI comes from context, not from a parameter that changes on every deploy", () => {
  template.hasResourceProperties("AWS::EC2::Instance", { ImageId: "ami-0123456789abcdef0" });
});

test("disco gp3 cifrado", () => {
  template.hasResourceProperties("AWS::EC2::Instance", {
    InstanceType: "t4g.small",
    BlockDeviceMappings: [Match.objectLike({ Ebs: Match.objectLike({ Encrypted: true, VolumeType: "gp3" }) })],
  });
});

test("the role only reads its own secret", () => {
  const secretId = Object.keys(template.findResources("AWS::SecretsManager::Secret"))[0];
  const policies = template.findResources("AWS::IAM::Policy");
  const statements = Object.values(policies).flatMap((p) => p.Properties.PolicyDocument.Statement);
  const secretStatements = statements.filter((s) =>
    [s.Action].flat().some((a: string) => a.startsWith("secretsmanager:")),
  );
  assert.ok(secretStatements.length > 0, "the role should be able to read the secret");
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

test("the user data installs kiwi as a service and clones the configured repo", () => {
  const rendered = renderUserData(template);
  const userData = [rendered, embeddedFile(rendered, "/etc/systemd/system/kiwi.service")].join("\n");
  for (const needle of [
    "useradd --create-home --shell /bin/bash kiwi",
    "systemctl enable --now kiwi",
    "KIWI_REPO=oscar1223/kiwi",
    "KIWI_WORKDIR=/home/kiwi/work/kiwi",
    "ExecStartPre=+/usr/local/bin/kiwi-prepare",
    // Without -H, sudo may leave HOME at /root and install kiwi where the
    // service does not look for it.
    "sudo -H -u kiwi",
  ]) {
    assert.ok(userData.includes(needle), `missing from the user data: ${needle}`);
  }
});

// The user data is an Fn::Join of text and tokens (the secret ARN).
function renderUserData(t: Template): string {
  const instance = Object.values(t.findResources("AWS::EC2::Instance"))[0];
  const parts: unknown[] = instance.Properties.UserData["Fn::Base64"]["Fn::Join"][1];
  return parts.map((p) => (typeof p === "string" ? p : "arn:aws:secretsmanager:eu-west-1:123456789012:secret:Env")).join("");
}

// Contents of the file the user data writes to dest (base64-encoded).
function embeddedFile(userData: string, dest: string): string {
  const escaped = dest.replace(/[/.]/g, "\\$&");
  const m = userData.match(new RegExp(`base64 -d > ${escaped} <<'EOF'\\n([A-Za-z0-9+/=\\n]*?)\\nEOF\\n`));
  assert.ok(m, `the user data should write ${dest}`);
  return Buffer.from(m[1].replace(/\n/g, ""), "base64").toString("utf8");
}

test("the user data and the scripts it writes are valid bash", () => {
  const userData = renderUserData(template);
  // A syntax error would only show up when the instance boots.
  execFileSync("bash", ["-n"], { input: userData });
  execFileSync("bash", ["-n"], { input: embeddedFile(userData, "/usr/local/bin/kiwi-prepare") });
  assert.match(embeddedFile(userData, "/etc/systemd/system/kiwi.service"), /^\[Unit\]/);
});

test("the user data is ASCII and the files arrive intact", () => {
  // CloudFormation replaces non-ASCII with "?": an accent in the user data
  // breaks the file and makes every diff see a change to the instance.
  const userData = renderUserData(template);
  assert.ok(/^[\x00-\x7f]*$/.test(userData), "the user data has non-ASCII characters");
  for (const [name, dest] of [
    ["kiwi-prepare.sh", "/usr/local/bin/kiwi-prepare"],
    ["kiwi.service", "/etc/systemd/system/kiwi.service"],
  ]) {
    const source = readFileSync(path.join(__dirname, "..", "lib", "instance", name), "utf8");
    assert.equal(embeddedFile(userData, dest), source);
  }
});

test("with no repo, it works in an empty directory", () => {
  const { template } = synth("");
  const userData = JSON.stringify(Object.values(template.findResources("AWS::EC2::Instance"))[0].Properties.UserData);
  assert.ok(userData.includes("KIWI_WORKDIR=/home/kiwi/work\\n"));
});

test("cdk-nag (AwsSolutions) passes, with every exception justified", () => {
  const app = new App({ context });
  new KiwiServeStack(app, "Nag", {
    env: { account: "123456789012", region: "eu-west-1" },
    repo: "oscar1223/kiwi",
    kiwiVersion: "latest",
    instanceType: "t4g.small",
  });
  Validations.of(app).addPlugins(new AwsSolutionsChecks(app));
  // A finding without an acknowledge makes synth fail.
  assert.doesNotThrow(() => app.synth());
});

test("kiwi:az pins the subnet's zone", () => {
  synth().template.hasResourceProperties("AWS::EC2::Subnet", { AvailabilityZone: "dummy1a" });
  const { template } = synth("oscar1223/kiwi", "eu-west-1b");
  template.resourceCountIs("AWS::EC2::Subnet", 1);
  template.hasResourceProperties("AWS::EC2::Subnet", { AvailabilityZone: "eu-west-1b" });
});
