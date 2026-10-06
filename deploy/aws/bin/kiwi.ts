#!/usr/bin/env node
import { App, Validations } from "aws-cdk-lib";
import { AwsSolutionsChecks } from "cdk-nag";

import { KiwiServeStack } from "../lib/kiwi-serve-stack";

const app = new App();

new KiwiServeStack(app, "KiwiServe", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
  description: "kiwi serve - bot de Telegram en un EC2 sin puertos de entrada",
  repo: app.node.tryGetContext("kiwi:repo") ?? "",
  kiwiVersion: app.node.tryGetContext("kiwi:version") ?? "latest",
  instanceType: app.node.tryGetContext("kiwi:instanceType") ?? "t4g.small",
  availabilityZone: app.node.tryGetContext("kiwi:az"),
});

Validations.of(app).addPlugins(new AwsSolutionsChecks(app, { verbose: true }));
