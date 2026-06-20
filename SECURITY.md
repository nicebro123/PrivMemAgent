# Security Model

## Protected Boundary

The raw dialogue, encryption key, and restored response belong on the trusted
edge device. Cloud memory systems receive placeholderized text only.

The isolation claim is valid only when:

- privacy detection runs on a trusted localhost endpoint;
- the detector completes before any cloud API or telemetry call;
- query text, answer options, and evaluator references pass through the same
  local protection boundary before any remote call;
- the SQLite database and encryption key remain on the trusted device;
- each user receives a distinct `PrivacyStore.namespace`;
- logs do not contain raw detector responses or restored text.

## Storage

Original values are encrypted with Fernet. Equality lookup uses a keyed HMAC;
plaintext values are not SQLite columns. Database and generated key files use
mode `0600`. Production deployments should inject `MEMPRIVACY_STORE_KEY` from
an OS keychain, secret manager, or hardware-backed keystore rather than relying
on a generated file.

## Explicit Remote Opt-In

Non-local detector URLs are rejected by default. Setting
`llm.allow_remote: true` permits remote detection but voids the claim that raw
privacy values never leave the device.

## Out Of Scope

This implementation does not defend against a compromised edge device,
keylogging, malicious local administrators, side-channel attacks, or a cloud
model inferring sensitive attributes from non-redacted context. The benchmark
query helper protects values already known from dialogue annotations; production
deployments still need a local detector for previously unseen query values.
Typed placeholders intentionally reveal privacy category and repeated-reference
linkability within one user namespace.
