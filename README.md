# Veratum Python SDK

**Cryptographic evidence for every AI decision.**

Veratum creates tamper-proof, cryptographically sealed receipts for every AI decision your application makes — automatically. Receipts are Bitcoin-anchored, RFC 3161 timestamped, and independently verifiable by any regulator without requiring a Veratum account.

## Install

```bash
pip install veratum
```

## Quickstart

```python
import veratum
import openai

sdk = veratum.init(api_key="vsk_live_••••••••••••••••")
client = sdk.wrap(openai.OpenAI())

# Every call now creates a tamper-proof receipt automatically.
# EU AI Act Article 12, Colorado SB 24-205, NYC LL144, GDPR — satisfied.
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Your prompt here"}]
)
```

That's it. No other changes to your code.

## Privacy-first mode

```python
# Raw prompts never leave your infrastructure
sdk = veratum.init(api_key="vsk_live_••••••••••••••••", hash_only=True)
```

## MCP Gateway (zero code changes)

```bash
docker run \
  -e VERATUM_API_KEY=vsk_live_•••• \
  -p 8080:8080 \
  veratum/gateway
```

Point your MCP client at `localhost:8080`. Every agent tool call is now evidence.

## What each receipt contains

- **Dual hash:** SHA-256 + SHA-3-256 of decision content
- **Dual signature:** Ed25519 + ML-DSA-65 (quantum-resistant)
- **RFC 3161 timestamp:** DigiCert qualified timestamp (eIDAS compliant)
- **Bitcoin anchor:** OpenTimestamps proof, independently verifiable forever
- **Compliance validation:** 17 regulatory frameworks checked automatically

## Verify receipts offline

```bash
pip install veratum[verify]
veratum-verify receipt.json
```

No internet required. No Veratum account required.

## Documentation

Full documentation at [docs.veratum.ai](https://docs.veratum.ai)

## Compliance coverage

EU AI Act · GDPR · EEOC · CFPB/ECOA · Colorado SB 24-205 · NYC Local Law 144
Illinois AIVA · FINRA 17a-4 · NAIC Model Bulletin · UK DRCF · Canada AIDA
Singapore FEAT · Japan Cabinet AI Guidelines · ISO 42001:2023 · and more

## Issues & support

- Bug reports and feature requests: [GitHub Issues](https://github.com/veratum-ai/sdk-python/issues)
- Enterprise & compliance questions: [ali.ashkir@veratum.ai](mailto:ali.ashkir@veratum.ai)
- Dashboard: [app.veratum.ai](https://app.veratum.ai)

## License

MIT License. See [LICENSE](LICENSE) for details.
