Preserved AE policy snapshot for the Docker image tagged `overflow-ae:azbasev4`.

Docker reference:
- Local tag: `overflow-ae:azbasev4`
- Remote tag: `asia-southeast1-docker.pkg.dev/til-ai-2026/repo-til-26-overflow/overflow-ae:azbasev4`
- Image ID: `sha256:ce4c03bc0b0b4029f98b3febd93e521740db5b9f19343c5dc3fe2e4e8785434c`

This is the pre-v5 candidate: it points `ae_manager.py` at
`berserker_base_policy.BerserkerBasePolicy`, and that policy adds the collected
tile cooldown / dead-base filtering on top of the restored azbase stack. It does
not include the later v5 emergency-defense and position-churn changes.

Source hash checks from the Docker image:
- `ae_manager.py`: `8dcb7f5ade871c87200324957a14e190b1fb27116c760c6026b1b4ff84cabb43`
- `berserker_base_policy.py`: `93506d9a1109f1f3cbd44015248c070fba2dd9c8cac9675c9ed4a5c931d0ebbf`
- `azbase_preserved/edited_policy.py`: `227db2f2f4368966e39bb69fcd440bb59e90cc44d4abb0347feb8ec6456d080a`
- `azbase_preserved/edited_policy_v2.py`: `943f6f5b29d31de1a86717c5b8d0e1e3cb728974c79197adc31355293ec0e55a`
- `azbase_preserved/berserker_base_azbase_policy.py`: `f3a53438c702df0964769943c0500ad4fc99c4e4d28cf58cf0826bf949a51963`


Score notes from evaluator submissions:
- 0.718
- 0.724