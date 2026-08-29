# Experiment profiles

The included profiles follow the four paper settings:

| Profile | Attack | Task | Evaluator |
|---|---|---|---|
| `badnets` | BadNets | targeted refusal | `refusal_keywords` |
| `vpi` | VPI | targeted refusal | `refusal_keywords` |
| `ctba` | CTBA | sentiment steering | `sentiment_steering_keywords` |
| `sleeper` | Sleeper | jailbreak | `jailbreak_keywords` |

Each profile can be used with the model and adapter paths supplied in `.env`.
For another experiment, provide a JSON file through
`LLM_BACKDOOR_EXPERIMENT_CONFIG`:

```json
{
  "experiment_id": "model_attack_task",
  "model_name": "Model name",
  "attack_name": "Attack name",
  "task_name": "Task name",
  "success_evaluator": "package.module:success_masks",
  "method_name": "Defense with Poisoning Again",
  "detector_artifact": "detector.pkl",
  "metrics_file": "metrics.json"
}
```

An evaluator receives generated outputs and returns two Boolean arrays:

```python
def success_masks(outputs):
    return valid_mask, attack_success_mask
```

The evaluator is used only during official evaluation. Detector fitting does
not use attacker triggers, attack targets, poisoned samples, or success labels.
