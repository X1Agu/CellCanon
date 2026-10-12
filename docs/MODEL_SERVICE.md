# API configuration

The `compile` and `explain` stages require a JSON chat completion service.
No API provider, endpoint, deployment, or credential is supplied. These settings
are blank in the example configuration and must be filled in locally.
Copy `config.example.json` to a private configuration file and fill in:

| Field | Value |
| --- | --- |
| `model_service_endpoint` | Final HTTPS completion URL, without URL credentials, query or fragment |
| `language_model` | Your model or deployment identifier |
| `model_service_credential_env` | Name of the environment variable containing your API key |
| `api_rates_per_million.input` | Input-token rate per million tokens |
| `api_rates_per_million.output` | Output-token rate per million tokens |
| `api_budget` | Positive budget per stage, in the same unit as both rates |
| `model_request_options` | Additional JSON request fields required by your service |

For example, set `model_service_credential_env` to `CELLCANON_API_KEY` and enter
the key in Bash without writing it to the configuration file or shell history:

```bash
read -r -s -p "API key: " CELLCANON_API_KEY
export CELLCANON_API_KEY
```

## Service contract

Requests use `Authorization: Bearer ...` and the JSON fields `model`,
`messages`, `temperature: 0`, `max_tokens`,
`response_format: {"type": "json_object"}` and `stream: false`.
`model_request_options` cannot override these reserved fields.

Responses must provide `choices[].message.content` as a JSON string, plus
nonnegative integer `usage.prompt_tokens` and `usage.completion_tokens`.
Choose a service that supports this contract. Use the paper's model and
inference settings when reproducing its experiments; other models may give
different results.

HTTP redirects are rejected. Supply the final HTTPS endpoint directly.

## Budget and outputs

Calls reserve budget before sending a request. Failed calls with unknown
usage retain their reservation. Compilation and explanation have separate
budgets; accounting uses the rates you supply.

Responses are cached for matching requests and service configurations.
Run outputs retain scientific prompts and responses, without credentials,
connection fields or service error bodies. Keep outputs outside the repository.
