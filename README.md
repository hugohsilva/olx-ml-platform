# OLX ML Platform

Plataforma local de gerenciamento do ciclo de vida de modelos de Machine Learning, desenvolvida
como case técnico para uma posição de Staff MLOps Engineer.

## Requisitos locais

- Python 3.11 ou superior
- [uv](https://docs.astral.sh/uv/)
- Docker instalado e em execução para as fases que constroem ou executam containers

## Configuração de desenvolvimento

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

O comando `mlplatform` é registrado durante a instalação. As operações de ciclo de vida serão
implementadas nas fases seguintes; consulte `specs/001-local-ml-lifecycle/quickstart.md` para o
fluxo planejado. Ele só deve ser tratado como executável quando a implementação correspondente
estiver concluída.

## Configuração Docker local

A configuração usa `MLPLATFORM_HOME` (padrão `~/.mlplatform`) para persistir o banco SQLite,
artefatos e logs. Os limites padrão são 1 CPU, 1 GB de memória e 600 segundos por estágio, com
máximo configurável de 1800 segundos. O registry base padrão é `docker.io`; a allowlist inclui
`pypi.org` e `files.pythonhosted.org`, além do registry configurado.

Antes de qualquer build, será obrigatório configurar `MLPLATFORM_EGRESS_POLICY_CHECK` com um comando
local confiável que verifique a política efetivamente aplicada. O verificador recebe
`--verify-build-egress --json` e um `--allowed-repository HOST` por origem permitida. Deve retornar
JSON com `enforced: true`, um `policy_id` não vazio e `allowed_repositories` exatamente iguais à
allowlist solicitada. A plataforma falha fechada se a política não puder ser verificada. O comando
é configuração operacional local, não uma política fornecida ou aplicada pelo Docker.

Outras configurações:

- `MLPLATFORM_DATABASE_PATH`
- `MLPLATFORM_ARTIFACTS_DIR`
- `MLPLATFORM_LOGS_DIR`
- `MLPLATFORM_DOCKER_CPUS`
- `MLPLATFORM_DOCKER_MEMORY_LIMIT`
- `MLPLATFORM_BUILD_TIMEOUT_SECONDS`
- `MLPLATFORM_TRAINING_TIMEOUT_SECONDS`
- `MLPLATFORM_BASE_IMAGE_REGISTRY`
- `MLPLATFORM_EXTRA_BUILD_REPOSITORIES`

O build de dependências e o treinamento executam código não confiável. Docker local compartilha o
kernel e o daemon do host e não é uma sandbox de segurança absoluta. A instalação de dependências
pode precisar de acesso à rede, restrito pela política local; o runtime de treinamento será sem rede.