# OLX ML Platform

Plataforma local de gerenciamento do ciclo de vida de modelos de Machine Learning, desenvolvida
como case técnico para uma posição de Staff MLOps Engineer. Cientistas de dados fornecem um projeto
Python; a plataforma cria uma execução, constrói uma imagem de treinamento isolada, persiste
resultados e permite servir o modelo por uma API REST local.

## Pré-requisitos

- Python 3.11 ou superior
- [uv](https://docs.astral.sh/uv/)
- Docker instalado, em execução e acessível ao usuário (`docker info`)

## Primeiro fluxo: Iris em cinco passos

Na raiz do repositório:

1. Instale as dependências do projeto:

   ```bash
   uv sync --locked --no-editable --reinstall-package olx-ml-platform
   ```

2. Confirme que o daemon Docker está disponível:

   ```bash
   docker info
   ```

3. Execute o treinamento de demonstração e guarde o `run_id` exibido:

   ```bash
   uv run --no-sync mlplatform train examples/iris
   ```

4. Consulte as execuções e os detalhes da execução:

   ```bash
   uv run --no-sync mlplatform executions list
   uv run --no-sync mlplatform executions show RUN_ID
   ```

5. Em um terminal, inicie o serving; em outro, envie uma predição. Substitua `RUN_ID` pelo valor
   obtido no passo 3:

   ```bash
   uv run --no-sync mlplatform serve RUN_ID
   ```

   ```bash
   curl -X POST http://127.0.0.1:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"instances":[[5.1,3.5,1.4,0.2],[6.0,2.9,4.5,1.5]]}'
   ```

A resposta contém uma predição por linha de `instances`, na mesma ordem, e o mesmo `run_id`
selecionado ao iniciar o servidor. Cada linha informa comprimento e largura da sépala, seguidos de
comprimento e largura da pétala. Os rótulos são `0` (Iris-setosa), `1` (Iris-versicolor) e `2`
(Iris-virginica). O serviço fica vinculado a `127.0.0.1`; mantenha-o em execução durante as chamadas.

## Projeto de treinamento

Um projeto deve conter `train.py`, `requirements.txt` com dependências fixadas e um diretório
`data/`. O script recebe `MLPLATFORM_OUTPUT_DIR` e deve gravar ali `model.joblib`,
`metrics.json` e `parameters.json`. O exemplo em `examples/iris` demonstra o contrato. A plataforma
valida e calcula fingerprints do projeto antes de construir a imagem.

O comando `train` aceita `--cpus` (máximo 1), `--memory-limit` (máximo 1g),
`--build-timeout-seconds` e `--timeout-seconds` (máximo 1800 cada). Os defaults são 1 CPU, 1 GB e
600 segundos por estágio.

## Armazenamento e configuração

Por padrão, SQLite, logs e artefatos são armazenados em `~/.mlplatform`. É possível configurar:

- `MLPLATFORM_HOME`
- `MLPLATFORM_DATABASE_PATH`
- `MLPLATFORM_ARTIFACTS_DIR`
- `MLPLATFORM_LOGS_DIR`
- `MLPLATFORM_DOCKER_CPUS`
- `MLPLATFORM_DOCKER_MEMORY_LIMIT`
- `MLPLATFORM_BUILD_TIMEOUT_SECONDS`
- `MLPLATFORM_TRAINING_TIMEOUT_SECONDS`
- `MLPLATFORM_BASE_IMAGE_REGISTRY`
- `MLPLATFORM_EXTRA_BUILD_REPOSITORIES` (lista de hostnames exatos separados por vírgula)

O build de dependências usa uma allowlist de egress e deve verificar o perfil efetivo de segurança e
recursos antes de executar instaladores. Se o Docker/BuildKit local não puder aplicar ou comprovar
esses controles, o build falha fechado; não há fallback para um perfil mais permissivo. O Docker
daemon ainda pode baixar imagens auxiliares fora do proxy da execução.

## Segurança e limitações conhecidas

Código Python e dependências fornecidos pelo usuário são não confiáveis. O build executa a
instalação de dependências em um ambiente limitado; o treinamento roda sem rede, como usuário não
root, sem capabilities e com filesystem raiz somente leitura. Artefatos serializados também são
não confiáveis e são carregados dentro de um container de inferência limitado.

Containers compartilham o kernel e o daemon Docker do host; portanto, esta plataforma local não
oferece uma fronteira de segurança absoluta nem substitui sandboxing gerenciado. A API é local,
sem autenticação ou TLS, e serve um único modelo por processo. O MVP não implementa AWS,
Kubernetes, treinamento distribuído, dashboard, monitoramento avançado ou alta disponibilidade.
Veja [DECISIONS.md](./DECISIONS.md) e [docs/aws-evolution.md](./docs/aws-evolution.md) para
decisões, limitações e uma evolução conceitual.

## Desenvolvimento

```bash
uv sync --dev
PYTHONPATH=src uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```
