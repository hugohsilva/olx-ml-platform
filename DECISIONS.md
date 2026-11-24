# Decisões de arquitetura

## Contexto e decisões

- **MVP local, aplicação única.** A CLI, API e serviços de aplicação compartilham um único
  processo/distribuição. Não há necessidade demonstrada para microserviços.
- **Domínio e aplicação independentes de adaptadores.** Regras de execução e inferência são
  expressas nos serviços e ports da aplicação; Docker, SQLite e filesystem são adaptadores locais.
- **Docker para isolamento.** Builds e treinamentos são containers separados, com limites
  configurados. O treinamento usa rede desativada e grava somente no diretório de saída montado.
  Builds aplicam política de egress limitada e são recusados se o perfil exigido não puder ser
  verificado.
- **SQLite e filesystem.** SQLite guarda execuções, métricas e metadados; logs e artefatos ficam no
  filesystem local, associados ao `run_id`. Artefatos de modelo são conferidos por tamanho e
  checksum antes do serving.
- **Imagem e identidade imutáveis por execução.** A execução registra a identidade da imagem de
  treinamento e as versões instaladas para tornar a proveniência consultável.
- **Inferência no container.** O artefato serializado e o payload são montados somente para leitura
  em um container baseado na imagem da execução. Isso evita desserializar modelos não confiáveis no
  processo local da CLI/API.
- **Contrato de treino pequeno.** Um projeto fornece `train.py`, `requirements.txt` e `data/`;
  resultados são escritos no diretório de saída definido pela plataforma.
- **Sem MLflow no MVP.** SQLite e filesystem cobrem a rastreabilidade local exigida sem adicionar
  um serviço e um modelo operacional que não são necessários para este case.

## Limitações e alternativas adiadas

- Docker compartilha kernel e daemon com o host; os controles locais não constituem isolamento
  absoluto. O daemon pode baixar imagens auxiliares fora da política de egress da execução.
- A API serve um modelo por processo, apenas em loopback, sem autenticação, TLS ou alta
  disponibilidade.
- SQLite e filesystem são armazenamento de uma única máquina; não há coordenação distribuída,
  retenção automatizada nem registry compartilhado.
- SageMaker, S3, IAM, CloudWatch, Kubernetes, treinamento distribuído, dashboard e monitoramento
  avançado estão fora do escopo de implementação do MVP. A evolução conceitual está em
  [docs/aws-evolution.md](./docs/aws-evolution.md).
- O BuildKit precisa demonstrar os limites e a configuração de segurança requeridos. A plataforma
  falha fechada quando o ambiente local não consegue aplicá-los; reduzir os controles para acomodar
  um daemon incompatível foi rejeitado.

## Validação das decisões

As decisões devem permanecer justificadas por testes de contrato, integração e ponta a ponta. A
suíte deve verificar estado e falhas persistidos, limites e mounts Docker, consulta de artefatos e
inferência usando o `run_id` selecionado. Mudanças para serviços gerenciados devem ser precedidas
por requisitos operacionais concretos, não por abstrações antecipadas.

## Evidência de execução manual

O autor executou manualmente o fluxo local de treinamento, serving e inferência HTTP. O treinamento
de `examples/iris` concluiu com `state=SUCCEEDED` e `run_id` `f053ec33-d5f8-430b-868a-104a5e1b7ed9`;
os detalhes consultados incluíram métricas e a referência ao log e à imagem Docker. Com o serving
desse `run_id` ativo, uma chamada `POST /predict` para
`{"instances":[[5.1,3.5,1.4,0.2]]}` retornou
`{"run_id":"f053ec33-d5f8-430b-868a-104a5e1b7ed9","predictions":[0]}`.

