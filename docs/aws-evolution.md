# Evolução conceitual para AWS

Este diagrama descreve opções futuras, não recursos implementados. O MVP é local e não requer
credenciais AWS.

```mermaid
flowchart LR
    Scientist["Cientista de dados / CLI"]
    API["API de controle"]
    Training["SageMaker Training Jobs"]
    Experiments["SageMaker Experiments / Lineage"]
    Registry["SageMaker Model Registry"]
    Storage["Amazon S3"]
    Serving["SageMaker Endpoints"]
    Orchestration["SageMaker Pipelines ou Step Functions"]
    IAM["IAM"]
    Observability["CloudWatch"]

    Scientist --> API
    API --> Orchestration
    Orchestration --> Training
    Training --> Storage
    Training --> Experiments
    Experiments --> Registry
    Registry --> Serving
    IAM -. permissões .-> API
    IAM -. permissões .-> Training
    IAM -. permissões .-> Storage
    IAM -. permissões .-> Serving
    API -. logs e métricas .-> Observability
    Training -. logs e métricas .-> Observability
    Serving -. logs e métricas .-> Observability
```

Uma migração exigiria definir identidade e autorização, criptografia e retenção dos dados,
estratégia de build de imagens, limites de custo e operação do endpoint. A escolha entre
Experiments/Lineage, Registry e orquestração deve atender a requisitos concretos da futura
plataforma; este case não replica serviços gerenciados localmente nem os provisiona.
