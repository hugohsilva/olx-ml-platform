# Uso de assistência de IA

Assistentes de IA foram usados como apoio à implementação, análise de testes e redação de
documentação. A assistência não substitui revisão humana: decisões de arquitetura, alterações de
código e resultados de validação devem ser confirmados por quem mantém o projeto.

## Correções realizadas durante a implementação

- Uma tentativa de inserir o probe de rede antes de `from __future__ import annotations` gerou erro
  de sintaxe; o probe foi movido para depois da declaração obrigatória e validado novamente.
- A revisão dos testes de limites encontrou uma indentação inválida que misturava o polling de
  timeout com a verificação de `HostConfig`. A validação de limites foi separada do loop de espera;
  os testes focados de executor, serviço, falhas e artefatos passaram após a correção.
- A validação é incremental: testes focados e Ruff precedem a suíte completa. Resultados são
  registrados apenas depois de executar os comandos, e tarefas não são marcadas como concluídas
  sem evidência.

Código gerado ou sugerido por IA deve passar pelo mesmo processo de revisão, lint, testes e
verificação de segurança que código escrito manualmente. Não inserir credenciais, dados pessoais ou
outros segredos nos prompts.
