# Tasks: Local ML Lifecycle Management

**Input**: Design documents from `/specs/001-local-ml-lifecycle/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Testes unitários, integração, contrato e E2E são exigidos pelos requisitos NFR-005 e pelos
critérios de aceitação.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- Every task includes an exact file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create package structure for `src/domain`, `src/application`, `src/infrastructure`, `src/cli`,
  `src/api`, `tests/unit`, `tests/integration`, `tests/e2e` and `tests/contract` per plan.md
- [X] T002 Initialize Python project dependencies and register the `mlplatform` console command in
  pyproject.toml
- [X] T003 [P] Configure Pytest and test discovery in pyproject.toml
- [X] T004 [P] Configure linting and formatting for Python source in pyproject.toml
- [X] T005 Create environment configuration for local paths, Docker build/training resource limits,
  exact build repository allowlist (default `pypi.org`, `files.pythonhosted.org` and the configured
  base-image registry) and verifiable egress-policy preflight in src/infrastructure/config.py
- [X] T006 [P] Create the Iris example project structure in examples/iris/train.py,
  examples/iris/requirements.txt and examples/iris/data/
- [X] T007 Document local prerequisites and commands in README.md

**Acceptance**: Project imports, tests collect and Docker configuration is discoverable.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST complete before any user story

- [X] T008 Define immutable domain models using `data-model.md` `TrainingProject.project_id`,
  `TrainingProject.entrypoint`, `TrainingProject.dependencies`, `TrainingProject.data_location`,
  `TrainingProject.contract_version`, `Run.run_id`, `Run.project_id`, `Run.git_commit`,
  `Run.code_hash`, `Run.dataset_identity`, `Run.dataset_hash`, `Run.config_hash`,
  `Run.dependency_versions`, `Run.python_version`, `Run.docker_image`,
  `Run.build_allowed_repositories` in src/domain/models.py
- [X] T009 Define execution state and transition validation using `data-model.md`
  `Run.state` and `Run.failure_reason` in src/domain/executions/state_machine.py
- [X] T010 Define model and inference domain entities using `data-model.md`
  `ModelArtifact.run_id`, `ModelArtifact.artifact_location`, `ModelArtifact.checksum`,
  `InferenceRequest.instances`, `InferenceResponse.predictions`,
  `InferenceResponse.run_id`
  in src/domain/models.py
- [X] T011 [P] Implement unit tests for execution state transitions in
  tests/unit/test_execution_state.py
- [X] T012 [P] Implement unit tests for domain entity validation in tests/unit/test_domain_models.py
- [X] T013 Define repository and executor ports with small contracts in
  src/application/ports.py
- [X] T014 Implement SQLite schema for runs and metrics using `data-model.md`
  `Run.code_hash`, `Run.dataset_hash`, `Run.config_hash`, `Run.build_allowed_repositories`,
  `Metric.run_id`, `Metric.name`, `Metric.value`, `Metric.unit` and `LogRecord` fields in
  src/infrastructure/sqlite/schema.py
- [X] T015 Implement SQLite repository adapter for execution records in
  src/infrastructure/sqlite/execution_repository.py
- [X] T016 Implement filesystem adapter for artifact, log and execution directories in
  src/infrastructure/filesystem/artifact_store.py
- [X] T017 [P] Implement integration tests for SQLite persistence in
  tests/integration/test_sqlite_repository.py
- [X] T018 [P] Implement integration tests for filesystem persistence in
  tests/integration/test_artifact_store.py
- [X] T019 Implement application error types and explicit failure mapping in
  src/application/errors.py
- [X] T020 Configure application logging with execution correlation in
  src/infrastructure/logging.py
- [X] T056 Create and stabilize the Typer entrypoint and command-registration interface in
  src/cli/main.py; command implementations MUST live in separate modules under src/cli/commands/

**Checkpoint**: Foundation ready; all stories can use domain entities, persistence ports and error handling.

**Acceptance**: State transitions and persistence work independently; invalid transitions are rejected;
every write is attributable to an execution.

---

## Phase 3: User Story 1 - Executar um treinamento rastreavel (Priority: P1) MVP

**Goal**: Receive and validate a project, execute it in Docker and persist metrics/artifacts.

**Independent Test**: Submit `examples/iris`, wait for success and query the returned `run_id`.

### Tests for User Story 1

- [ ] T021 [P] [US1] Add training project contract tests for required files and dependency manifest in
  tests/contract/test_training_project_contract.py
- [ ] T022 [P] [US1] Add Docker executor integration tests with a valid and failing fixture in
  tests/integration/test_docker_executor.py
- [ ] T023 [P] [US1] Add training service integration tests for success, timeout and missing output in
  tests/integration/test_training_service.py; assert successful `Run` persistence of
  `code_hash`, `dataset_hash` and `config_hash`, and explicit validation failure for unavailable
  input sets with null unavailable hash, `failure_reason=validation` and no image build
- [ ] T024 [P] [US1] Add unit tests for metrics and parameters parsing in
  tests/unit/test_training_outputs.py
- [ ] T077 [P] [US1] Add one Docker integration acceptance scenario with two isolated attempts: a
  successful project that installs dependencies, persists installed versions by
  `data-model.md` `Run.dependency_versions` and immutable image identity by `Run.docker_image`, and
  runs a trainer fixture whose controlled outbound connection is blocked; then force a
  dependency-build failure and verify `image_build_error` is persisted and training never starts, in
  tests/integration/test_training_image_lifecycle.py
- [ ] T078 [P] [US1] Add Docker security integration tests for the build profile (effective non-root
  user, `NoNewPrivs=1`, zero effective capabilities observed inside dependency installation,
  CPU/memory/time limits, no host mounts, allowlisted repository access and denied non-allowlisted
  egress using a controlled local policy fixture) and training profile
  (`privileged=false`, `cap-drop=ALL`, `no-new-privileges`, `network=none`, read-only root
  filesystem, denied writes to root/input paths and successful writes to `/tmp`/output) in
  tests/integration/test_docker_security.py
- [ ] T079 [P] [US1] Add unit tests for SHA-256 file manifests, excluded project paths, stable
  ordering, canonical configuration JSON and unavailable input sets in
  tests/unit/test_input_hashes.py

### Implementation for User Story 1

- [ ] T025 [US1] Implement training project validation using `data-model.md`
  `TrainingProject.entrypoint`, `TrainingProject.dependencies`, `TrainingProject.data_location`
  and `TrainingProject.contract_version`; calculate `TrainingProject.code_hash`,
  `TrainingProject.dataset_hash` and `TrainingProject.config_hash` using the canonical SHA-256
  rules in src/application/training/project_validator.py
- [ ] T033 [US1] Declare Iris example dependencies in examples/iris/requirements.txt with pinned
  package versions recommended by contracts/training-project.md
- [ ] T075 [US1] After T005, T025 and T033, build a project-specific image from a platform-generated
  Dockerfile and install `requirements.txt` during build; execute the installer as non-root through
  a platform-owned wrapper that sets `no-new-privileges` and clears capabilities. Require the
  BuildKit sandbox without privileged entitlements, no host bind mounts, effective CPU/memory
  limits using BuildKit resource options, `Run.build_timeout_seconds` and exact egress allowlist from
  local policy. Reject the build before running installers if any control cannot be applied or
  verified; return logs and classify timeout versus build/profile failure in
  src/infrastructure/docker/image_builder.py
- [ ] T076 [US1] After T075, capture the actual installed package versions and immutable image
  ID/digest (using `pip freeze --all` for versions) in the image-build result using
  `data-model.md` `Run.dependency_versions` and
  `Run.docker_image` in src/infrastructure/docker/image_builder.py
- [ ] T027 [US1] Configure runtime mounts and writable paths: project/data input read-only, output
  writable for the non-root runtime user, and `/tmp` writable while the root filesystem is
  read-only when viable in
  src/infrastructure/docker/container_config.py
- [ ] T026 [US1] After T027, T075 and T076, implement the training Docker executor with runtime
  `--network none`, `privileged=false`, non-root user, `--cap-drop=ALL`,
  `--security-opt=no-new-privileges`, read-only root filesystem, defaults `cpus=1.0`, `memory=1g`,
  and `data-model.md` `Run.timeout_seconds` (600 default, 1800 maximum) in
  src/infrastructure/docker/training_executor.py
- [ ] T028 [US1] After T025-T027 and T075-T076, create the run before invoking final project
  validation/fingerprinting and build; persist canonical hashes and
  `Run.build_allowed_repositories`, set build timestamps, set
  `started_at` when training runtime begins, enforce independent build/training timeouts, persist
  build logs/metadata, map hash validation failures to `FAILED`/`validation`, build errors/timeouts to
  `image_build_error`/`image_build_timeout` and script errors/timeouts to
  `training_error`/`timeout`, and implement training orchestration using
  `data-model.md` `Run.run_id`,
  `Run.project_id`, `Run.git_commit`, `Run.code_hash`, `Run.dataset_identity`,
  `Run.dataset_hash`, `Run.config_hash`, `Run.dependency_versions`, `Run.python_version`,
  `Run.docker_image`, `Run.build_allowed_repositories`, `Run.image_build_started_at`,
  `Run.image_build_finished_at`,
  `Run.image_build_log_location`, `Run.state`, `Run.build_timeout_seconds`, `Run.timeout_seconds`,
  `Run.max_cpus`, `Run.memory_limit`, `Run.created_at`, `Run.started_at`, `Run.finished_at`,
  `Run.failure_reason`
  and `Run.failure_message` in src/application/training/training_service.py
- [ ] T029 [US1] Persist metrics and parameters by `data-model.md` `Metric.run_id`,
  `Metric.name`, `Metric.value`, `Metric.unit` in
  src/infrastructure/sqlite/metrics_repository.py
- [ ] T030 [US1] Capture container logs using `data-model.md` `LogRecord.run_id`,
  `LogRecord.sequence`, `LogRecord.timestamp`, `LogRecord.stream`, `LogRecord.message`
  in src/application/training/log_collector.py
- [ ] T031 [US1] Persist and validate model artifacts using `data-model.md`
  `ModelArtifact.run_id`, `ModelArtifact.artifact_location`, `ModelArtifact.size_bytes`,
  `ModelArtifact.checksum`, `ModelArtifact.created_at`, `ModelArtifact.serving_status`
  in src/application/training/artifact_ingestor.py
- [ ] T032 [US1] Implement the Iris Scikit-learn trainer and declared output files in
  examples/iris/train.py
- [ ] T034 [US1] Add a public Iris sample dataset or deterministic data preparation in
  examples/iris/data/
- [ ] T035 [US1] After T056, T025-T028 and T075-T076, register `mlplatform train PROJECT` with
  validated CPU, memory, build-timeout and training-timeout arguments from contracts/cli.md in
  src/cli/commands/train.py
- [ ] T036 [US1] Add E2E validation for the training-only Iris flow in tests/e2e/test_training_flow.py

**Checkpoint**: US1 independently runs a valid Iris training project and persists execution, metrics,
log and artifact.

**Acceptance**:
- Invalid project structure is rejected before container creation.
- A project-specific image installs `requirements.txt` during build and records actual package
  versions and immutable image identity.
- The image build applies and verifies its non-root security profile, resource/time limits and
  egress allowlist before executing dependency installation.
- Image-build failures/timeouts are `FAILED` with `image_build_error`/`image_build_timeout` and do
  not start a training container.
- A unique run_id exists before container start.
- Docker is used for every training run.
- Training runtime has no network and installs no dependencies.
- Runtime security settings and writable mount boundaries are applied and integration-tested.
- `code_hash`, `dataset_hash` and `config_hash` use canonical SHA-256 rules; unavailable required
  input sets fail validation before build with the missing hash and cause made explicit.
- Success persists metrics/artifacts; timeout/failure persists an explicit terminal state.

---

## Phase 4: User Story 2 - Consultar o historico de execucoes (Priority: P2)

**Goal**: List past executions and show details or a not-found result.

**Independent Test**: Create a succeeded execution, list it and retrieve detail by run_id.

### Tests for User Story 2

- [ ] T037 [P] [US2] Add repository integration tests for execution listing and lookup in
  tests/integration/test_execution_queries.py
- [ ] T038 [P] [US2] Add unit tests for execution detail assembly and missing execution handling in
  tests/unit/test_execution_query_service.py

### Implementation for User Story 2

- [ ] T039 [US2] Implement execution history and detail application service using `data-model.md`
  `Run.run_id`, `Run.project_id`, `Run.state`, `Run.created_at`, `Run.started_at`,
  `Run.finished_at`, `Run.parameters`, `Run.metrics`, `Run.log_location`,
  `Run.failure_reason` and `Run.failure_message` in src/application/executions/query_service.py
- [ ] T040 [US2] Extend SQLite repository with list ordered by `created_at` descending, tied by
  `run_id` ascending, and detail reads in
  src/infrastructure/sqlite/execution_repository.py
- [ ] T041 [US2] After T056 and T039-T040, register `mlplatform executions list/show RUN_ID` with
  newest-first and deterministic `run_id` tie-break behavior from contracts/cli.md in
  src/cli/commands/executions.py
- [ ] T042 [US2] Add list/detail E2E validation in tests/e2e/test_execution_queries.py

**Checkpoint**: US1 and US2 work together while US2 remains independently testable.

**Acceptance**:
- History returns `run_id` and state summary.
- Detail returns timestamps, parameters, metrics, logs and artifact references.
- Unknown ID returns explicit not-found behavior.

---

## Phase 5: User Story 3 - Servir um modelo para inferencia (Priority: P1)

**Goal**: Start a server with one succeeded model selected by `run_id` and return predictions with
that `run_id`.

**Independent Test**: Serve an Iris execution and call REST inference with one valid instance.

### Tests for User Story 3

- [ ] T043 [P] [US3] Add inference REST contract tests for valid payload and rejection of missing,
  malformed or additional request fields including `run_id` in
  tests/contract/test_inference_contract.py
- [ ] T044 [P] [US3] Add unit tests for inference payload validation in
  tests/unit/test_inference_payload.py
- [ ] T045 [P] [US3] Add integration tests for selecting a model by `run_id`, rejecting missing or
  unavailable artifacts before serving, and loading a valid artifact in
  tests/integration/test_model_serving.py
- [ ] T046 [P] [US3] Add REST integration tests for valid predictions and rejection of invalid or
  additional payload fields using the server-loaded model in
  tests/integration/test_inference_api.py

### Implementation for User Story 3

- [ ] T047 [US3] Implement model availability rules using `data-model.md`
  `ModelArtifact.serving_status`, `ModelArtifact.run_id` and
  `ModelArtifact.artifact_location`, `ModelArtifact.artifact_name` in
  src/application/inference/model_service.py; resolve the requested `run_id`, validate successful
  completion and artifact availability before server startup
- [ ] T048 [US3] Implement model loading and prediction orchestration using `data-model.md`
  `InferenceRequest.instances`, `InferenceResponse.predictions` and `InferenceResponse.run_id`;
  use only the model loaded for the server process in src/application/inference/prediction_service.py
- [ ] T049 [US3] Implement FastAPI application and POST `/predict` according to
  contracts/inference-rest.md in src/api/app.py
- [ ] T050 [US3] Implement request schema that accepts only `instances` and rejects extra fields,
  plus response schema with `run_id` and `predictions` in src/api/schemas.py
- [ ] T051 [US3] Implement explicit REST errors for invalid payload and prediction failure in
  src/api/errors.py
- [ ] T052 [US3] After T056 and T047-T051, register `mlplatform serve RUN_ID` in
  src/cli/commands/serve.py; load that model before launching the server and return an explicit
  nonzero CLI error without starting the server when validation fails
- [ ] T053 [US3] Add REST E2E validation that starts serving for a valid `run_id`, posts only
  `instances` to `/predict`, and checks response `run_id` and predictions; also verify invalid
  or unavailable run IDs prevent server startup in
  tests/e2e/test_inference_flow.py

**Checkpoint**: US3 serves a completed model and rejects invalid or unavailable inference explicitly.

**Acceptance**:
- Only succeeded models with available artifacts can be served.
- Valid request returns predictions and the `run_id` loaded by the server.
- Invalid payload and inference failure return explicit errors; a missing model fails server startup.

---

## Phase 6: User Story 4 - Operar o fluxo pela CLI (Priority: P2)

**Goal**: Provide a small, usable CLI for train, history, detail and serve operations.

**Independent Test**: Run the CLI flow in an installed local environment with the Iris example.

### Tests for User Story 4

- [ ] T054 [P] [US4] After T056, add CLI contract tests for command names, arguments, defaults,
  validation and exit codes in
  tests/contract/test_cli_contract.py
- [ ] T055 [US4] After T035, T041 and T052, add CLI integration tests for train, list, show and
  serve in
  tests/integration/test_cli.py

### Implementation for User Story 4

- [ ] T057 [US4] Implement human-readable execution and artifact output in src/cli/formatters.py
- [ ] T059 [US4] After T035, T041, T052 and T057, add CLI E2E validation for the documented
  five-step first-run workflow, including serving and REST inference, in
  tests/e2e/test_cli_flow.py

**Checkpoint**: All main operations are reachable through a small CLI and return clear statuses.

**Acceptance**:
- `mlplatform train`, `mlplatform executions list/show RUN_ID` and `mlplatform serve RUN_ID`
  are available.
- Success returns IDs/results; failures show a concise cause and nonzero status.
- The example is documented in at most five user steps.

---

## Phase 7: End-to-End and Cross-Cutting Validation

**Purpose**: Connect stories and prove the complete MVP flow

- [ ] T060 [P] Add complete train/persist/query/serve/predict E2E test asserting one successful output,
  returned `run_id`, and functional payload in
  tests/e2e/test_full_lifecycle.py
- [ ] T061 [P] Add separate build-timeout, training-timeout and container-failure E2E coverage in
  tests/e2e/test_training_failures.py
- [ ] T062 [P] Add sensitive-log and mounting-boundary security tests in
  tests/unit/test_security_boundaries.py
- [ ] T063 Re-evaluate and remove temporary or stale Docker containers in
  src/infrastructure/docker/container_lifecycle.py
- [ ] T064 Document prerequisites and the five-step Docker, CLI and API workflow in README.md
- [ ] T065 Document architecture decisions and rejected alternatives in DECISIONS.md
- [ ] T066 Document AI usage and corrections in AI_USAGE.md
- [ ] T067 Add a conceptual AWS evolution diagram in docs/aws-evolution.md
- [ ] T068 [P] Add unit tests for CPU, memory, build-timeout and training-timeout validation in
  tests/unit/test_resource_limits.py
- [ ] T069 [P] Add Docker integration tests proving effective CPU/memory settings for build and
  training, and independent timeout behavior in tests/integration/test_docker_resource_limits.py
- [ ] T070 [P] Add failure persistence tests for `image_build_error`, `image_build_timeout`,
  including build OOM mapped to `image_build_error`, `training_error`, training `timeout`,
  training OOM mapped to `oom_killed` and resource-limit outcomes in
  tests/integration/test_resource_failure_status.py
- [ ] T071 [P] Add one scenario that measures the complete functional acceptance set: valid
  persist/query/predict/`run_id`, no arbitrary latency claim, and all invalid cases in
  tests/e2e/test_acceptance_criteria.py
- [ ] T072 Run the full test suite and quickstart.md validation
- [ ] T073 Document known limitations, untrusted dependency-install code and excluded scope in
  README.md
- [ ] T074 After T064, T072 and T073, have a person not responsible for authoring the feature execute
  the five-step quickstart workflow and record first-success in DECISIONS.md

**Checkpoint**: Full lifecycle passes locally and documentation matches verified behavior.

**Acceptance**:
- Full Iris flow passes one automated E2E test.
- Failure cases and security boundaries have passing tests.
- No AWS, Kubernetes or dashboard implementation exists.
- README can be followed in at most five steps.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Phase 1; blocks all user stories
- **T056**: Foundational entrypoint; complete before T035, T041, T052, T054 and T055
- **US1 (Phase 3)**: Depends on Foundational
- **US2 (Phase 4)**: Depends on Foundational and can integrate with US1 artifacts
- **US3 (Phase 5)**: Depends on Foundational and US1 model artifact behavior
- **US4 (Phase 6)**: Depends on Foundational and interfaces used from US1-US3
- **Validation (Phase 7)**: Depends on all selected stories

### User Story Dependencies

- **US1**: No dependency on US2-US4.
- **US2**: Reuses persisted execution records produced by US1.
- **US3**: Reuses successful model artifact behavior from US1.
- **US4**: Aggregates operations from US1-US3.

### Task Dependencies

- **T035** depends on T056, training validation/orchestration tasks T025-T028 and image build/
  metadata tasks T075-T076.
- **T075** depends on T005, T025 and T033; it builds only after the local allowlist/policy
  configuration, project validation and dependency manifest are available.
- **T076** depends on T075 because installed versions and image identity are outputs of the build.
- **T026** depends on T027, T075 and T076; the executor needs runtime mount/security configuration
  and a successfully built image with recorded metadata.
- **T028** depends on T025-T027 and T075-T076; it creates/persists fingerprints, the allowlist,
  separate build and training failure reasons and the build log location.
- **T023** verifies hash persistence after T025 and T028 are implemented.
- **T077** runs its single integrated acceptance scenario after T025, T026, T028, T033 and
  T075-T076; the scenario covers both a successful offline training run and a failed image build.
- **T078** verifies the effective security profiles after T005, T026, T027 and T075 are implemented.
- **T079** is independent unit-test work for deterministic input fingerprinting and can be authored
  before T025.
- **T041** depends on T056 and query service/repository tasks T039-T040.
- **T052** depends on T056 and model startup/API tasks T047-T051.
- **T054** depends on T056 and contracts/cli.md; it validates command registration, arguments and
  validation behavior.
- **T055** depends on T035, T041 and T052 because it exercises the integrated command set.
- **T059** depends on T035, T041, T052 and T057; it validates the documented five-step CLI lifecycle.
- **T074** depends on the completed documentation and validation tasks T064, T072 and T073.

## Parallel Opportunities

- T003-T006 can run in parallel after project initialization.
- T011-T012 and T017-T018 are independent test and adapter work.
- Contract tests T021, T022, T037 and T043 can run in separate files; T054 can run after T056 while
  command modules are implemented in parallel.
- T077 and T078 use separate integration-test files; T079 is an independent unit-test file.
- T023's hash-persistence assertions can be authored with the existing training-service tests.
- US2 can proceed in parallel with US3 after US1's artifact contract is stable.
- Documentation tasks T064-T067 can run in parallel after behavior is verified.

## Implementation Strategy

### MVP First (User Story 1)

1. Complete Setup and Foundational.
2. Implement and test US1.
3. Stop and run the Iris training-only E2E test.
4. Demonstrate unique run_id, persisted metrics and artifacts.

### Incremental Delivery

1. Add US2 to demonstrate execution history.
2. Add US3 to demonstrate REST inference.
3. Add US4 to make the operations accessible from one CLI.
4. Finish cross-cutting validation and documentation.

### Parallel Team Strategy

- Developer A: Foundational domain, persistence and T056 CLI entrypoint.
- Developer B: image builder, hardened Docker executor and US1.
- Developer C: US2 history and US3 API.
- Developer D: Typer CLI and E2E tests.
