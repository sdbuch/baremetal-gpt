# TODO

- [ ] Basic setup on TRC infra.
  - [x] Local tests while SSHed into box. Matmul, parallel matmul
  - [x] Basic configuration of uv on box and runner script, allowing to remote
    execute the code. Configure aliases for spinning up (install uv) and
    launching. Target single-host
  - [x] Set up infra for multi-host running next, if can get a two-host VM
    - [x] Study the TPU playbook: sharding and parallel primitives.
  - [ ] Infra for spot VMs, checkpointing and resuming? Or model first...
    - [ ] Stretch: think about doing data/checkpoints through a persistent disk
      (more expensive)
