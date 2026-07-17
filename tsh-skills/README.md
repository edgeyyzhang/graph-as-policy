# tsh-skills

A [gap](https://github.com/graph-robots/graph-as-policy) skill registry —
robot skill bundles under `skills/` and model-backed tool bundles under
`tools/`, in the Agent Skills format.

## Use it

```bash
gap registry add tsh-skills /home/sumesh/gap/tsh-skills
gap check                      # what can run here?
```

## Add a bundle

```bash
gap skills new my-skill --kind skill --registry tsh-skills
gap skills check --skills /home/sumesh/gap/tsh-skills
gap skills test my-skill
```

See the open-robot-skills repo for the canonical examples and the full
SKILL.md contract.
