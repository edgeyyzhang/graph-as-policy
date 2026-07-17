Created oracle handover with as robust as possible primitives (using hardcoded constants)
    - using goto GaP skill with Shuangyu's IK in LIBERO-YAM




DESIGN DECISION: I would really like to have arm-arm avoidance, which will make the grasp 
theoretically less of a headache.  First, need to define collision spheres in order to utilize curobo for the YAM

1. The other arm must be modeled as cuboids, not spheres. cuRoboV2's scene checker (load_from_scene_cfg) only ingests cuboid, mesh, and voxel — it silently drops sphere/capsule/cylinder. And the mesh path is the broken-warp one. So the other arm has to go in as per-link bounding boxes. (Good news: my connector's _load_world/plan_to_pose are already cuboid-based, so that's aligned.)
2. The planner is picky about feasibility. It plans fine for reachable targets, but my synthetic detour tests kept failing — partly because I placed obstacles/targets near the workspace edge where no collision-free path exists (e.g. a box blocking a near-base lateral sweep). I can't cleanly prove "it detours" with contrived geometry; the honest test is the real handover, where a collision-free path provably exists (the current scripted diagonal is one).


### Things I'm worried about
- Recommended subgraph in SKILL.md


### Training an ACT