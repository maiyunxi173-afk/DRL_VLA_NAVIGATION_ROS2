# VLA Bridge

Integrated Demo4/Demo5/Demo6 scripts used by the SAC navigation stack.

This folder was consolidated from the previous external
`~/bpx_vla_navigation/src/vla_bridge` workspace so the full demo can run from a
single repository.

| File | Function |
| --- | --- |
| `vla_bridge/demo4_test_batch.py` | Qwen/DashScope scene and task interpretation |
| `vla_bridge/demo5_ros_detector.py` | Qwen-VL or local color bbox detection |
| `vla_bridge/demo5_ros_bridge.py` | Compatibility bridge for file-based Demo5 results |
| `vla_bridge/demo6_vla_coord_node.py` | Bbox + depth + camera info + TF to target pose |
| `vla_bridge/vla_agent_node.py` | Manual task adapter; never publishes `/cmd_vel` |

Runtime outputs are written to:

```text
<DRLNAV_BASE_PATH>/results/vla_bridge/
```

The `results/` directory is ignored by Git.
