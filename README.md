## GOMA: Proactive Embodied Cooperative Communication via Goal-Oriented Mental Alignment<br><sub>IROS 2024 | [Paper](https://arxiv.org/abs/2403.11075) | [Website](https://www.lanceying.com/goma.html) | [Video](https://youtu.be/U4zGkDb-mHU)</sub>

## Introduction

The GOMA algorithm casts human-robot communication as a planning problem by selecting utterances that maximizally improves the efficiency of the joint plan in a partially observable environment.

- Reward of robot sharing information X to human: <br>
$R$(request X) = KL($\mathbb{E}$[human plan | human mind + X] || $\mathbb{E}$[human plan | human mind ]) - $C$

- Reward of robot requesting information X from human: <br>
$R$(request X) = KL($\mathbb{E}$[robot plan | robot mind + X] || $\mathbb{E}$[robot plan | robot mind ]) - $C$

where C is the communication cost.

## Installation

Before running the following code, replace `${VH_PATH}` and `${GOMA_PATH}` with the path on your machine, and set `OPENAI_API_KEY` with your own OpenAI API key.

```sh
# 1. prepare virtualhome environment
## code
git clone https://github.com/xavierpuigf/virtualhome ${VH_PATH}
cd ${VH_PATH}
git switch wah
## executable
wget http://virtual-home.org/release/simulator/v2.0/v2.2.4/linux_exec.zip -O v2.2.4.zip
unzip v2.2.4.zip -d ${VH_PATH}/bin/v2.2.4
export VH_BIN="${VH_PATH}/bin/v2.2.4/linux_exec.v2.2.4.x86_64"
export PYPATH_VH="${VH_PATH}/virtualhome:${VH_PATH}/virtualhome/simulation"

# 2. clone this repo
git clone https://github.com/SCAI-JHU/GOMA ${GOMA_PATH}

# 3. setup environment variables
export OPENCV_IO_ENABLE_OPENEXR=1
export OPENAI_API_KEY=...
export OPENAI_MODEL="gpt-4"
export OPENAI_MAX_TOKENS="256"

# 4. run experiment
cd ${GOMA_PATH}/testing_agents
export PYTHONPATH="${PYPATH_VH}:${GOMA_PATH}:$PYTHONPATH"
python test_template_agent_structured.py \
  --base-port=8088 \
  --num-belief-particles=10 \
  --num-proc=10 \
  --model="goma"
```

For VSCode users, we also provide [`.vscode/launch.json`](.vscode/launch.json) for quick debugging.

## Dataset

[`train_env_task_set.pik`](./train_env_task_set.pik) contains 21 collaborative scenarios in VirtualHome across 4 goal types (`setup_table`, `put_fridge`, `prepare_food`, `put_dishwasher`) and 3 simulated apartments.

Below is an example of the task format:

```json
[
  {
    "task_id": 5,
    "task_name": "setup_table",
    "env_id": 0,
    "task_goal": {
      "0": {
        "on_wineglass_231": 3,
        "on_plate_231": 3,
        "on_cutleryfork_231": 3
      },
      "1": {}
    },
    "level": 0,
    "init_rooms": ["bedroom", "bathroom"],
    "init_graph": {
      "nodes": [
        {
          "id": 11,
          "category": "Rooms",
          "class_name": "bathroom",
          "prefab_name": "PRE_ROO_Bathroom_01",
          "obj_transform": {
            "position": [-6.385, -0.003, -0.527],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0]
          },
          "bounding_box": {
            "center": [-5.135, 1.247, 0.723],
            "size": [8.0, 3.0, 5.5]
          },
          "properties": [],
          "states": []
        }, {}, {}, {}
      ],
      "edges": [
        {
          "from_id": 12,
          "to_id": 11,
          "relation_type": "INSIDE"
        }, {}, {}, {}
      ]
    }
  }, {}, {}, {}
]
```

## Troubleshooting

#### OSError: The port 8088 is already being used

```sh
lsof -i :8088 -t | xargs -r kill -9
```

## Citing GOMA

Please cite our paper and star this repo if you find it interesting or useful. Thank you!

```bibtex
@inproceedings{ying2024goma,
  title={Goma: Proactive embodied cooperative communication via goal-oriented mental alignment},
  author={Ying, Lance and Jha, Kunal and Aarya, Shivam and Tenenbaum, Joshua B and Torralba, Antonio and Shu, Tianmin},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={7099--7106},
  year={2024},
  organization={IEEE}
}
```
