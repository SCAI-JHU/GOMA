# GOMA: Proactive Embodied Cooperative Communication via Goal-Oriented Mental Alignment

The GOMA algorithm casts human-robot communication as a planning problem by selecting utterances that maximizally improves the efficiency of the joint plan in a partially observable environment.

- Reward of robot sharing information X to human: <br>
$R$(request X) = KL($\mathbb{E}$[human plan | human mind + X] || $\mathbb{E}$[human plan | human mind ]) - $C$ 

- Reward of robot requesting information X from human: <br>
$R$(request X) = KL($\mathbb{E}$[robot plan | robot mind + X] || $\mathbb{E}$[robot plan | robot mind ]) - $C$ 

where C is the communication cost. 


You can find a demonstration video on the website: lanceying.com/GOMA

The current implementation is tested in the virtual home domain. For installing virtual home, please clone this repo: https://github.com/xavierpuigf/virtualhome

To run the experiment use the following code:


```
python3 test_template_agent_structured.py --num-belief-particles 10 --num-proc 10 --model goma
```
