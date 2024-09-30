# Reasoning Bench
A suite of benchmark datasets to evaluate LLM reasoning capabilities.


## NumSeqBench: Benchmarking Inductive Reasoning in Language Models via Number Sequences
[Blog](https://huggingface.co/blog/kenhktsui/numseqbench) |[Dataset](https://huggingface.co/datasets/kenhktsui/num_seq_bench)

Inductive reasoning is critical to human cognition. It is a form of logical thinking where we draw general conclusions from specific observations. The beautify of it is that it allows us to learn from experience, without being told the principle, and apply that knowledge to new situations.

Do LLM perform well in inductive reasoning? Few shot prompting finds that by adding more samples, task performance increases. It is an early sign that LLM can peform inductive reasoning.

We introduced a benchmark dataset named NumSeqBench to evaluate LLM capability inductive reasoning in numeric sequence. 
"Find the next term" in a number sequence is common in most of the cognitive tests. However, we argue that it is alone is subject to two limitations:
- it is not uncommon in web, heightening the data contamination risk
- find the next term is close to next token prediction pretraining objective, and as such might have an favorable performance. The short-term pattern recognition however does not necessarily mean the model understand the sequence generating function. 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/60e50ce5350d181892d5a636/p-_AgyZxvkinv8UbeWx6D.png)


To address these limitations, we propose two extra tests - "next nth term (nth)" and "previous term (previous)" task to evaluate LLM ability to do long range inference, and backward inference. 
If a model can conclude the sequence generating function by induction, it can inference a term in any arbitrary position (next, nth, and the previous). Moreover, it can conclude whether such generating function exists. To test so, we added a novel test - inclusion of monotonic random sequence to evaluate if LLM can determine it is a random sequence and know it is not able to answer.


### To reproduce
```shell
# install
pip install -r requirements.txt
# setting API key
export ANTHROPIC_API_KEY=[API_KEY]
export OPENAI_API_KEY=[API_KEY]
export DEEPINFRA_API_KEY=[API_KEY]
export DEEPKSEEK_API_KEY=[API_KEY]
export GOOGLEAI_API_KEY=[API_KEY]
export REKA_API_KEY=[API_KEY]
export YI_API_KEY=[API_KEY]
export MISTRAL_API_KEY=[API_KEY]
# run
python -m num_seq_bench.run_llm_api_inference
python -m num_seq_bench.analysis
```

### Citation
```
@misc{numseqbench2024,
    title={NumSeqBench: Benchmarking Inductive Reasoning in Language Models via Number Sequences},
    author={Ken Tsui},
    url={https://huggingface.co/blog/kenhktsui/numseqbench},
    year={2024}
}
```