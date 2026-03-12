import os

import opengradient as og

alpha = og.Alpha(private_key=os.environ.get("OG_PRIVATE_KEY"))

queries = [
    "how much protein should a female eat",
    "are judo throws allowed in wrestling?",
]
instruction = ["Given a web search query, retrieve relevant passages that answer the query"]
passages = [
    "As a guideline, the CDC's average requirement of protein for women ages  to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
]

model_embeddings = alpha.infer(
    model_cid="intfloat/multilingual-e5-large-instruct",
    model_input={"queries": queries, "instruction": instruction, "passages": passages},
    inference_mode=og.InferenceMode.VANILLA,
)

print(f"Output: {model_embeddings.model_output}")
print(f"Tx hash: {model_embeddings.transaction_hash}")
