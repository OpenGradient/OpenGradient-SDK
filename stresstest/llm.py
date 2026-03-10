import argparse
import statistics

from utils import stress_test_wrapper

import opengradient as og

# Number of requests to run serially
NUM_REQUESTS = 100
MODEL = "anthropic/claude-haiku-4-5"


def main(private_key: str):
    client = og.Client(private_key=private_key)

    def run_prompt(prompt: str):
        client.llm.completion(MODEL, prompt, max_tokens=50)

    latencies, failures = stress_test_wrapper(run_prompt, num_requests=NUM_REQUESTS, is_llm=True)

    # Calculate and print statistics
    total_requests = NUM_REQUESTS
    success_rate = (len(latencies) / total_requests) * 100 if total_requests > 0 else 0

    if latencies:
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
    else:
        avg_latency = median_latency = min_latency = max_latency = p95_latency = 0

    print("\nOpenGradient LLM Stress Test Results:")
    print(f"Using model '{MODEL}'")
    print("=" * 20 + "\n")
    print(f"Total Requests: {total_requests}")
    print(f"Successful Requests: {len(latencies)}")
    print(f"Failed Requests: {failures}")
    print(f"Success Rate: {success_rate:.2f}%\n")
    print(f"Average Latency: {avg_latency:.4f} seconds")
    print(f"Median Latency: {median_latency:.4f} seconds")
    print(f"Min Latency: {min_latency:.4f} seconds")
    print(f"Max Latency: {max_latency:.4f} seconds")
    print(f"95th Percentile Latency: {p95_latency:.4f} seconds")

    if failures > 0:
        print("\n🛑 WARNING: TEST FAILED")
    else:
        print("\n✅ NO FAILURES")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM inference stress test")
    parser.add_argument("private_key", help="Private key for inference")
    args = parser.parse_args()

    main(args.private_key)
