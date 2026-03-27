import random
import time
import uuid
from typing import Callable, Coroutine, List, Tuple


def generate_unique_prompt(request_id: int) -> str:
    """Generate a unique prompt for testing."""
    topics = ["science", "history", "technology", "art", "sports", "music", "literature", "philosophy", "politics", "economics"]
    adjectives = ["interesting", "surprising", "little-known", "controversial", "inspiring", "thought-provoking"]

    topic = random.choice(topics)
    adjective = random.choice(adjectives)
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of a UUID

    return f"Request {request_id}: Tell me a {adjective} fact about {topic}. Keep it short. Unique ID: {unique_id}"


async def stress_test_wrapper(infer_function: Callable[..., Coroutine], num_requests: int) -> Tuple[List[float], int]:
    """
    Async wrapper function to stress test the inference.

    Args:
    infer_function (Callable): An async inference function to test.
    num_requests (int): Number of requests to send.

    Returns:
    Tuple[List[float], int]: List of latencies for each request and the number of failures.
    """
    latencies = []
    failures = 0

    for i in range(num_requests):
        input_data = generate_unique_prompt(i)

        start_time = time.time()

        try:
            _ = await infer_function(input_data)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            print(f"Request {i + 1}/{num_requests} completed. Latency: {latency:.4f} seconds")
        except Exception as e:
            failures += 1
            print(f"Request {i + 1}/{num_requests} failed. Error: {str(e)}")

    return latencies, failures
