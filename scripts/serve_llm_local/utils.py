import os
import asyncio
from openai import AsyncOpenAI
import json
import time
from collections import deque
from typing import Deque, Optional, List
import logging
from tqdm.asyncio import tqdm_asyncio
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_server_ips() -> List[str]:
    """Get list of server IPs from the most recent complete server instances file"""
    json_files = glob.glob("server_status/server_instances_complete_*.json")
    if not json_files:
        raise RuntimeError("No server instances file found")
    
    # Get the most recent file
    latest_file = max(json_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        server_info = json.load(f)
    
    # Extract IPs from server info
    ips = []
    for info in server_info:
        if 'ip' in info:
            ips.append(info['ip'])
    
    if not ips:
        raise RuntimeError("No IPs found in server instances file")
    
    logger.info(f"Found {len(ips)} server instances: {ips}")
    return ips

class RateLimiter:
    def __init__(self, max_window_size: int):
        self.max_window_size = max_window_size
        self.semaphore = asyncio.Semaphore(max_window_size)
        logger.info(f"Created rate limiter with max_window_size={max_window_size}")
    
    async def acquire(self):
        await self.semaphore.acquire()
        logger.info(f"Successfully acquired slot. Available slots: {self.semaphore._value}/{self.max_window_size}")
    
    async def release(self):
        self.semaphore.release()
        logger.info(f"Successfully released slot. Available slots: {self.semaphore._value}/{self.max_window_size}")

class RoundRobinClient:
    def __init__(self, ips: List[str], timeout: int, rate_limiters: List[RateLimiter]):
        self.ips = ips
        self.current_index = 0
        self.clients = [
            AsyncOpenAI(
                base_url=f"http://{ip}:8000/v1",
                api_key="token-abc123",
                timeout=timeout,
            ) for ip in ips
        ]
        self.rate_limiters = rate_limiters
        logger.info(f"Initialized RoundRobinClient with {len(ips)} instances")
        for i, ip in enumerate(ips):
            logger.info(f"Instance {i}: {ip} with rate limit {rate_limiters[i].max_window_size}")
    
    async def get_next_available_client(self) -> tuple[AsyncOpenAI, RateLimiter]:
        # Find the instance with the most available slots
        max_available = -1
        best_client = None
        best_limiter = None
        best_index = -1
        
        # Log current state of all instances
        logger.info("Current instance states:")
        for i in range(len(self.clients)):
            available = self.rate_limiters[i].semaphore._value
            used = self.rate_limiters[i].max_window_size - available
            logger.info(f"Instance {i}: {self.ips[i]} - Used: {used}, Available: {available}/{self.rate_limiters[i].max_window_size}")
        
        # Check all instances to find the one with most available slots
        for i in range(len(self.clients)):
            available = self.rate_limiters[i].semaphore._value
            if available > max_available and available > 0:
                max_available = available
                best_client = self.clients[i]
                best_limiter = self.rate_limiters[i]
                best_index = i
                logger.info(f"Selected instance {best_index} ({self.ips[best_index]}) with {max_available} available slots")
        
        if best_client is not None:
            # Found an instance with available capacity
            await best_limiter.acquire()
            return best_client, best_limiter
        else:
            # All instances are at capacity, distribute waiting requests across all instances
            logger.info("All instances at capacity, distributing wait across all instances")
            
            # Create a list of all instances that we can wait on
            wait_tasks = []
            for i in range(len(self.clients)):
                task = asyncio.create_task(self.rate_limiters[i].acquire())
                wait_tasks.append((i, task))
            
            # Wait for any instance to become available
            try:
                # Wait for the first instance to become available
                done, pending = await asyncio.wait(
                    [task for _, task in wait_tasks],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel all other waiting tasks
                for _, task in wait_tasks:
                    if task not in done:
                        task.cancel()
                
                # Find which instance became available
                for i, task in wait_tasks:
                    if task in done:
                        logger.info(f"Instance {i} ({self.ips[i]}) became available")
                        return self.clients[i], self.rate_limiters[i]
                
                # This should never happen, but just in case
                raise RuntimeError("No instance became available despite wait completion")
            except Exception as e:
                logger.error(f"Error while waiting for instances: {e}")
                raise

class LLMClient:
    def __init__(self, timeout_seconds: int, max_window_size_per_instance: int):
        self.timeout_seconds = timeout_seconds
        server_ips = get_server_ips()
        # Create a rate limiter for each instance
        rate_limiters = [RateLimiter(max_window_size_per_instance) for _ in server_ips]
        self.client_manager = RoundRobinClient(server_ips, timeout_seconds, rate_limiters)
        logger.info(f"Initialized LLMClient with {len(server_ips)} instances, {max_window_size_per_instance} slots per instance")

    async def single_call(self, inputs, model):
        try:
            start_time = time.time()
            logger.info("Starting to get next available client...")
            # Get next available client and its rate limiter (already acquired)
            client, rate_limiter = await self.client_manager.get_next_available_client()
            logger.info(f"Got client for {client.base_url}")
            try:
                logger.info(f"Making request to {client.base_url}")
                response = await client.chat.completions.create(
                    model=model, 
                    messages=inputs,
                    timeout=self.timeout_seconds
                )
                duration = time.time() - start_time
                
                if duration > self.timeout_seconds * 0.9:  # If request took more than 90% of timeout
                    logger.warning(f"Request took {duration:.2f}s (close to timeout {self.timeout_seconds}s)")
                
                logger.info(f"Request completed in {duration:.2f}s")
                return response.choices[0].message.content
            finally:
                await rate_limiter.release()
                logger.info(f"Released rate limiter slot. Available slots: {rate_limiter.semaphore._value}")
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {self.timeout_seconds}s")
            return None
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return None

    async def process_all_inputs(self, inputs_list, num_generations=1, model=None):
        try:
            # Create N tasks for each input
            all_tasks = []
            for inputs in inputs_list:
                tasks = []
                for _ in range(num_generations):
                    tasks.append(self.single_call(inputs, model=model))
                all_tasks.extend(tasks)
            
            # Run all tasks concurrently with progress bar
            responses = await tqdm_asyncio.gather(*all_tasks, desc="Processing requests")
            
            # Group responses by input
            grouped_responses = []
            for i in range(0, len(responses), num_generations):
                grouped_responses.append(responses[i:i + num_generations])
            
            return grouped_responses
        finally:
            pass  # No need to stop the rate limiters as they're managed by the acquire/release calls 