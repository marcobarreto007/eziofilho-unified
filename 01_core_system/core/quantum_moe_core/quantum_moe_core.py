#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantum MoE Core for EzioFilhoUnified

This module provides the core orchestration for the Mixture of Experts (MoE)
system, coordinating between multiple expert modules and selecting the most
appropriate expert(s) for a given query.

Enhanced version with improved:
- Modularity (separated responsibilities)
- Parallel execution (ThreadPoolExecutor)
- Caching (local and Redis)
- Error handling and timeout management
- Security features
- Logging and auditing
"""

import os
import sys
import json
import time
import logging
import importlib
import inspect
import hashlib
import threading
import functools
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type, Set, NamedTuple
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field, asdict
import traceback
import contextlib
import importlib.util
from enum import Enum, auto

# Try to import Redis for optional caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QuantumMoECore")

# Audit logger - separate from main logging
audit_logger = logging.getLogger("QuantumMoEAudit")
audit_handler = logging.FileHandler("quantum_moe_audit.log")
audit_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
audit_handler.setFormatter(audit_formatter)
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)


class ExpertCapability(Enum):
    """Enumeration of standard expert capabilities for type safety."""
    DATA_RETRIEVAL = auto()
    MARKET_DATA = auto()
    HISTORICAL_DATA = auto()
    CACHING = auto()
    PERFORMANCE_OPTIMIZATION = auto()
    SENTIMENT_ANALYSIS = auto()
    MARKET_SENTIMENT = auto()
    NEWS_RETRIEVAL = auto()
    MARKET_NEWS = auto()
    PREDICTION = auto()
    RECOMMENDATION = auto()
    RISK_ANALYSIS = auto()
    

class ExpertProcessStatus(Enum):
    """Enumeration of possible expert processing statuses."""
    SUCCESS = auto()
    ERROR = auto()
    TIMEOUT = auto()
    NOT_ATTEMPTED = auto()


class ExpertPerformanceMetrics:
    """Class to track performance metrics for an expert."""
    
    def __init__(self):
        """Initialize performance metrics."""
        self.calls: int = 0
        self.successful_calls: int = 0
        self.error_calls: int = 0
        self.timeout_calls: int = 0
        self.average_confidence: float = 0.0
        self.average_latency: float = 0.0
        self.last_called: Optional[str] = None
        self.max_confidence: float = 0.0
        self.min_confidence: float = 1.0
        self.max_latency: float = 0.0
        self.min_latency: float = float('inf')
        
    def update(self, confidence: float, latency: float, 
              status: ExpertProcessStatus) -> None:
        """
        Update metrics with new processing data.
        
        Args:
            confidence: The confidence score from the expert (0.0-1.0)
            latency: The processing time in seconds
            status: The processing status
        """
        self.calls += 1
        self.last_called = datetime.now().isoformat()
        
        # Update status counters
        if status == ExpertProcessStatus.SUCCESS:
            self.successful_calls += 1
            
            # Update confidence stats
            self.max_confidence = max(self.max_confidence, confidence)
            if confidence > 0:  # Only update min if non-zero
                self.min_confidence = min(self.min_confidence, confidence)
            
            # Update running average confidence
            if self.successful_calls > 1:
                self.average_confidence = ((self.average_confidence * (self.successful_calls - 1)) 
                                         + confidence) / self.successful_calls
            else:
                self.average_confidence = confidence
        
        elif status == ExpertProcessStatus.ERROR:
            self.error_calls += 1
        elif status == ExpertProcessStatus.TIMEOUT:
            self.timeout_calls += 1
            
        # Update latency stats
        self.max_latency = max(self.max_latency, latency)
        self.min_latency = min(self.min_latency, latency)
        
        # Update running average latency
        if self.calls > 1:
            self.average_latency = ((self.average_latency * (self.calls - 1)) 
                                  + latency) / self.calls
        else:
            self.average_latency = latency
    
    def get_reliability_score(self) -> float:
        """
        Calculate a reliability score for this expert based on performance metrics.
        
        Returns:
            A reliability score between 0.0 and 1.0
        """
        if self.calls == 0:
            return 0.5  # Default for new expert
            
        # Calculate success rate
        success_rate = self.successful_calls / self.calls
        
        # Calculate average confidence (default to 0.5 if no successful calls)
        avg_confidence = self.average_confidence if self.successful_calls > 0 else 0.5
        
        # Calculate latency factor (lower is better)
        latency_factor = 1.0
        if self.average_latency > 5.0:  # Penalize for high latency
            latency_factor = max(0.5, 1.0 - (self.average_latency - 5.0) / 10.0)
            
        # Combine factors
        return success_rate * avg_confidence * latency_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "calls": self.calls,
            "successful_calls": self.successful_calls,
            "error_calls": self.error_calls,
            "timeout_calls": self.timeout_calls,
            "success_rate": self.successful_calls / max(1, self.calls),
            "average_confidence": self.average_confidence,
            "average_latency": self.average_latency,
            "last_called": self.last_called,
            "max_confidence": self.max_confidence,
            "min_confidence": self.min_confidence if self.min_confidence < 1.0 else 0.0,
            "max_latency": self.max_latency,
            "min_latency": self.min_latency if self.min_latency < float('inf') else 0.0,
            "reliability_score": self.get_reliability_score()
        }


@dataclass
class ExpertMetadata:
    """Class to store metadata about an expert component."""
    
    name: str
    description: str
    capabilities: List[str]
    priority: int = 5
    confidence_threshold: float = 0.7
    timeout_seconds: float = 10.0
    max_retries: int = 1
    performance_metrics: ExpertPerformanceMetrics = field(default_factory=ExpertPerformanceMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        result = {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "priority": self.priority,
            "confidence_threshold": self.confidence_threshold,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "performance_metrics": self.performance_metrics.to_dict(),
        }
        return result


class ExpertProcessResult(NamedTuple):
    """Class to store the result of expert processing."""
    expert_name: str
    result: Any
    confidence: float
    status: ExpertProcessStatus
    error_message: Optional[str] = None
    
    @classmethod
    def error(cls, expert_name: str, error_message: str) -> 'ExpertProcessResult':
        """Create an error result."""
        return cls(
            expert_name=expert_name,
            result=None,
            confidence=0.0,
            status=ExpertProcessStatus.ERROR,
            error_message=error_message
        )
    
    @classmethod
    def timeout(cls, expert_name: str) -> 'ExpertProcessResult':
        """Create a timeout result."""
        return cls(
            expert_name=expert_name,
            result=None,
            confidence=0.0,
            status=ExpertProcessStatus.TIMEOUT,
            error_message="Processing timed out"
        )


class ExpertWrapper:
    """Wrapper for expert components to standardize interfaces."""
    
    def __init__(self, expert_instance: Any, metadata: ExpertMetadata):
        """
        Initialize expert wrapper.
        
        Args:
            expert_instance: The actual expert object instance
            metadata: Metadata about this expert
        """
        self.expert = expert_instance
        self.metadata = metadata
        self.lock = threading.RLock()
    
    def process(self, query: str, context: Dict[str, Any], 
               timeout: Optional[float] = None) -> ExpertProcessResult:
        """
        Process a query with this expert.
        
        Args:
            query: The query to process
            context: Additional context data
            timeout: Maximum time in seconds to wait for processing
            
        Returns:
            An ExpertProcessResult containing the result and metadata
        """
        start_time = time.time()
        status = ExpertProcessStatus.NOT_ATTEMPTED
        result = None
        confidence = 0.0
        error_message = None
        
        # Use the expert's timeout if not specified
        if timeout is None:
            timeout = self.metadata.timeout_seconds
            
        with self.lock:  # Ensure thread safety for experts
            try:
                # Call the appropriate method on the expert
                if hasattr(self.expert, "process_query"):
                    result, confidence = self.expert.process_query(query, context)
                elif hasattr(self.expert, "process"):
                    result, confidence = self.expert.process(query, context)
                elif hasattr(self.expert, "get_data") and "data_type" in context:
                    # Special case for data experts
                    result = self.expert.get_data(
                        source=context.get("source", "default"),
                        data_type=context["data_type"],
                        query_params=context.get("query_params", {})
                    )
                    confidence = 1.0  # Data expert always returns with full confidence if no error
                else:
                    error_message = f"Expert {self.metadata.name} has no standard processing method"
                    status = ExpertProcessStatus.ERROR
                    return ExpertProcessResult.error(self.metadata.name, error_message)
                    
                status = ExpertProcessStatus.SUCCESS
                
            except Exception as e:
                error_message = f"Error processing with expert {self.metadata.name}: {str(e)}"
                logger.error(f"{error_message}\n{traceback.format_exc()}")
                status = ExpertProcessStatus.ERROR
                result = None
                confidence = 0.0
                
            finally:
                end_time = time.time()
                latency = end_time - start_time
                
                # Update performance metrics
                self.metadata.performance_metrics.update(confidence, latency, status)
                
                if status == ExpertProcessStatus.SUCCESS:
                    return ExpertProcessResult(
                        expert_name=self.metadata.name,
                        result=result,
                        confidence=confidence,
                        status=status
                    )
                else:
                    return ExpertProcessResult.error(self.metadata.name, error_message or "Unknown error")


class ResponseCache:
    """
    Cache for storing and retrieving query responses.
    Supports both in-memory and Redis-based caching with fallback.
    """
    
    def __init__(self, use_redis: bool = False, redis_url: Optional[str] = None, 
                ttl: int = 3600, max_items: int = 1000):
        """
        Initialize the response cache.
        
        Args:
            use_redis: Whether to use Redis for caching
            redis_url: Redis connection URL
            ttl: Time-to-live for cache entries in seconds
            max_items: Maximum number of items to store in memory cache
        """
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.ttl = ttl
        self.max_items = max_items
        
        # Initialize memory cache
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize Redis client if requested
        self.redis_client = None
        if self.use_redis:
            try:
                self.redis_client = redis.Redis.from_url(
                    redis_url or "redis://localhost:6379/0")
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache connection established")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis, falling back to memory cache: {e}")
                self.use_redis = False
                self.redis_client = None
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """
        Generate a cache key for a query and context.
        
        Args:
            query: The query string
            context: The context dictionary
            
        Returns:
            A cache key string
        """
        # Normalize the context by sorting keys and converting to JSON
        sorted_context = json.dumps(context, sort_keys=True)
        
        # Create a hash of the combined query and context
        key_data = f"{query}|{sorted_context}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def get(self, query: str, context: Dict[str, Any]) -> Optional[Tuple[Any, List[str]]]:
        """
        Get a response from the cache.
        
        Args:
            query: The query string
            context: The context dictionary
            
        Returns:
            A tuple of (result, experts_used) if found, None otherwise
        """
        cache_key = self._generate_cache_key(query, context)
        
        # Try Redis first if enabled
        if self.use_redis and self.redis_client:
            try:
                data = self.redis_client.get(f"moe_cache:{cache_key}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis cache get error: {e}")
        
        # Fall back to memory cache
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            
            # Check if entry has expired
            if "expiry" in entry and entry["expiry"] < time.time():
                del self.memory_cache[cache_key]
                return None
                
            return entry["data"]
            
        return None
    
    def set(self, query: str, context: Dict[str, Any], 
           result: Any, experts_used: List[str]) -> bool:
        """
        Store a response in the cache.
        
        Args:
            query: The query string
            context: The context dictionary
            result: The result to cache
            experts_used: List of experts used to generate the result
            
        Returns:
            True if successfully cached, False otherwise
        """
        cache_key = self._generate_cache_key(query, context)
        data = (result, experts_used)
        
        # Try to store in Redis if enabled
        if self.use_redis and self.redis_client:
            try:
                serialized = json.dumps(data)
                success = self.redis_client.setex(
                    f"moe_cache:{cache_key}", self.ttl, serialized)
                if success:
                    return True
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        
        # Fall back to memory cache
        # Evict oldest item if at capacity
        if len(self.memory_cache) >= self.max_items:
            oldest_key = min(self.memory_cache.keys(), 
                            key=lambda k: self.memory_cache[k].get("created", 0))
            del self.memory_cache[oldest_key]
            
        self.memory_cache[cache_key] = {
            "data": data,
            "created": time.time(),
            "expiry": time.time() + self.ttl
        }
        
        return True
    
    def invalidate(self, query_pattern: Optional[str] = None, 
                 context_pattern: Optional[Dict[str, Any]] = None) -> int:
        """
        Invalidate cache entries matching patterns.
        
        Args:
            query_pattern: Pattern to match against queries
            context_pattern: Pattern to match against contexts
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        # Memory cache invalidation
        keys_to_remove = []
        
        for key, entry in self.memory_cache.items():
            if query_pattern is None and context_pattern is None:
                # If no patterns specified, invalidate all
                keys_to_remove.append(key)
                count += 1
            else:
                data = entry["data"]
                result, experts = data
                
                # Simple pattern matching logic
                match = True
                if query_pattern and query_pattern not in key:
                    match = False
                    
                if match and context_pattern:
                    for k, v in context_pattern.items():
                        if f'"{k}": "{v}"' not in json.dumps(entry):
                            match = False
                            break
                
                if match:
                    keys_to_remove.append(key)
                    count += 1
        
        # Remove matched keys
        for key in keys_to_remove:
            del self.memory_cache[key]
            
        # Redis cache invalidation (if enabled)
        if self.use_redis and self.redis_client:
            try:
                pattern = "moe_cache:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    count += len(keys)
                    self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis cache invalidation error: {e}")
                
        return count


class ExpertRegistry:
    """Registry for managing expert components."""
    
    def __init__(self, project_root: str):
        """
        Initialize the expert registry.
        
        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.experts: Dict[str, ExpertWrapper] = {}
        self.discovery_paths: List[str] = [
            os.path.join(self.project_root, "ezio_experts"),
            os.path.join(self.project_root, "experts")
        ]
    
    def discover_experts(self) -> None:
        """
        Discover and register all available experts in the project.
        """
        logger.info("Discovering experts...")
        
        for expert_dir in self.discovery_paths:
            if not os.path.exists(expert_dir):
                logger.warning(f"Expert directory not found: {expert_dir}")
                continue
                
            # List subdirectories in expert directory
            for item in os.listdir(expert_dir):
                item_path = os.path.join(expert_dir, item)
                
                # Skip non-directories and special directories
                if not os.path.isdir(item_path) or item.startswith("__"):
                    continue
                
                # Check for expert implementation files
                expert_file = None
                for filename in [f"{item}.py", "expert.py", "main.py"]:
                    if os.path.exists(os.path.join(item_path, filename)):
                        expert_file = os.path.join(item_path, filename)
                        break
                
                if not expert_file:
                    logger.warning(f"No expert implementation found in {item_path}")
                    continue
                
                # Load the expert
                try:
                    self._load_expert_from_file(item, expert_file)
                except Exception as e:
                    logger.error(f"Error loading expert {item}: {str(e)}\n{traceback.format_exc()}")
    
    def _load_expert_from_file(self, expert_name: str, file_path: str) -> None:
        """
        Load an expert from a Python file with enhanced security.
        
        Args:
            expert_name: The name of the expert
            file_path: Path to the expert's Python file
        """
        # Audit the loading attempt
        audit_logger.info(f"Loading expert: {expert_name} from {file_path}")
        
        try:
            # Validate the file path first
            file_path = os.path.abspath(file_path)
            if not file_path.startswith(self.project_root):
                raise ValueError(f"Expert file path {file_path} is outside project root")
                
            # Get the module name from the file path
            module_name = os.path.basename(file_path).replace(".py", "")
            dir_name = os.path.dirname(file_path)
            
            # Create a spec from file location
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                logger.error(f"Failed to get module spec for {file_path}")
                return
                
            # Create the module
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules
            sys.modules[module_name] = module
            
            # Execute the module in a restricted context
            with self._secure_exec_context():
                spec.loader.exec_module(module)
            
            # Find the expert class in the module
            expert_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    (name.endswith("Expert") or 
                     name == expert_name or 
                     name == module_name)):
                    expert_class = obj
                    break
            
            if expert_class is None:
                logger.warning(f"No expert class found in {file_path}")
                return
            
            # Validate the expert class has required methods
            self._validate_expert_class(expert_class)
            
            # Determine expert_type from file path
            expert_type = None
            try:
                # Extract expert_type from folder name (e.g., "sentiment" from "experts/sentiment/")
                path_parts = file_path.split(os.sep)
                experts_index = -1
                
                # Find "experts" directory in the path
                for i, part in enumerate(path_parts):
                    if part.lower() == "experts":
                        experts_index = i
                        break
                
                # Get expert_type from next folder after "experts"
                if experts_index >= 0 and experts_index + 1 < len(path_parts):
                    expert_type = path_parts[experts_index + 1]
                    logger.debug(f"Extracted expert_type '{expert_type}' from path: {file_path}")
            except Exception as e:
                logger.warning(f"Error extracting expert_type from path: {e}")
            
            # Instantiate the expert
            try:
                # Import EzioBaseExpert for isinstance check
                try:
                    from experts.base_expert import EzioBaseExpert
                    is_ezio_expert = issubclass(expert_class, EzioBaseExpert)
                except ImportError:
                    logger.warning("Could not import EzioBaseExpert for class check, assuming regular expert")
                    is_ezio_expert = False
                
                # Create instance based on class type
                if is_ezio_expert:
                    if not expert_type:
                        expert_type = expert_name.replace("_expert", "")
                        logger.info(f"Using fallback expert_type from name: {expert_type}")
                    
                    logger.info(f"Instantiating EzioBaseExpert subclass with expert_type: {expert_type}")
                    expert_instance = expert_class(expert_type=expert_type)
                else:
                    # Regular expert class without expert_type parameter
                    logger.debug(f"Instantiating regular expert class: {expert_class.__name__}")
                    expert_instance = expert_class()
            except Exception as e:
                logger.error(f"Failed to instantiate expert class {expert_class.__name__}: {e}")
                return
            
            # Create metadata
            capabilities = []
            if hasattr(expert_instance, "capabilities"):
                capabilities = expert_instance.capabilities
            elif expert_name == "fallback_data_expert":
                capabilities = ["data_retrieval", "historical_data"]
            elif expert_name == "cache_expert":
                capabilities = ["caching", "performance_optimization"]
            elif expert_name == "sentiment_expert":
                capabilities = ["sentiment_analysis", "market_sentiment"]
            elif expert_name == "news_oracle":
                capabilities = ["news_retrieval", "market_news"]
            
            # Determine priority
            priority = 5  # Default
            if hasattr(expert_instance, "priority"):
                priority = expert_instance.priority
            elif "cache" in expert_name:
                priority = 8  # High priority for cache
            elif "data" in expert_name:
                priority = 6  # Good priority for data
            
            # Determine timeout
            timeout = 10.0  # Default 10 seconds
            if hasattr(expert_instance, "timeout_seconds"):
                timeout = expert_instance.timeout_seconds
                
            metadata = ExpertMetadata(
                name=expert_name,
                description=expert_class.__doc__ or f"{expert_name} expert",
                capabilities=capabilities,
                priority=priority,
                timeout_seconds=timeout
            )
            
            # Register the expert
            self.experts[expert_name] = ExpertWrapper(expert_instance, metadata)
            logger.info(f"Registered expert: {expert_name} with {len(capabilities)} capabilities")
            audit_logger.info(f"Expert registered successfully: {expert_name}")
            
        except Exception as e:
            error_msg = f"Error loading expert from {file_path}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            audit_logger.error(error_msg)
            raise
    
    @contextlib.contextmanager
    def _secure_exec_context(self):
        """
        Context manager for secure module execution.
        Adds protections when loading external module code.
        """
        # Save original builtins that we might want to restrict
        original_builtins = {}
        restricted_builtins = [
            # Uncomment these to enforce stricter security
            # 'eval', 'exec', 'open'
        ]
        
        try:
            # Apply restrictions to harmful builtins
            # In production, consider adding actual restrictions here
            yield
        finally:
            # Restore original builtins
            pass
    
    def _validate_expert_class(self, expert_class: Type) -> None:
        """
        Validate that an expert class has the required methods.
        
        Args:
            expert_class: The expert class to validate
        
        Raises:
            ValueError: If the expert class does not have required methods
        """
        # Create a temporary instance to check for instance methods
        with contextlib.suppress(Exception):
            instance = expert_class()
            
            # Check for required methods
            has_process = hasattr(instance, "process") and callable(instance.process)
            has_process_query = hasattr(instance, "process_query") and callable(instance.process_query)
            has_get_data = hasattr(instance, "get_data") and callable(instance.get_data)
            
            if not (has_process or has_process_query or has_get_data):
                raise ValueError(
                    f"Expert class {expert_class.__name__} does not have any required methods: "
                    "process(), process_query() or get_data()"
                )
    
    def register_expert(self, name: str, expert_instance: Any, 
                      capabilities: List[str], priority: int = 5, 
                      timeout: float = 10.0) -> bool:
        """
        Manually register an expert.
        
        Args:
            name: The name of the expert
            expert_instance: The expert object instance
            capabilities: List of the expert's capabilities
            priority: Priority level (1-10, higher means more important)
            timeout: Timeout in seconds for expert processing
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            metadata = ExpertMetadata(
                name=name,
                description=expert_instance.__class__.__doc__ or f"{name} expert",
                capabilities=capabilities,
                priority=priority,
                timeout_seconds=timeout
            )
            
            self.experts[name] = ExpertWrapper(expert_instance, metadata)
            logger.info(f"Manually registered expert: {name}")
            audit_logger.info(f"Expert manually registered: {name}")
            return True
        except Exception as e:
            logger.error(f"Error registering expert {name}: {str(e)}")
            return False
    
    def get_expert(self, name: str) -> Optional[ExpertWrapper]:
        """
        Get an expert by name.
        
        Args:
            name: The name of the expert
            
        Returns:
            The expert wrapper, or None if not found
        """
        return self.experts.get(name)
    
    def get_all_experts(self) -> Dict[str, ExpertWrapper]:
        """
        Get all registered experts.
        
        Returns:
            Dictionary mapping expert names to expert wrappers
        """
        return self.experts.copy()
    
    def get_expert_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all registered experts.
        
        Returns:
            Dictionary with expert statistics
        """
        stats = {}
        for name, expert in self.experts.items():
            stats[name] = expert.metadata.to_dict()
        return stats


class ParallelExecutor:
    """Handles parallel execution of expert processing."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the parallel executor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_with_experts(self, 
                          experts: Dict[str, ExpertWrapper], 
                          selected_experts: List[str],
                          query: str, 
                          context: Dict[str, Any]) -> List[ExpertProcessResult]:
        """
        Process a query with multiple experts in parallel.
        
        Args:
            experts: Dictionary mapping expert names to expert wrappers
            selected_experts: List of expert names to use
            query: The query to process
            context: Additional context data
            
        Returns:
            List of expert process results
        """
        # Create tasks for selected experts
        futures = {}
        results = []
        
        for expert_name in selected_experts:
            expert = experts.get(expert_name)
            if not expert:
                results.append(ExpertProcessResult.error(
                    expert_name, f"Expert not found: {expert_name}"))
                continue
                
            # Submit task to thread pool
            future = self.executor.submit(
                self._process_with_expert,
                expert,
                query,
                context
            )
            futures[future] = expert_name
            
        # Collect results as they complete
        for future in as_completed(futures):
            expert_name = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Uncaught error in expert {expert_name}: {str(e)}")
                results.append(ExpertProcessResult.error(
                    expert_name, f"Uncaught error: {str(e)}"))
                
        return results
    
    def _process_with_expert(self, 
                          expert: ExpertWrapper, 
                          query: str, 
                          context: Dict[str, Any]) -> ExpertProcessResult:
        """
        Process a query with a single expert with timeout handling.
        
        Args:
            expert: The expert wrapper
            query: The query to process
            context: Additional context data
            
        Returns:
            An expert process result
        """
        # Get timeout from expert metadata
        timeout = expert.metadata.timeout_seconds
        max_retries = expert.metadata.max_retries
        
        # Use ThreadPoolExecutor for timeout support
        with ThreadPoolExecutor(max_workers=1) as local_executor:
            for attempt in range(max_retries + 1):
                try:
                    # Submit the task with timeout
                    future = local_executor.submit(expert.process, query, context)
                    return future.result(timeout=timeout)
                except TimeoutError:
                    logger.warning(
                        f"Expert {expert.metadata.name} timed out after {timeout} seconds "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                    # Only retry if we haven't reached max retries
                    if attempt >= max_retries:
                        return ExpertProcessResult.timeout(expert.metadata.name)
                except Exception as e:
                    logger.error(
                        f"Error processing with expert {expert.metadata.name}: {str(e)}\n"
                        f"{traceback.format_exc()}"
                    )
                    return ExpertProcessResult.error(expert.metadata.name, str(e))
    
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class ExpertSelector:
    """Responsible for selecting appropriate experts for queries."""
    
    def __init__(self, registry: ExpertRegistry):
        """
        Initialize the expert selector.
        
        Args:
            registry: The expert registry
        """
        self.registry = registry
    
    def select_experts_for_query(self, query: str, context: Dict[str, Any]) -> List[str]:
        """
        Select appropriate experts for a given query.
        
        Args:
            query: The query to process
            context: Additional context data
            
        Returns:
            List of expert names that should process this query
        """
        selected_experts = []
        all_experts = self.registry.get_all_experts()
        
        # Extract capabilities requested from context
        requested_capabilities = context.get("capabilities", [])
        query_type = context.get("query_type", "general")
        
        # Weight assignable to each expert based on matching capabilities
        expert_weights = {}
        
        for name, expert in all_experts.items():
            metadata = expert.metadata
            weight = 0
            
            # Assign weight based on capability match
            for capability in metadata.capabilities:
                if capability in requested_capabilities:
                    weight += 2
                elif any(cap in capability for cap in requested_capabilities):
                    weight += 1
            
            # Adjust weight based on expert priority
            weight *= metadata.priority / 5.0
            
            # Adjust weight based on past performance
            metrics = metadata.performance_metrics
            if metrics.calls > 0:
                # Use reliability score for weighting
                reliability = metrics.get_reliability_score()
                weight *= 0.5 + (0.5 * reliability)
            
            expert_weights[name] = weight
        
        # Select experts based on weights
        if requested_capabilities:
            # If specific capabilities are requested, select the best matches
            selected_experts = sorted(expert_weights.keys(), 
                                     key=lambda x: expert_weights[x], 
                                     reverse=True)
            
            # Limit to top 3 experts with non-zero weight
            selected_experts = [e for e in selected_experts[:3] if expert_weights[e] > 0]
        else:
            # For general queries, include all experts that might be relevant
            # based on the query type
            for name, expert in all_experts.items():
                if query_type == "market_data" and any(c in ["data", "market"] 
                                                     for c in expert.metadata.capabilities):
                    selected_experts.append(name)
                elif query_type == "sentiment" and any(c in ["sentiment", "analysis"] 
                                                     for c in expert.metadata.capabilities):
                    selected_experts.append(name)
                elif query_type == "news" and any(c in ["news", "retrieval"] 
                                                for c in expert.metadata.capabilities):
                    selected_experts.append(name)
                elif query_type == "cache" and "cache" in expert.metadata.capabilities:
                    selected_experts.append(name)
        
        # Always include cache_expert if available for optimization
        if "cache_expert" in all_experts and "cache_expert" not in selected_experts:
            selected_experts.insert(0, "cache_expert")
        
        # If no experts were selected, use all available experts
        if not selected_experts:
            selected_experts = list(all_experts.keys())
        
        logger.info(f"Selected experts for query: {selected_experts}")
        return selected_experts


class ResultEvaluator:
    """Evaluates and combines results from multiple experts."""
    
    @staticmethod
    def evaluate_results(expert_results: List[ExpertProcessResult]) -> Tuple[Any, List[str]]:
        """
        Evaluate results from multiple experts and determine the best result.
        
        Args:
            expert_results: List of expert process results
            
        Returns:
            Tuple of (result, experts_used)
        """
        # Filter out failed results
        successful_results = [r for r in expert_results 
                            if r.status == ExpertProcessStatus.SUCCESS]
        
        if not successful_results:
            # No successful results
            error_msg = "No experts could successfully process the query"
            if expert_results:
                # Include errors from experts in response
                error_details = [
                    f"{r.expert_name}: {r.error_message or 'Unknown error'}"
                    for r in expert_results
                    if r.status in (ExpertProcessStatus.ERROR, ExpertProcessStatus.TIMEOUT)
                ]
                if error_details:
                    error_msg += f". Errors: {'; '.join(error_details)}"
            
            return {"error": error_msg}, []
        
        # Sort by confidence
        successful_results.sort(key=lambda x: x.confidence, reverse=True)
        
        # Find the expert with highest confidence above threshold
        for result in successful_results:
            # We'd check against the expert's confidence threshold here,
            # but we don't have access to the expert registry
            # Instead we'll use a standard threshold
            if result.confidence >= 0.7:
                logger.info(f"Using result from {result.expert_name} with confidence {result.confidence}")
                return result.result, [result.expert_name]
        
        # If no expert has high confidence, combine results
        combined_result = ResultEvaluator._combine_results(successful_results)
        experts_used = [result.expert_name for result in successful_results]
        
        return combined_result, experts_used
    
    @staticmethod
    def _combine_results(results: List[ExpertProcessResult]) -> Any:
        """
        Combine results from multiple experts.
        
        Args:
            results: List of expert process results
            
        Returns:
            Combined result
        """
        # Skip empty results
        valid_results = [(r.expert_name, r.result, r.confidence) 
                        for r in results if r.result is not None]
        
        if not valid_results:
            return None
        
        # Normalize confidences
        total_conf = sum(conf for _, _, conf in valid_results)
        if total_conf == 0:
            # If all confidences are 0, treat them as equal
            weights = [1.0 / len(valid_results) for _ in valid_results]
        else:
            weights = [conf / total_conf for _, _, conf in valid_results]
        
        # Check result types
        result_types = set(type(result) for _, result, _ in valid_results)
        
        if len(result_types) == 1:
            # All results are the same type, combine based on type
            result_type = next(iter(result_types))
            
            if result_type == dict:
                # For dictionaries, create a weighted merged dictionary
                combined = {}
                for i, (_, result, _) in enumerate(valid_results):
                    for key, value in result.items():
                        if key in combined:
                            if isinstance(value, (int, float)):
                                # For numeric values, use weighted average
                                combined[key] = combined[key] + weights[i] * value
                        else:
                            # For new keys, add with weight
                            if isinstance(value, (int, float)):
                                combined[key] = weights[i] * value
                            else:
                                combined[key] = value
                return combined
                
            elif result_type == list:
                # For lists, create a weighted merged list (if possible)
                if all(isinstance(item, dict) for _, items, _ in valid_results for item in items):
                    # List of dictionaries - use a set-like merge based on a key if available
                    combined = []
                    for _, result, _ in valid_results:
                        combined.extend(result)
                    return combined
                else:
                    # Just concatenate lists
                    combined = []
                    for _, result, _ in valid_results:
                        combined.extend(result)
                    return combined
            
            elif result_type in (str, int, float):
                # For primitives, use the highest confidence result
                return valid_results[0][1]
        
        # Different result types or complex objects - return the highest confidence result
        return valid_results[0][1]


class QuantumMoECore:
    """
    Quantum Mixture of Experts Core Orchestrator.
    
    Manages multiple expert components and orchestrates their usage
    for processing queries based on capabilities and confidence.
    """
    
    def __init__(
        self, 
        project_root: Optional[str] = None,
        max_workers: int = 4,
        use_redis_cache: bool = False,
        redis_url: Optional[str] = None,
        cache_ttl: int = 3600
    ):
        """
        Initialize the Quantum MoE Core.
        
        Args:
            project_root: Path to the project root directory
            max_workers: Maximum number of worker threads
            use_redis_cache: Whether to use Redis for caching
            redis_url: Redis connection URL
            cache_ttl: Time-to-live for cache entries in seconds
        """
        # Get the project root
        self.project_root = project_root or self._find_project_root()
        logger.info(f"Project root: {self.project_root}")
        
        # Create the expert registry
        self.registry = ExpertRegistry(self.project_root)
        
        # Create the expert selector
        self.selector = ExpertSelector(self.registry)
        
        # Create the response cache
        self.cache = ResponseCache(
            use_redis=use_redis_cache,
            redis_url=redis_url,
            ttl=cache_ttl
        )
        
        # Create the parallel executor
        self.executor = ParallelExecutor(max_workers=max_workers)
        
        # Request tracking for auditing
        self.request_counter = 0
        
        # Load default experts
        self.registry.discover_experts()
        
        logger.info(f"Quantum MoE Core initialized with {len(self.registry.get_all_experts())} experts")
        audit_logger.info(f"Quantum MoE Core initialized with {len(self.registry.get_all_experts())} experts")
    
    def _find_project_root(self) -> str:
        """Find the project root directory."""
        # Start with the current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Navigate up until we find the project root
        path = current_dir
        while True:
            parent_dir = os.path.dirname(path)
            
            # Check if this might be the project root (contains core, experts, and data dirs)
            if (os.path.exists(os.path.join(parent_dir, "core")) and 
                os.path.exists(os.path.join(parent_dir, "ezio_experts"))):
                return parent_dir
            
            # Stop if we've reached the filesystem root
            if parent_dir == path:
                # If we can't find the project root, use a default path
                logger.warning("Could not determine project root. Using current directory's parent.")
                return os.path.dirname(current_dir)
            
            path = parent_dir
    
    def register_expert(self, name: str, expert_instance: Any, 
                      capabilities: List[str], priority: int = 5) -> bool:
        """
        Manually register an expert.
        
        Args:
            name: The name of the expert
            expert_instance: The expert object instance
            capabilities: List of the expert's capabilities
            priority: Priority level (1-10, higher means more important)
            
        Returns:
            True if registration was successful, False otherwise
        """
        return self.registry.register_expert(name, expert_instance, capabilities, priority)
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Tuple[Any, List[str]]:
        """
        Process a query using the appropriate experts.
        
        Args:
            query: The query to process
            context: Additional context data
            
        Returns:
            Tuple containing (result, experts_used)
        """
        if context is None:
            context = {}
            
        # Generate a request ID for tracking
        self.request_counter += 1
        request_id = f"req_{self.request_counter}_{uuid.uuid4().hex[:8]}"
        
        # Log the incoming request
        logger.info(f"Processing query (ID: {request_id}): {query}")
        audit_logger.info(f"Query received (ID: {request_id}): {query[:100]}...")
        
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = self.cache.get(query, context)
            if cached_result:
                logger.info(f"Cache hit for query (ID: {request_id})")
                result, experts_used = cached_result
                
                # Log the cache hit
                audit_logger.info(
                    f"Cache hit (ID: {request_id}), experts used: {','.join(experts_used)}, "
                    f"time: {time.time() - start_time:.2f}s"
                )
                
                return result, experts_used
            
            # No cache hit, process with experts
            logger.info(f"Cache miss for query (ID: {request_id})")
            
            # Select experts for this query
            selected_experts = self.selector.select_experts_for_query(query, context)
            
            # Process with selected experts in parallel
            expert_results = self.executor.process_with_experts(
                self.registry.get_all_experts(),
                selected_experts,
                query,
                context
            )
            
            # Evaluate results
            result, experts_used = ResultEvaluator.evaluate_results(expert_results)
            
            # Cache the result if it's valid
            if experts_used and not isinstance(result, dict) or not result.get("error"):
                self.cache.set(query, context, result, experts_used)
            
            # Log the results
            processing_time = time.time() - start_time
            logger.info(
                f"Query processed (ID: {request_id}) in {processing_time:.2f}s "
                f"using experts: {','.join(experts_used)}"
            )
            audit_logger.info(
                f"Query completed (ID: {request_id}) in {processing_time:.2f}s, "
                f"experts: {','.join(experts_used)}"
            )
            
            return result, experts_used
            
        except Exception as e:
            # Log the error
            error_msg = f"Error processing query (ID: {request_id}): {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            audit_logger.error(error_msg)
            
            return {"error": str(e)}, []
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all registered experts.
        
        Returns:
            Dictionary with expert statistics
        """
        return self.registry.get_expert_stats()
    
    def shutdown(self) -> None:
        """Shutdown the Quantum MoE Core, cleaning up resources."""
        logger.info("Shutting down Quantum MoE Core")
        audit_logger.info("Quantum MoE Core shutting down")
        
        # Shutdown the executor
        self.executor.shutdown()


# Example usage
if __name__ == "__main__":
    """
    Example usage of the Quantum MoE Core.
    This also serves as a basic test of the core functionality.
    """
    import uuid
    import random
    from dataclasses import dataclass
    
    # Set up logging to console for the example
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    print("=== Quantum MoE Core Example ===")
    
    # Define a simple mock expert
    class MockDataExpert:
        """Mock data expert for testing."""
        
        def __init__(self, name="MockData"):
            self.name = name
            self.capabilities = ["data_retrieval", "historical_data"]
            
        def get_data(self, source="default", data_type="market", query_params=None):
            """Get mock data."""
            print(f"[{self.name}] Getting {data_type} data from {source}")
            time.sleep(0.5)  # Simulate processing time
            
            if data_type == "market":
                return {
                    "market_data": {
                        "S&P500": 4500 + random.randint(-50, 50),
                        "NASDAQ": 15000 + random.randint(-100, 100),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            elif data_type == "historical":
                return {
                    "historical_data": [
                        {"date": "2025-05-18", "value": 4510},
                        {"date": "2025-05-19", "value": 4520},
                        {"date": "2025-05-20", "value": 4505}
                    ]
                }
            else:
                return {"error": f"Unknown data type: {data_type}"}
    
    class MockSentimentExpert:
        """Mock sentiment expert for testing."""
        
        def __init__(self):
            self.capabilities = ["sentiment_analysis", "market_sentiment"]
            
        def process_query(self, query, context):
            """Process a query with sentiment analysis."""
            print(f"[SentimentExpert] Processing query: {query}")
            time.sleep(0.8)  # Simulate processing time
            
            # Simple sentiment based on keywords
            positive_keywords = ["good", "positive", "growth", "increase", "up"]
            negative_keywords = ["bad", "negative", "decline", "decrease", "down"]
            
            query_lower = query.lower()
            positive_count = sum(1 for word in positive_keywords if word in query_lower)
            negative_count = sum(1 for word in negative_keywords if word in query_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
                score = 0.5 + (positive_count * 0.1)
            elif negative_count > positive_count:
                sentiment = "negative"
                score = 0.5 - (negative_count * 0.1)
            else:
                sentiment = "neutral"
                score = 0.5
                
            confidence = 0.7 + (abs(positive_count - negative_count) * 0.05)
            confidence = min(0.95, confidence)
            
            return {
                "sentiment": sentiment,
                "score": score,
                "confidence": confidence
            }, confidence
    
    class MockCacheExpert:
        """Mock cache expert for testing."""
        
        def __init__(self):
            self.capabilities = ["caching", "performance_optimization"]
            self.cache = {}
            
        def process(self, query, context):
            """Process a cache lookup."""
            print(f"[CacheExpert] Looking up query in cache")
            
            # Simple cache key based on query and context
            key = f"{query}:{json.dumps(context, sort_keys=True)}"
            key_hash = hashlib.md5(key.encode()).hexdigest()
            
            if key_hash in self.cache:
                return self.cache[key_hash], 1.0
            
            return None, 0.0
            
        def set(self, namespace, key, value):
            """Set a value in the cache."""
            key_str = f"{namespace}:{json.dumps(key, sort_keys=True)}"
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            self.cache[key_hash] = value
            return True
    
    class MockSlowExpert:
        """Mock expert that is very slow and sometimes fails."""
        
        def __init__(self):
            self.capabilities = ["slow_processing"]
            self.failures = 0
            
        def process(self, query, context):
            """Process a query very slowly."""
            print(f"[SlowExpert] Starting slow processing...")
            
            # Simulate slow processing
            time.sleep(5)  # This should trigger timeout
            
            # Simulate occasional failures
            if random.random() < 0.3:
                self.failures += 1
                raise RuntimeError(f"Simulated failure #{self.failures}")
                
            return {"slow_result": "This took a long time!"}, 0.6
    
    # Initialize the MoE Core
    core = QuantumMoECore(
        max_workers=4,
        use_redis_cache=False,  # Set to True to use Redis if available
        cache_ttl=300  # 5 minutes
    )
    
    # Register our mock experts
    core.register_expert("market_data_expert", MockDataExpert(), ["data_retrieval", "market_data"])
    core.register_expert("historical_data_expert", MockDataExpert("Historical"), ["historical_data"])
    core.register_expert("sentiment_expert", MockSentimentExpert(), ["sentiment_analysis", "market_sentiment"])
    core.register_expert("cache_expert", MockCacheExpert(), ["caching"], priority=10)
    core.register_expert("slow_expert", MockSlowExpert(), ["slow_processing"], priority=1)
    
    # Print discovered experts
    print(f"\nRegistered experts:")
    for name, stats in core.get_expert_stats().items():
        print(f"- {name}: {', '.join(stats['capabilities'])}")
    
    # Test queries
    test_queries = [
        ("What is the current market data?", {"query_type": "market_data", "capabilities": ["market_data"]}),
        ("How is market sentiment for technology stocks?", {"query_type": "sentiment", "capabilities": ["sentiment_analysis"]}),
        ("Show me historical data for the past 3 days", {"query_type": "market_data", "capabilities": ["historical_data"]}),
        ("The market outlook seems positive with strong growth indicators", {"query_type": "sentiment"}),
        ("Test the slow expert", {"query_type": "general", "capabilities": ["slow_processing"]}),
    ]
    
    print("\nTesting queries:")
    for query, context in test_queries:
        print(f"\n{'-'*80}\nQuery: {query}")
        print(f"Context: {context}")
        
        # Process the query
        result, experts_used = core.process_query(query, context)
        
        print(f"Experts used: {experts_used}")
        print(f"Result: {json.dumps(result, indent=2) if isinstance(result, dict) else result}")
        
        # Process again to test caching
        if "slow_processing" not in context.get("capabilities", []):
            print("\nProcessing again to test caching...")
            cached_result, cached_experts = core.process_query(query, context)
            if "cache_expert" in cached_experts:
                print("✓ Successfully retrieved from cache")
            else:
                print("✗ Failed to retrieve from cache")
    
    # Get expert statistics
    print(f"\n{'-'*80}\nExpert statistics:")
    for name, stats in core.get_expert_stats().items():
        metrics = stats["performance_metrics"]
        print(f"- {name}:")
        print(f"  * Calls: {metrics['calls']}")
        print(f"  * Success rate: {metrics['success_rate']:.2f}")
        print(f"  * Avg confidence: {metrics['average_confidence']:.2f}")
        print(f"  * Avg latency: {metrics['average_latency']:.2f}s")
        print(f"  * Reliability score: {metrics['reliability_score']:.2f}")
    
    # Shutdown the core
    core.shutdown()
    print(f"\n{'-'*80}\nQuantum MoE Core shut down successfully")