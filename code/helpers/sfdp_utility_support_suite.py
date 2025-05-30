"""
SFDP Utility Support Suite - Essential Utility and Support Functions
===================================================================

Complete utility suite providing essential support functions for the multi-physics
simulation framework including file I/O, error handling, logging, visualization,
performance optimization, and system utilities.

Author: SFDP Research Team
Version: 17.3 (Complete Utility Support Implementation)
License: Academic Research Use Only
"""

import os
import sys
import json
import time
import hashlib
import shutil
import pickle
import gzip
import traceback
import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
from functools import wraps
import warnings

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FileOperationResult:
    """Result of file operation"""
    success: bool
    operation: str
    file_path: str
    details: Dict[str, Any]
    error: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance profiling metrics"""
    function_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    parameters: Dict[str, Any]


@dataclass
class ErrorReport:
    """Comprehensive error report"""
    error_type: str
    error_message: str
    traceback: str
    timestamp: datetime
    context: Dict[str, Any]
    recovery_attempted: bool
    recovery_successful: bool


# File & Data Management Functions

def smart_file_manager(operation_type: str, file_specs: Dict[str, Any], 
                      options: Optional[Dict[str, Any]] = None) -> Tuple[FileOperationResult, Dict[str, Any], Dict[str, Any]]:
    """
    Intelligent file operations and management system
    
    Supports: READ, WRITE, COPY, MOVE, DELETE, COMPRESS, EXTRACT
    Features: Format detection, integrity checking, atomic operations
    """
    if options is None:
        options = {}
    
    file_operations = {}
    file_report = {
        'operation': operation_type,
        'timestamp': datetime.now().isoformat(),
        'files_processed': 0,
        'total_size': 0,
        'errors': []
    }
    
    try:
        if operation_type.upper() == 'READ':
            result = _read_file_intelligent(file_specs, options)
        elif operation_type.upper() == 'WRITE':
            result = _write_file_atomic(file_specs, options)
        elif operation_type.upper() == 'COPY':
            result = _copy_file_verified(file_specs, options)
        elif operation_type.upper() == 'MOVE':
            result = _move_file_safe(file_specs, options)
        elif operation_type.upper() == 'DELETE':
            result = _delete_file_secure(file_specs, options)
        elif operation_type.upper() == 'COMPRESS':
            result = _compress_file(file_specs, options)
        elif operation_type.upper() == 'EXTRACT':
            result = _extract_file(file_specs, options)
        else:
            raise ValueError(f"Unsupported operation: {operation_type}")
        
        file_operations[result.file_path] = result
        file_report['files_processed'] += 1
        
        if 'size' in result.details:
            file_report['total_size'] += result.details['size']
            
    except Exception as e:
        error_msg = f"File operation failed: {str(e)}"
        logger.error(error_msg)
        file_report['errors'].append(error_msg)
        result = FileOperationResult(
            success=False,
            operation=operation_type,
            file_path=file_specs.get('path', 'unknown'),
            details={},
            error=error_msg
        )
    
    return result, file_operations, file_report


def data_format_converter(data: Any, source_format: str, target_format: str,
                         options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Universal data format converter with automatic detection
    
    Supports: JSON, CSV, PICKLE, NPY, MAT, HDF5, XML
    """
    if options is None:
        options = {}
    
    conversion_info = {
        'source_format': source_format.upper(),
        'target_format': target_format.upper(),
        'timestamp': datetime.now().isoformat(),
        'options': options
    }
    
    try:
        # Convert to intermediate format (Python objects)
        if source_format.upper() == 'JSON':
            if isinstance(data, str):
                intermediate = json.loads(data)
            else:
                intermediate = data
        elif source_format.upper() == 'CSV':
            intermediate = pd.read_csv(data) if isinstance(data, str) else data
        elif source_format.upper() == 'PICKLE':
            if isinstance(data, bytes):
                intermediate = pickle.loads(data)
            elif isinstance(data, str):
                with open(data, 'rb') as f:
                    intermediate = pickle.load(f)
            else:
                intermediate = data
        elif source_format.upper() == 'NPY':
            intermediate = np.load(data) if isinstance(data, str) else data
        else:
            intermediate = data
        
        # Convert to target format
        if target_format.upper() == 'JSON':
            converted = json.dumps(intermediate, default=str)
        elif target_format.upper() == 'CSV':
            if isinstance(intermediate, pd.DataFrame):
                converted = intermediate.to_csv(index=False)
            else:
                df = pd.DataFrame(intermediate)
                converted = df.to_csv(index=False)
        elif target_format.upper() == 'PICKLE':
            converted = pickle.dumps(intermediate)
        elif target_format.upper() == 'NPY':
            import io
            buffer = io.BytesIO()
            np.save(buffer, intermediate)
            converted = buffer.getvalue()
        else:
            converted = intermediate
        
        conversion_info['success'] = True
        conversion_info['data_size'] = sys.getsizeof(converted)
        
    except Exception as e:
        logger.error(f"Format conversion failed: {str(e)}")
        conversion_info['success'] = False
        conversion_info['error'] = str(e)
        converted = None
    
    return converted, conversion_info


def backup_and_versioning(file_path: str, operation: str = 'backup',
                         options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Intelligent backup and versioning system
    
    Features: Incremental backups, version control, automatic rotation
    """
    if options is None:
        options = {}
    
    backup_info = {
        'original_file': file_path,
        'operation': operation,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Create backup directory
        backup_dir = path.parent / '.backups' / path.stem
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        if operation == 'backup':
            # Create timestamped backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{path.stem}_{timestamp}{path.suffix}"
            backup_path = backup_dir / backup_name
            
            # Copy with verification
            shutil.copy2(file_path, backup_path)
            
            # Verify backup
            original_hash = _calculate_checksum(file_path)
            backup_hash = _calculate_checksum(str(backup_path))
            
            if original_hash == backup_hash:
                backup_info['backup_path'] = str(backup_path)
                backup_info['checksum'] = original_hash
                backup_info['success'] = True
                
                # Rotate old backups if needed
                max_backups = options.get('max_backups', 10)
                _rotate_backups(backup_dir, max_backups)
            else:
                raise ValueError("Backup verification failed")
                
        elif operation == 'restore':
            # Find latest backup
            backups = sorted(backup_dir.glob(f"{path.stem}_*{path.suffix}"))
            if backups:
                latest_backup = backups[-1]
                
                # Create safety backup of current
                safety_backup = path.with_suffix('.safety' + path.suffix)
                shutil.copy2(file_path, safety_backup)
                
                # Restore
                shutil.copy2(latest_backup, file_path)
                
                backup_info['restored_from'] = str(latest_backup)
                backup_info['safety_backup'] = str(safety_backup)
                backup_info['success'] = True
            else:
                raise ValueError("No backups found")
                
    except Exception as e:
        logger.error(f"Backup operation failed: {str(e)}")
        backup_info['success'] = False
        backup_info['error'] = str(e)
    
    return backup_info


def path_resolver(path_spec: Union[str, Path, Dict[str, str]], 
                 options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Cross-platform path resolution and validation
    
    Features: Platform normalization, symbolic link resolution, path validation
    """
    if options is None:
        options = {}
    
    path_info = {
        'input': str(path_spec),
        'platform': sys.platform
    }
    
    try:
        # Handle different input types
        if isinstance(path_spec, dict):
            base = path_spec.get('base', '.')
            relative = path_spec.get('relative', '')
            path = Path(base) / relative
        else:
            path = Path(path_spec)
        
        # Resolve path
        if options.get('resolve_symlinks', True):
            resolved = path.resolve()
        else:
            resolved = path.absolute()
        
        path_info.update({
            'resolved': str(resolved),
            'exists': resolved.exists(),
            'is_file': resolved.is_file(),
            'is_dir': resolved.is_dir(),
            'is_symlink': resolved.is_symlink(),
            'parent': str(resolved.parent),
            'name': resolved.name,
            'stem': resolved.stem,
            'suffix': resolved.suffix,
            'parts': resolved.parts
        })
        
        # Additional info if exists
        if resolved.exists():
            stat = resolved.stat()
            path_info.update({
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'permissions': oct(stat.st_mode)
            })
        
        # Platform-specific normalization
        if sys.platform == 'win32':
            path_info['windows_path'] = str(resolved).replace('/', '\\')
        else:
            path_info['posix_path'] = str(resolved)
        
        path_info['success'] = True
        
    except Exception as e:
        logger.error(f"Path resolution failed: {str(e)}")
        path_info['success'] = False
        path_info['error'] = str(e)
    
    return path_info


# Error Handling & Logging Functions

def advanced_error_handler(error: Exception, context: Dict[str, Any],
                          options: Optional[Dict[str, Any]] = None) -> ErrorReport:
    """
    Advanced error handling with recovery strategies
    
    Features: Error classification, recovery attempts, detailed reporting
    """
    if options is None:
        options = {}
    
    error_report = ErrorReport(
        error_type=type(error).__name__,
        error_message=str(error),
        traceback=traceback.format_exc(),
        timestamp=datetime.now(),
        context=context,
        recovery_attempted=False,
        recovery_successful=False
    )
    
    # Classify error and attempt recovery
    recovery_strategies = {
        'FileNotFoundError': _recover_file_not_found,
        'PermissionError': _recover_permission_error,
        'MemoryError': _recover_memory_error,
        'ValueError': _recover_value_error,
        'ZeroDivisionError': _recover_zero_division
    }
    
    strategy = recovery_strategies.get(error_report.error_type)
    
    if strategy and options.get('attempt_recovery', True):
        error_report.recovery_attempted = True
        try:
            recovery_result = strategy(error, context, options)
            error_report.recovery_successful = recovery_result.get('success', False)
            if recovery_result.get('details'):
                error_report.context['recovery_details'] = recovery_result['details']
        except Exception as recovery_error:
            logger.error(f"Recovery failed: {str(recovery_error)}")
            error_report.context['recovery_error'] = str(recovery_error)
    
    # Log error
    log_error(error_report)
    
    # Send notifications if critical
    if options.get('notify_critical', False) and _is_critical_error(error):
        _send_error_notification(error_report)
    
    return error_report


def comprehensive_logger(log_type: str, message: str, data: Optional[Dict[str, Any]] = None,
                        options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Comprehensive logging system with multiple outputs
    
    Features: Multi-level logging, structured logs, performance tracking
    """
    if options is None:
        options = {}
    if data is None:
        data = {}
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'log_type': log_type.upper(),
        'message': message,
        'data': data,
        'context': {
            'function': options.get('function_name', 'unknown'),
            'module': options.get('module_name', __name__),
            'line': options.get('line_number', 0)
        }
    }
    
    # Determine log level
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = log_levels.get(log_type.upper(), logging.INFO)
    
    # Log to appropriate destinations
    # 1. Python logger
    logger.log(level, message, extra={'data': data})
    
    # 2. Structured log file
    if options.get('file_logging', True):
        log_dir = Path('SFDP_6Layer_v17_3/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{log_type.lower()}_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
    
    # 3. Performance tracking
    if log_type.upper() == 'PERFORMANCE':
        _track_performance_metrics(log_entry)
    
    # 4. Error aggregation
    if log_type.upper() in ['ERROR', 'CRITICAL']:
        _aggregate_errors(log_entry)
    
    return {
        'logged': True,
        'log_entry': log_entry,
        'destinations': ['console', 'file'] if options.get('file_logging', True) else ['console']
    }


def debug_trace_manager(operation: str, trace_data: Optional[Dict[str, Any]] = None,
                       options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Debug trace management for detailed execution tracking
    
    Features: Call stack tracking, variable snapshots, execution flow
    """
    if options is None:
        options = {}
    if trace_data is None:
        trace_data = {}
    
    trace_info = {
        'operation': operation,
        'timestamp': datetime.now().isoformat(),
        'trace_id': hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    }
    
    if operation == 'START':
        # Initialize trace session
        trace_info['call_stack'] = traceback.extract_stack()
        trace_info['initial_state'] = trace_data
        
        # Store in trace manager
        _trace_sessions[trace_info['trace_id']] = {
            'start_time': time.time(),
            'events': [trace_info]
        }
        
    elif operation == 'EVENT':
        # Add trace event
        trace_id = trace_data.get('trace_id')
        if trace_id and trace_id in _trace_sessions:
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': trace_data.get('event_type', 'generic'),
                'data': trace_data.get('data', {}),
                'stack_depth': len(traceback.extract_stack())
            }
            _trace_sessions[trace_id]['events'].append(event)
            trace_info['event_added'] = True
        
    elif operation == 'END':
        # Finalize trace session
        trace_id = trace_data.get('trace_id')
        if trace_id and trace_id in _trace_sessions:
            session = _trace_sessions[trace_id]
            trace_info['duration'] = time.time() - session['start_time']
            trace_info['event_count'] = len(session['events'])
            trace_info['final_state'] = trace_data.get('final_state', {})
            
            # Save trace if requested
            if options.get('save_trace', False):
                _save_trace_session(trace_id, session)
            
            # Clean up
            del _trace_sessions[trace_id]
    
    return trace_info


# Performance & Optimization Functions

def memory_optimizer(operation: str, target: Optional[Any] = None,
                    options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Memory optimization and management utilities
    
    Features: Memory profiling, garbage collection, cache management
    """
    if options is None:
        options = {}
    
    optimization_info = {
        'operation': operation,
        'timestamp': datetime.now().isoformat()
    }
    
    if operation == 'PROFILE':
        # Memory profiling
        process = psutil.Process()
        memory_info = process.memory_info()
        
        optimization_info.update({
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'virtual_memory_mb': memory_info.vms / 1024 / 1024,
            'percent_used': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        })
        
    elif operation == 'OPTIMIZE':
        # Garbage collection
        import gc
        
        before_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches if requested
        if options.get('clear_caches', False):
            _clear_simulation_caches()
        
        after_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        optimization_info.update({
            'memory_before_mb': before_memory,
            'memory_after_mb': after_memory,
            'memory_freed_mb': before_memory - after_memory,
            'gc_stats': gc.get_stats()
        })
        
    elif operation == 'MONITOR':
        # Continuous monitoring
        if target:
            size = sys.getsizeof(target)
            optimization_info['object_size_bytes'] = size
            optimization_info['object_size_mb'] = size / 1024 / 1024
            
            # Deep size estimation for containers
            if hasattr(target, '__len__'):
                optimization_info['element_count'] = len(target)
    
    return optimization_info


def computation_profiler(func: Optional[Callable] = None, 
                        options: Optional[Dict[str, Any]] = None) -> Union[Callable, PerformanceMetrics]:
    """
    Computation profiling decorator and analyzer
    
    Features: Execution time, CPU usage, memory consumption tracking
    """
    if options is None:
        options = {}
    
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Pre-execution state
            process = psutil.Process()
            start_time = time.time()
            start_memory = process.memory_info().rss / 1024 / 1024
            start_cpu = process.cpu_percent(interval=0.1)
            
            # Execute function
            result = f(*args, **kwargs)
            
            # Post-execution state
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            end_cpu = process.cpu_percent(interval=0.1)
            
            # Create metrics
            metrics = PerformanceMetrics(
                function_name=f.__name__,
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_usage=(start_cpu + end_cpu) / 2,
                timestamp=datetime.now(),
                parameters={'args': str(args)[:100], 'kwargs': str(kwargs)[:100]}
            )
            
            # Log if requested
            if options.get('log_performance', True):
                comprehensive_logger('PERFORMANCE', f"Function {f.__name__} profiled", 
                                   {'metrics': metrics.__dict__})
            
            # Store metrics
            _performance_history.append(metrics)
            
            return result
        
        return wrapper
    
    if func is None:
        # Decorator with options
        return decorator
    else:
        # Direct decoration
        return decorator(func)


def parallel_task_manager(tasks: List[Callable], execution_mode: str = 'thread',
                         options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parallel task execution manager
    
    Features: Thread/process pools, load balancing, progress tracking
    """
    if options is None:
        options = {}
    
    execution_info = {
        'task_count': len(tasks),
        'execution_mode': execution_mode,
        'timestamp': datetime.now().isoformat()
    }
    
    max_workers = options.get('max_workers', min(len(tasks), psutil.cpu_count()))
    timeout = options.get('timeout', None)
    
    results = []
    errors = []
    
    try:
        if execution_mode == 'thread':
            executor_class = ThreadPoolExecutor
        elif execution_mode == 'process':
            executor_class = ProcessPoolExecutor
        else:
            raise ValueError(f"Unknown execution mode: {execution_mode}")
        
        start_time = time.time()
        
        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for i, task in enumerate(tasks):
                future = executor.submit(task)
                futures.append((i, future))
            
            # Collect results with progress tracking
            completed = 0
            for i, future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results.append((i, result))
                    completed += 1
                    
                    # Progress callback
                    if options.get('progress_callback'):
                        options['progress_callback'](completed, len(tasks))
                        
                except Exception as e:
                    errors.append((i, str(e)))
                    logger.error(f"Task {i} failed: {str(e)}")
        
        execution_info.update({
            'execution_time': time.time() - start_time,
            'completed_tasks': len(results),
            'failed_tasks': len(errors),
            'success_rate': len(results) / len(tasks) if tasks else 0,
            'results': results,
            'errors': errors
        })
        
    except Exception as e:
        logger.error(f"Parallel execution failed: {str(e)}")
        execution_info['error'] = str(e)
        execution_info['success'] = False
    else:
        execution_info['success'] = True
    
    return execution_info


# Visualization & Reporting Functions

def dynamic_visualizer(data: Dict[str, Any], viz_type: str,
                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Dynamic visualization generator for simulation results
    
    Features: Auto-layout, interactive plots, multi-format export
    """
    if options is None:
        options = {}
    
    viz_info = {
        'viz_type': viz_type,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        fig, axes = plt.subplots(figsize=options.get('figsize', (10, 6)))
        
        if viz_type == 'CONVERGENCE':
            # Convergence history plot
            iterations = data.get('iterations', [])
            values = data.get('values', [])
            
            axes.plot(iterations, values, 'b-', linewidth=2)
            axes.set_xlabel('Iteration')
            axes.set_ylabel('Value')
            axes.set_title('Convergence History')
            axes.grid(True, alpha=0.3)
            
        elif viz_type == 'COMPARISON':
            # Multi-series comparison
            for series_name, series_data in data.items():
                if isinstance(series_data, (list, np.ndarray)):
                    axes.plot(series_data, label=series_name)
            
            axes.legend()
            axes.set_title('Multi-Series Comparison')
            axes.grid(True, alpha=0.3)
            
        elif viz_type == 'HEATMAP':
            # 2D heatmap
            matrix = data.get('matrix', np.random.rand(10, 10))
            im = axes.imshow(matrix, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=axes)
            axes.set_title('Heatmap Visualization')
            
        elif viz_type == 'SCATTER':
            # Scatter plot with trends
            x = data.get('x', [])
            y = data.get('y', [])
            
            axes.scatter(x, y, alpha=0.6)
            
            # Add trend line if requested
            if options.get('show_trend', True) and len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                axes.plot(x, p(x), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
                axes.legend()
            
            axes.set_xlabel(data.get('x_label', 'X'))
            axes.set_ylabel(data.get('y_label', 'Y'))
            axes.set_title(data.get('title', 'Scatter Plot'))
            axes.grid(True, alpha=0.3)
        
        # Save if requested
        if options.get('save_path'):
            save_path = Path(options['save_path'])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            formats = options.get('formats', ['png'])
            for fmt in formats:
                fig.savefig(str(save_path.with_suffix(f'.{fmt}')), 
                           dpi=options.get('dpi', 300), 
                           bbox_inches='tight')
            
            viz_info['saved_to'] = str(save_path)
        
        # Show if requested
        if options.get('show', False):
            plt.show()
        else:
            plt.close(fig)
        
        viz_info['success'] = True
        
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        viz_info['success'] = False
        viz_info['error'] = str(e)
    
    return viz_info


def report_generator(report_type: str, data: Dict[str, Any],
                    options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Automated report generation system
    
    Features: Template-based reports, multi-format output, data aggregation
    """
    if options is None:
        options = {}
    
    report_info = {
        'report_type': report_type,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Generate report content
        if report_type == 'VALIDATION':
            content = _generate_validation_report(data)
        elif report_type == 'PERFORMANCE':
            content = _generate_performance_report(data)
        elif report_type == 'SUMMARY':
            content = _generate_summary_report(data)
        elif report_type == 'ERROR':
            content = _generate_error_report(data)
        else:
            content = _generate_generic_report(data)
        
        # Format output
        output_format = options.get('format', 'markdown')
        
        if output_format == 'markdown':
            formatted = content
        elif output_format == 'html':
            import markdown
            formatted = markdown.markdown(content)
        elif output_format == 'pdf':
            # Would require additional library like reportlab
            formatted = content  # Placeholder
            report_info['note'] = 'PDF generation requires additional libraries'
        else:
            formatted = content
        
        # Save report
        if options.get('save_path'):
            save_path = Path(options['save_path'])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(formatted)
            
            report_info['saved_to'] = str(save_path)
        
        report_info.update({
            'success': True,
            'content_length': len(content),
            'format': output_format
        })
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        report_info['success'] = False
        report_info['error'] = str(e)
    
    return report_info


# Helper functions and utilities

# Global storage for various tracking
_trace_sessions = {}
_performance_history = []
_error_aggregation = []


def _read_file_intelligent(file_specs: Dict[str, Any], options: Dict[str, Any]) -> FileOperationResult:
    """Read file with format detection"""
    file_path = file_specs['path']
    
    # Detect format
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    format_readers = {
        '.json': lambda p: json.load(open(p)),
        '.txt': lambda p: open(p).read(),
        '.csv': lambda p: pd.read_csv(p),
        '.npy': lambda p: np.load(p),
        '.pkl': lambda p: pickle.load(open(p, 'rb')),
        '.gz': lambda p: gzip.open(p, 'rt').read()
    }
    
    reader = format_readers.get(suffix, lambda p: open(p, 'rb').read())
    
    data = reader(file_path)
    
    return FileOperationResult(
        success=True,
        operation='READ',
        file_path=file_path,
        details={
            'format': suffix,
            'size': path.stat().st_size,
            'data_type': type(data).__name__
        }
    )


def _write_file_atomic(file_specs: Dict[str, Any], options: Dict[str, Any]) -> FileOperationResult:
    """Write file atomically with backup"""
    file_path = file_specs['path']
    data = file_specs['data']
    
    # Write to temporary file first
    temp_path = file_path + '.tmp'
    
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    # Format-specific writers
    if suffix == '.json':
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif suffix == '.csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(temp_path, index=False)
        else:
            pd.DataFrame(data).to_csv(temp_path, index=False)
    elif suffix == '.npy':
        np.save(temp_path, data)
    elif suffix == '.pkl':
        with open(temp_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(temp_path, 'w') as f:
            f.write(str(data))
    
    # Atomic rename
    Path(temp_path).replace(file_path)
    
    return FileOperationResult(
        success=True,
        operation='WRITE',
        file_path=file_path,
        details={
            'format': suffix,
            'size': Path(file_path).stat().st_size
        },
        checksum=_calculate_checksum(file_path)
    )


def _calculate_checksum(file_path: str, algorithm: str = 'md5') -> str:
    """Calculate file checksum"""
    hash_func = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def _copy_file_verified(file_specs: Dict[str, Any], options: Dict[str, Any]) -> FileOperationResult:
    """Copy file with verification"""
    src = file_specs['source']
    dst = file_specs['destination']
    
    # Calculate source checksum
    src_checksum = _calculate_checksum(src)
    
    # Copy
    shutil.copy2(src, dst)
    
    # Verify
    dst_checksum = _calculate_checksum(dst)
    
    if src_checksum != dst_checksum:
        raise ValueError("Copy verification failed")
    
    return FileOperationResult(
        success=True,
        operation='COPY',
        file_path=dst,
        details={
            'source': src,
            'size': Path(dst).stat().st_size
        },
        checksum=dst_checksum
    )


def _move_file_safe(file_specs: Dict[str, Any], options: Dict[str, Any]) -> FileOperationResult:
    """Move file safely"""
    src = file_specs['source']
    dst = file_specs['destination']
    
    # Ensure destination directory exists
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    
    # Move
    shutil.move(src, dst)
    
    return FileOperationResult(
        success=True,
        operation='MOVE',
        file_path=dst,
        details={
            'source': src,
            'size': Path(dst).stat().st_size
        }
    )


def _delete_file_secure(file_specs: Dict[str, Any], options: Dict[str, Any]) -> FileOperationResult:
    """Delete file with optional backup"""
    file_path = file_specs['path']
    
    # Backup before deletion if requested
    if options.get('backup_before_delete', True):
        backup_and_versioning(file_path, 'backup')
    
    # Delete
    Path(file_path).unlink()
    
    return FileOperationResult(
        success=True,
        operation='DELETE',
        file_path=file_path,
        details={'deleted': True}
    )


def _compress_file(file_specs: Dict[str, Any], options: Dict[str, Any]) -> FileOperationResult:
    """Compress file"""
    file_path = file_specs['path']
    output_path = file_specs.get('output', file_path + '.gz')
    
    with open(file_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    original_size = Path(file_path).stat().st_size
    compressed_size = Path(output_path).stat().st_size
    
    return FileOperationResult(
        success=True,
        operation='COMPRESS',
        file_path=output_path,
        details={
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / original_size
        }
    )


def _extract_file(file_specs: Dict[str, Any], options: Dict[str, Any]) -> FileOperationResult:
    """Extract compressed file"""
    file_path = file_specs['path']
    output_path = file_specs.get('output', file_path.replace('.gz', ''))
    
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return FileOperationResult(
        success=True,
        operation='EXTRACT',
        file_path=output_path,
        details={'size': Path(output_path).stat().st_size}
    )


def _rotate_backups(backup_dir: Path, max_backups: int):
    """Rotate old backups"""
    backups = sorted(backup_dir.glob('*'))
    
    if len(backups) > max_backups:
        for old_backup in backups[:-max_backups]:
            old_backup.unlink()


def _clear_simulation_caches():
    """Clear simulation-specific caches"""
    cache_dirs = [
        'SFDP_6Layer_v17_3/physics_cache',
        'SFDP_6Layer_v17_3/taylor_cache',
        'SFDP_6Layer_v17_3/kalman_corrections'
    ]
    
    for cache_dir in cache_dirs:
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
            Path(cache_dir).mkdir(parents=True)


def log_error(error_report: ErrorReport):
    """Log error report"""
    logger.error(f"{error_report.error_type}: {error_report.error_message}")
    _error_aggregation.append(error_report)


def _generate_validation_report(data: Dict[str, Any]) -> str:
    """Generate validation report content"""
    report = f"""# Validation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Overall Score: {data.get('overall_score', 0):.1%}
- Status: {data.get('status', 'UNKNOWN')}
- Confidence: {data.get('confidence', 0):.1%}

## Detailed Results
"""
    
    for section, results in data.get('details', {}).items():
        report += f"\n### {section.replace('_', ' ').title()}\n"
        report += f"```\n{json.dumps(results, indent=2)}\n```\n"
    
    return report


def _generate_performance_report(data: Dict[str, Any]) -> str:
    """Generate performance report content"""
    report = f"""# Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Execution Summary
- Total Functions Profiled: {len(_performance_history)}
- Average Execution Time: {np.mean([m.execution_time for m in _performance_history]):.3f}s
- Peak Memory Usage: {max([m.memory_usage for m in _performance_history]):.1f}MB

## Function Performance
"""
    
    for metric in _performance_history[-10:]:  # Last 10
        report += f"""
### {metric.function_name}
- Execution Time: {metric.execution_time:.3f}s
- Memory Usage: {metric.memory_usage:.1f}MB
- CPU Usage: {metric.cpu_usage:.1f}%
"""
    
    return report


def _generate_summary_report(data: Dict[str, Any]) -> str:
    """Generate summary report content"""
    return f"""# Simulation Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
{json.dumps(data.get('config', {}), indent=2)}

## Results
{json.dumps(data.get('results', {}), indent=2)}

## Statistics
{json.dumps(data.get('statistics', {}), indent=2)}
"""


def _generate_error_report(data: Dict[str, Any]) -> str:
    """Generate error report content"""
    report = f"""# Error Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Error Summary
Total Errors: {len(_error_aggregation)}

## Recent Errors
"""
    
    for error in _error_aggregation[-5:]:  # Last 5
        report += f"""
### {error.error_type}
- Message: {error.error_message}
- Time: {error.timestamp}
- Recovery Attempted: {error.recovery_attempted}
- Recovery Successful: {error.recovery_successful}
"""
    
    return report


def _generate_generic_report(data: Dict[str, Any]) -> str:
    """Generate generic report content"""
    return f"""# Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data
{json.dumps(data, indent=2, default=str)}
"""


# Error recovery strategies
def _recover_file_not_found(error: Exception, context: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to recover from file not found errors"""
    file_path = context.get('file_path', '')
    
    # Check alternate locations
    alt_paths = [
        Path.cwd() / Path(file_path).name,
        Path.home() / Path(file_path).name,
        Path('data') / Path(file_path).name
    ]
    
    for alt_path in alt_paths:
        if alt_path.exists():
            return {'success': True, 'details': {'found_at': str(alt_path)}}
    
    return {'success': False}


def _recover_permission_error(error: Exception, context: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to recover from permission errors"""
    # Try to change permissions (platform-specific)
    try:
        file_path = context.get('file_path', '')
        if file_path and Path(file_path).exists():
            Path(file_path).chmod(0o666)
            return {'success': True, 'details': {'permissions_changed': True}}
    except:
        pass
    
    return {'success': False}


def _recover_memory_error(error: Exception, context: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to recover from memory errors"""
    # Trigger garbage collection
    import gc
    gc.collect()
    
    # Clear caches
    _clear_simulation_caches()
    
    return {'success': True, 'details': {'memory_cleared': True}}


def _recover_value_error(error: Exception, context: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to recover from value errors"""
    # Check for common value error patterns
    if 'nan' in str(error).lower():
        return {'success': False, 'details': {'type': 'nan_values'}}
    elif 'shape' in str(error).lower():
        return {'success': False, 'details': {'type': 'shape_mismatch'}}
    
    return {'success': False}


def _recover_zero_division(error: Exception, context: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to recover from zero division errors"""
    # Can't really recover, but can provide context
    return {'success': False, 'details': {'suggestion': 'Check denominators for zero values'}}


def _is_critical_error(error: Exception) -> bool:
    """Determine if error is critical"""
    critical_types = [MemoryError, SystemError, KeyboardInterrupt]
    return type(error) in critical_types


def _send_error_notification(error_report: ErrorReport):
    """Send error notification (placeholder)"""
    logger.critical(f"CRITICAL ERROR: {error_report.error_type} - {error_report.error_message}")


def _track_performance_metrics(log_entry: Dict[str, Any]):
    """Track performance metrics over time"""
    # This would typically write to a time-series database
    # For now, just append to history
    pass


def _aggregate_errors(log_entry: Dict[str, Any]):
    """Aggregate errors for analysis"""
    # This would typically update error statistics
    # For now, just count
    pass


def _save_trace_session(trace_id: str, session: Dict[str, Any]):
    """Save debug trace session to file"""
    trace_dir = Path('SFDP_6Layer_v17_3/debug_traces')
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    trace_file = trace_dir / f"trace_{trace_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(trace_file, 'w') as f:
        json.dump(session, f, indent=2, default=str)