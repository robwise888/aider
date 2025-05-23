"""
Utility module for displaying progress bars in the terminal.

This module provides functions for creating and updating progress bars
in the terminal, with support for different styles and customization.
"""

import sys
import time
import threading
from typing import Optional, Callable, Any

class ProgressBar:
    """
    A simple progress bar for terminal output.
    
    This class provides a progress bar that can be updated to show progress
    during long-running operations. It supports different styles and can
    be customized with different characters and colors.
    """
    
    def __init__(self, 
                 total: int = 100, 
                 prefix: str = 'Progress:', 
                 suffix: str = 'Complete', 
                 decimals: int = 1,
                 length: int = 50, 
                 fill: str = '█', 
                 print_end: str = '\r',
                 show_time: bool = True,
                 show_percent: bool = True):
        """
        Initialize the progress bar.
        
        Args:
            total: Total iterations
            prefix: Prefix string
            suffix: Suffix string
            decimals: Positive number of decimals in percent complete
            length: Character length of bar
            fill: Bar fill character
            print_end: End character (e.g. '\r', '\n')
            show_time: Whether to show elapsed time
            show_percent: Whether to show percentage
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.show_time = show_time
        self.show_percent = show_percent
        
        self.iteration = 0
        self.start_time = time.time()
        self.is_finished = False
    
    def update(self, iteration: Optional[int] = None, suffix: Optional[str] = None) -> None:
        """
        Update the progress bar.
        
        Args:
            iteration: Current iteration (if None, increment by 1)
            suffix: New suffix string (if None, use existing suffix)
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1
            
        if suffix is not None:
            self.suffix = suffix
            
        self._print()
    
    def finish(self, suffix: Optional[str] = None) -> None:
        """
        Finish the progress bar.
        
        Args:
            suffix: Final suffix string (if None, use existing suffix)
        """
        if suffix is not None:
            self.suffix = suffix
            
        self.iteration = self.total
        self.is_finished = True
        self._print()
        print()  # Add a newline at the end
    
    def _print(self) -> None:
        """Print the progress bar."""
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
        filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        time_str = f" {elapsed_time:.1f}s" if self.show_time else ""
        percent_str = f" {percent}%" if self.show_percent else ""
        
        # Print progress bar
        print(f'\r{self.prefix} |{bar}|{percent_str}{time_str} {self.suffix}', end=self.print_end)
        sys.stdout.flush()

class IndeterminateProgressBar:
    """
    An indeterminate progress bar for operations with unknown duration.
    
    This class provides a spinner or moving bar that indicates that an operation
    is in progress, but without showing a specific percentage of completion.
    """
    
    def __init__(self, 
                 prefix: str = 'Loading:', 
                 suffix: str = '', 
                 spinner_chars: str = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏',
                 print_end: str = '\r',
                 show_time: bool = True):
        """
        Initialize the indeterminate progress bar.
        
        Args:
            prefix: Prefix string
            suffix: Suffix string
            spinner_chars: Characters to use for the spinner
            print_end: End character (e.g. '\r', '\n')
            show_time: Whether to show elapsed time
        """
        self.prefix = prefix
        self.suffix = suffix
        self.spinner_chars = spinner_chars
        self.print_end = print_end
        self.show_time = show_time
        
        self.index = 0
        self.start_time = time.time()
        self.is_running = False
        self.thread = None
    
    def start(self) -> None:
        """Start the indeterminate progress bar."""
        self.is_running = True
        self.start_time = time.time()
        
        # Start a thread to update the spinner
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self, suffix: Optional[str] = None) -> None:
        """
        Stop the indeterminate progress bar.
        
        Args:
            suffix: Final suffix string (if None, use existing suffix)
        """
        if suffix is not None:
            self.suffix = suffix
            
        self.is_running = False
        if self.thread:
            self.thread.join()
            
        # Clear the line
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        
        # Print final message
        elapsed_time = time.time() - self.start_time
        time_str = f" ({elapsed_time:.1f}s)" if self.show_time else ""
        print(f"{self.prefix} {self.suffix}{time_str}")
        sys.stdout.flush()
    
    def _spin(self) -> None:
        """Update the spinner."""
        while self.is_running:
            char = self.spinner_chars[self.index % len(self.spinner_chars)]
            self.index += 1
            
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time
            time_str = f" {elapsed_time:.1f}s" if self.show_time else ""
            
            # Print spinner
            sys.stdout.write(f'\r{self.prefix} {char} {self.suffix}{time_str}')
            sys.stdout.flush()
            time.sleep(0.1)

def run_with_progress(func: Callable[..., Any], 
                     args: tuple = (), 
                     kwargs: dict = None,
                     prefix: str = 'Loading:',
                     suffix: str = 'Please wait...',
                     success_suffix: str = 'Complete!',
                     error_suffix: str = 'Failed!') -> Any:
    """
    Run a function with an indeterminate progress bar.
    
    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        prefix: Prefix string for the progress bar
        suffix: Suffix string for the progress bar
        success_suffix: Suffix string to show on success
        error_suffix: Suffix string to show on error
        
    Returns:
        The result of the function
    """
    if kwargs is None:
        kwargs = {}
        
    progress = IndeterminateProgressBar(prefix=prefix, suffix=suffix)
    progress.start()
    
    try:
        result = func(*args, **kwargs)
        progress.stop(suffix=success_suffix)
        return result
    except Exception as e:
        progress.stop(suffix=f"{error_suffix} ({str(e)})")
        raise
